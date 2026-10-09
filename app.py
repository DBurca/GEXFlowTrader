"""
Three-panel intraday 0DTE dashboard.

Run a full simulated session without Schwab credentials:

    python app.py --simulate

Live mode polls Schwab with the limits in ``schwab_client``:

    python app.py --symbol SPY
"""

from __future__ import annotations

import argparse
import threading
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from plotly.subplots import make_subplots

import gex_engine
import ta_engine
from schwab_client import (
    SPOT_POLL_SECONDS,
    EmptyOptionChainError,
    MarketClosedError,
    NoSessionBarsError,
    SchwabIntradayClient,
    generate_simulated_session,
    in_regular_trading_hours,
)

EASTERN = ZoneInfo("America/New_York")
PRICE_COLOR = "#F4F7FB"
CALL_WALL_COLOR = "#3DDC97"
PUT_WALL_COLOR = "#FF6B8A"
ZERO_GAMMA_COLOR = "#E2C15A"
VWAP_COLOR = "#8EB4FF"
BULLISH_COLOR = "#2ECC71"
BEARISH_COLOR = "#E74C3C"
IV_COLOR = "#C084FC"
POSITIVE_SHADE = "rgba(46, 204, 113, 0.30)"
NEGATIVE_SHADE = "rgba(231, 76, 60, 0.30)"
FLAT_SHADE = "rgba(160, 170, 180, 0.08)"
PAPER = "#0b0e14"
PLOT = "#11161f"
GRID = "#243044"


@dataclass
class DashboardView:
    symbol: str
    simulate: bool
    bars: pd.DataFrame
    metrics: pd.DataFrame
    error: str | None
    notice: str | None
    refreshed_at: datetime | None


class MarketFeed:
    """Owns bars and the chronological options snapshot history."""

    def __init__(self, symbol: str = "SPY", *, simulate: bool = False, seed: int = 11) -> None:
        self.symbol = symbol.upper()
        self.simulate = simulate
        self.seed = seed
        self.client = None if simulate else SchwabIntradayClient()
        self.bars = pd.DataFrame()
        self.metrics = gex_engine.empty_snapshot_history()
        self.error: str | None = None
        self.notice: str | None = None
        self.refreshed_at: datetime | None = None
        self._rng = np.random.default_rng(seed + 99)
        self._lock = threading.Lock()

    def bootstrap(self) -> None:
        with self._lock:
            if self.simulate:
                self._load_simulation()
            else:
                self._poll_live(force=True)
            self.refreshed_at = datetime.now(EASTERN)

    def tick(self) -> None:
        with self._lock:
            if self.simulate:
                self._nudge_last_bar()
            else:
                self._poll_live(force=False)
            self.refreshed_at = datetime.now(EASTERN)

    def view(self) -> DashboardView:
        with self._lock:
            return DashboardView(
                symbol=self.symbol,
                simulate=self.simulate,
                bars=self.bars.copy(),
                metrics=self.metrics.copy(),
                error=self.error,
                notice=self.notice,
                refreshed_at=self.refreshed_at,
            )

    def _load_simulation(self) -> None:
        session = generate_simulated_session(self.symbol, seed=self.seed)
        self.bars = session.bars
        self.metrics = gex_engine.build_snapshot_history(session.chains)
        self.error = None
        self.notice = (
            "Simulated regular-hours session. No Schwab requests are being sent."
        )

    def _nudge_last_bar(self) -> None:
        if self.bars.empty:
            return
        last = self.bars.index[-1]
        price = float(self.bars.at[last, "Close"])
        price = max(price + float(self._rng.normal(0.0, price * 0.0002)), 0.01)
        self.bars.at[last, "Close"] = price
        self.bars.at[last, "High"] = max(float(self.bars.at[last, "High"]), price)
        self.bars.at[last, "Low"] = min(float(self.bars.at[last, "Low"]), price)

    def _poll_live(self, *, force: bool) -> None:
        assert self.client is not None
        try:
            self.bars = self.client.fetch_intraday_bars(self.symbol, force=force)
            self.notice = None
        except NoSessionBarsError as exc:
            self.notice = str(exc)
        except Exception as exc:
            self.error = str(exc)
            return

        if not in_regular_trading_hours():
            self.notice = (
                "Outside regular trading hours (09:30–16:00 ET). "
                "Option-chain polling is paused."
            )
            return

        if self.bars.empty:
            return

        spot = float(self.bars["Close"].iloc[-1])
        try:
            fetched = self.client.fetch_0dte_chain(
                self.symbol,
                spot=spot,
                force=force,
            )
            if fetched.refreshed:
                snapshot = gex_engine.build_intraday_snapshot(
                    fetched.contracts,
                    fetched.spot,
                    datetime.now(EASTERN),
                )
                self.metrics = gex_engine.append_snapshot(self.metrics, snapshot)
            self.error = None
        except MarketClosedError as exc:
            self.notice = str(exc)
        except EmptyOptionChainError as exc:
            self.error = str(exc)
        except Exception as exc:
            self.error = str(exc)


def build_dashboard_figure(view: DashboardView) -> go.Figure:
    """Three time-aligned panels: price structure, delta flow, and ATM IV."""
    if view.bars.empty:
        return _empty_figure(
            view.notice or view.error or "Waiting for regular-hours 1-minute bars."
        )

    try:
        session = ta_engine.calculate_vwap_bands(view.bars)
    except ValueError as exc:
        return _empty_figure(str(exc))

    session = _clock_frame(session)
    metrics = _clock_frame(view.metrics) if not view.metrics.empty else view.metrics
    if not metrics.empty and "zero_gamma" in metrics.columns:
        metrics["zero_gamma"] = metrics["zero_gamma"].ffill()
    x_end = session.index[-1]

    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.48, 0.26, 0.26],
        subplot_titles=(
            "Price & Structural Walls",
            "Order Flow",
            "Friction & Regime",
        ),
    )

    _add_bands(figure, session)
    figure.add_trace(
        go.Scatter(
            x=session.index,
            y=session["vwap"],
            name="VWAP",
            mode="lines",
            line=dict(color=VWAP_COLOR, width=1.6),
            hovertemplate="VWAP %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=session.index,
            y=session["Close"],
            name="Spot",
            mode="lines",
            line=dict(color=PRICE_COLOR, width=2.2),
            hovertemplate="Spot %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=[session.index[-1]],
            y=[session["Close"].iloc[-1]],
            mode="markers",
            marker=dict(size=8, color=PRICE_COLOR),
            showlegend=False,
            hovertemplate="Last %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    if not metrics.empty:
        _add_level(figure, metrics, "call_wall", "Call Wall", CALL_WALL_COLOR, "solid", x_end)
        _add_level(figure, metrics, "put_wall", "Put Wall", PUT_WALL_COLOR, "solid", x_end)
        _add_level(
            figure,
            metrics,
            "zero_gamma",
            "Zero Gamma",
            ZERO_GAMMA_COLOR,
            "dash",
            x_end,
        )
        _add_delta_flow(figure, metrics, x_end)
        _add_atm_iv(figure, metrics, x_end)
        _shade_gex_regime(figure, metrics, x_end)

    _style_figure(figure, session)
    return figure


def _add_bands(figure: go.Figure, session: pd.DataFrame) -> None:
    figure.add_trace(
        go.Scatter(
            x=session.index,
            y=session["upper_2"],
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=session.index,
            y=session["lower_2"],
            name="2σ Band",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(186, 196, 214, 0.10)",
            hovertemplate="2σ %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=session.index,
            y=session["upper_1"],
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=session.index,
            y=session["lower_1"],
            name="1σ Band",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(186, 196, 214, 0.18)",
            hovertemplate="1σ %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )


def _add_level(
    figure: go.Figure,
    metrics: pd.DataFrame,
    column: str,
    name: str,
    color: str,
    dash: str,
    x_end: pd.Timestamp,
) -> None:
    series = metrics[column]
    if not np.isfinite(series.to_numpy(dtype=float)).any():
        return
    x_values, y_values = _hold_to(series.index, series, x_end)
    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            name=name,
            mode="lines",
            line=dict(color=color, width=1.7, dash=dash, shape="hv"),
            hovertemplate=f"{name} %{{y:.2f}}<extra></extra>",
            connectgaps=False,
        ),
        row=1,
        col=1,
    )


def _add_delta_flow(figure: go.Figure, metrics: pd.DataFrame, x_end: pd.Timestamp) -> None:
    series = metrics["cdf"].astype(float)
    extended = series.copy()
    if extended.index[-1] < x_end:
        extended.loc[x_end] = extended.iloc[-1]
    times = list(extended.index)
    values = extended.to_numpy(dtype=float)
    bull_x, bull_y = _signed_points(times, values, positive=True)
    bear_x, bear_y = _signed_points(times, values, positive=False)
    figure.add_trace(
        go.Scatter(
            x=bull_x,
            y=bull_y,
            name="Bullish Flow",
            mode="lines",
            line=dict(color=BULLISH_COLOR, width=2),
            fill="tozeroy",
            fillcolor="rgba(46, 204, 113, 0.22)",
            hovertemplate="CDF %{y:,.0f}<extra></extra>",
            connectgaps=False,
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=bear_x,
            y=bear_y,
            name="Bearish Flow",
            mode="lines",
            line=dict(color=BEARISH_COLOR, width=2),
            fill="tozeroy",
            fillcolor="rgba(231, 76, 60, 0.22)",
            hovertemplate="CDF %{y:,.0f}<extra></extra>",
            connectgaps=False,
        ),
        row=2,
        col=1,
    )
    figure.add_hline(
        y=0,
        line_width=1,
        line_color="rgba(215, 221, 232, 0.45)",
        row=2,
        col=1,
    )


def _add_atm_iv(figure: go.Figure, metrics: pd.DataFrame, x_end: pd.Timestamp) -> None:
    series = metrics["atm_iv"].astype(float)
    if not np.isfinite(series.to_numpy(dtype=float)).any():
        return
    x_values, y_values = _hold_to(series.index, series, x_end)
    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            name="ATM IV",
            mode="lines",
            line=dict(color=IV_COLOR, width=2),
            hovertemplate="ATM IV %{y:.2%}<extra></extra>",
        ),
        row=3,
        col=1,
    )


def _shade_gex_regime(figure: go.Figure, metrics: pd.DataFrame, x_end: pd.Timestamp) -> None:
    signed = np.sign(metrics["net_gex"].to_numpy(dtype=float))
    signed = np.where(np.isfinite(signed), signed, 0.0)
    start = 0
    times = list(metrics.index)
    for index in range(1, len(signed) + 1):
        if index == len(signed) or signed[index] != signed[start]:
            x0 = times[start]
            x1 = times[index] if index < len(signed) else x_end
            if pd.Timestamp(x1) > pd.Timestamp(x0):
                figure.add_vrect(
                    x0=x0,
                    x1=x1,
                    fillcolor=_shade_color(signed[start]),
                    opacity=1,
                    line_width=0,
                    layer="below",
                    row=3,
                    col=1,
                )
            start = index


def _shade_color(sign: float) -> str:
    if sign > 0:
        return POSITIVE_SHADE
    if sign < 0:
        return NEGATIVE_SHADE
    return FLAT_SHADE


def _style_figure(figure: go.Figure, session: pd.DataFrame) -> None:
    day = pd.Timestamp(session.index[0]).date()
    x0 = pd.Timestamp(day) + pd.Timedelta(hours=9, minutes=30)
    x1 = pd.Timestamp(day) + pd.Timedelta(hours=16)
    lows = [
        float(session["Low"].min()),
        float(session["lower_2"].min()),
    ]
    highs = [
        float(session["High"].max()),
        float(session["upper_2"].max()),
    ]
    finite_levels = []
    for trace in figure.data:
        if trace.name in {"Call Wall", "Put Wall", "Zero Gamma"} and trace.y is not None:
            finite_levels.extend(value for value in trace.y if isinstance(value, (int, float)))
    if finite_levels:
        lows.append(min(finite_levels))
        highs.append(max(finite_levels))
    pad = max((max(highs) - min(lows)) * 0.06, 0.25)

    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor=PAPER,
        plot_bgcolor=PLOT,
        hovermode="x unified",
        uirevision="rth-session",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.08,
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color="#d7dde8"),
        ),
        margin=dict(l=64, r=24, t=28, b=72),
        font=dict(color="#d7dde8"),
    )
    figure.update_xaxes(
        range=[x0, x1],
        tickformat="%H:%M",
        dtick=30 * 60 * 1000,
        showgrid=True,
        gridcolor=GRID,
        zeroline=False,
        title_text="",
    )
    figure.update_xaxes(title_text="Time (ET)", row=3, col=1)
    figure.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    figure.update_yaxes(title_text="Price", range=[min(lows) - pad, max(highs) + pad], row=1, col=1)
    figure.update_yaxes(title_text="Delta flow", row=2, col=1)
    figure.update_yaxes(title_text="ATM IV", tickformat=".1%", row=3, col=1)
    for annotation in figure.layout.annotations:
        annotation.font = dict(size=13, color="#c5d0e0")


def _empty_figure(message: str) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor=PAPER,
        plot_bgcolor=PLOT,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[
            dict(
                text=message,
                showarrow=False,
                font=dict(size=16, color="#d7dde8"),
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
            )
        ],
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return figure


def _clock_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop timezone info after converting to Eastern clock time for Plotly."""
    copied = frame.copy()
    index = pd.DatetimeIndex(copied.index)
    if index.tz is not None:
        index = index.tz_convert(EASTERN).tz_localize(None)
    copied.index = index
    return copied


def _hold_to(
    index: pd.DatetimeIndex,
    values: pd.Series,
    until: pd.Timestamp,
) -> tuple[list, list]:
    x_values = list(index)
    y_values: list[float | None] = [
        None if pd.isna(value) else float(value) for value in values
    ]
    if x_values and pd.Timestamp(x_values[-1]) < pd.Timestamp(until):
        x_values.append(until)
        y_values.append(y_values[-1])
    return x_values, y_values


def _signed_points(
    times: list,
    values: np.ndarray,
    *,
    positive: bool,
) -> tuple[list, list]:
    x_values: list = []
    y_values: list = []

    def add(moment: object, value: float) -> None:
        x_values.append(moment)
        y_values.append(value)

    def split() -> None:
        x_values.append(None)
        y_values.append(None)

    for index, value in enumerate(values):
        if index:
            previous = float(values[index - 1])
            current = float(value)
            if previous * current < 0:
                left = pd.Timestamp(times[index - 1])
                right = pd.Timestamp(times[index])
                fraction = previous / (previous - current)
                crossed = left + (right - left) * fraction
                add(crossed, 0.0)
                if (previous > 0) == positive:
                    split()
        if value == 0 or (value > 0 and positive) or (value < 0 and not positive):
            add(times[index], float(value))
    return x_values, y_values


def _format_gex(value: float) -> str:
    sign = "+" if value > 0 else ""
    magnitude = abs(value)
    if magnitude >= 1e9:
        return f"{sign}{value / 1e9:.2f}B"
    if magnitude >= 1e6:
        return f"{sign}{value / 1e6:.2f}M"
    if magnitude >= 1e3:
        return f"{sign}{value / 1e3:.1f}k"
    return f"{sign}{value:,.0f}"


def _latest_finite(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    values = frame[column].dropna()
    if values.empty:
        return None
    value = float(values.iloc[-1])
    if not np.isfinite(value):
        return None
    return value


def _format_level(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.2f}"


def _header(view: DashboardView) -> html.Div:
    spot = float(view.bars["Close"].iloc[-1]) if not view.bars.empty else float("nan")
    latest = view.metrics.iloc[-1] if not view.metrics.empty else None
    net_gex = float(latest["net_gex"]) if latest is not None else float("nan")
    if np.isfinite(net_gex) and net_gex > 0:
        regime = "Positive gamma"
        regime_kind = "pos"
    elif np.isfinite(net_gex) and net_gex < 0:
        regime = "Negative gamma"
        regime_kind = "neg"
    else:
        regime = "Flat gamma"
        regime_kind = ""

    as_of = view.metrics.index[-1] if not view.metrics.empty else None
    if as_of is not None:
        stamp = pd.Timestamp(as_of)
        if stamp.tzinfo is not None:
            stamp = stamp.tz_convert(EASTERN)
        weight = "Open interest" if stamp.time() <= gex_engine.OI_WEIGHT_UNTIL else "40% OI + 60% volume"
    else:
        weight = "—"

    session_day = datetime.now(EASTERN)
    if not view.bars.empty:
        bar_time = pd.Timestamp(view.bars.index[-1])
        if bar_time.tzinfo is not None:
            bar_time = bar_time.tz_convert(EASTERN)
        session_day = bar_time.to_pydatetime()

    refreshed = ""
    if view.refreshed_at is not None:
        refreshed = view.refreshed_at.astimezone(EASTERN).strftime("%H:%M:%S ET")

    pills = [
        html.Span(
            "SIMULATED" if view.simulate else "LIVE",
            className="pill sim" if view.simulate else "pill live",
        ),
        _pill("Spot", f"{spot:.2f}" if np.isfinite(spot) else "—"),
        _pill("Net GEX", _format_gex(net_gex) if np.isfinite(net_gex) else "—", regime_kind),
        _pill("Regime", regime, regime_kind),
        _pill("Call wall", _format_level(None if latest is None else latest["call_wall"])),
        _pill("Put wall", _format_level(None if latest is None else latest["put_wall"])),
        _pill("Zero gamma", _format_level(_latest_finite(view.metrics, "zero_gamma"))),
        _pill(
            "ATM IV",
            f"{float(latest['atm_iv']):.1%}" if latest is not None and np.isfinite(latest["atm_iv"]) else "—",
        ),
        _pill("Weight", weight),
        _pill("Updated", refreshed or "—"),
    ]
    return html.Div(
        [
            html.Div(
                [
                    html.Div("0DTE STRUCTURE", className="eyebrow"),
                    html.H1(f"{view.symbol}  ·  {session_day.strftime('%a %d %b %Y')}"),
                ]
            ),
            html.Div(pills, className="pills"),
        ],
        className="header",
    )


def _pill(label: str, value: str, kind: str = "") -> html.Span:
    class_name = f"pill {kind}".strip()
    return html.Span(
        [html.Span(f"{label} ", className="muted"), html.Strong(value)],
        className=class_name,
    )


def _banner(view: DashboardView) -> html.Div:
    children = []
    if view.error:
        children.append(html.Div(view.error, className="error"))
    if view.notice:
        children.append(html.Div(view.notice, className="notice"))
    return html.Div(children, className="banner")


INDEX_STRING = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                margin: 0;
                background: #0b0e14;
                color: #d7dde8;
                font-family: "Segoe UI", Helvetica, Arial, sans-serif;
            }
            .page { padding: 14px 18px 10px; }
            .header {
                display: flex;
                flex-direction: column;
                align-items: flex-start;
                gap: 8px;
                margin-bottom: 10px;
            }
            h1 { margin: 0; font-size: 20px; font-weight: 600; }
            .eyebrow {
                color: #8ea0b8;
                font-size: 11px;
                letter-spacing: 0.14em;
                margin-bottom: 2px;
            }
            .pills {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                justify-content: flex-start;
            }
            .pill {
                background: #171d29;
                border: 1px solid #2a3548;
                border-radius: 999px;
                padding: 4px 10px;
                font-size: 12px;
            }
            .pill.live { color: #7dffa8; border-color: #1f6b45; }
            .pill.sim { color: #ffd37a; border-color: #6b531f; }
            .pill.pos { color: #8ee0b2; }
            .pill.neg { color: #ff9aa8; }
            .muted { color: #93a0b5; }
            .error, .notice {
                border-radius: 8px;
                padding: 8px 12px;
                margin-bottom: 8px;
                font-size: 13px;
            }
            .error { background: #3a1720; color: #ffb4c0; }
            .notice { background: #2a2416; color: #f0d48a; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""


def create_app(feed: MarketFeed) -> Dash:
    app = Dash(__name__)
    app.title = f"{feed.symbol} 0DTE Dashboard"
    app.index_string = INDEX_STRING
    initial = feed.view()
    interval_ms = 4_000 if feed.simulate else SPOT_POLL_SECONDS * 1000
    app.layout = html.Div(
        [
            html.Div(id="header", children=_header(initial)),
            html.Div(id="banner", children=_banner(initial)),
            dcc.Graph(
                id="dashboard",
                figure=build_dashboard_figure(initial),
                style={"height": "calc(100vh - 168px)", "width": "100%"},
                config={
                    "displaylogo": False,
                    "scrollZoom": True,
                    "responsive": True,
                    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                },
            ),
            dcc.Interval(id="tick", interval=interval_ms, n_intervals=0),
        ],
        className="page",
    )

    @app.callback(
        Output("header", "children"),
        Output("banner", "children"),
        Output("dashboard", "figure"),
        Input("tick", "n_intervals"),
    )
    def refresh(_n_intervals: int):
        feed.tick()
        current = feed.view()
        return _header(current), _banner(current), build_dashboard_figure(current)

    return app


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Intraday 0DTE dealer-positioning dashboard")
    parser.add_argument("--symbol", default="SPY", help="Underlying symbol (default: SPY). Use $SPX for the index.")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Render a deterministic simulated session instead of calling Schwab.",
    )
    parser.add_argument("--seed", type=int, default=11, help="Random seed used with --simulate.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    feed = MarketFeed(symbol=args.symbol, simulate=args.simulate, seed=args.seed)
    feed.bootstrap()
    app = create_app(feed)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
