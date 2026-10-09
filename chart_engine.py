"""
Candlestick visualization with future-gap GEX zone overlays.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd

from gex_engine import GEXZone

FUTURE_GAP_BARS = 10
POSITIVE_GEX_COLOR = "#00FF66"  # bright green
NEGATIVE_GEX_COLOR = "#FF3333"  # bright red
MIN_ALPHA = 0.15
MAX_ALPHA = 0.85


def extend_with_future_gap(
    ohlcv: pd.DataFrame,
    gap_bars: int = FUTURE_GAP_BARS,
) -> pd.DataFrame:
    """Append empty (NaN) rows so the chart extends into the future."""
    if ohlcv.empty:
        raise ValueError("ohlcv is empty; cannot extend chart.")

    last_index = ohlcv.index[-1]
    if isinstance(last_index, pd.Timestamp):
        freq = pd.infer_freq(ohlcv.index)
        if freq:
            future_index = pd.date_range(
                start=last_index,
                periods=gap_bars + 1,
                freq=freq,
            )[1:]
        else:
            step = ohlcv.index[-1] - ohlcv.index[-2] if len(ohlcv) > 1 else pd.Timedelta(days=1)
            future_index = pd.date_range(
                start=last_index + step,
                periods=gap_bars,
                freq=step,
            )
    else:
        future_index = range(int(last_index) + 1, int(last_index) + gap_bars + 1)

    future = pd.DataFrame(np.nan, index=future_index, columns=ohlcv.columns)
    return pd.concat([ohlcv, future])


def _normalize_zone_alphas(zones: list[GEXZone]) -> dict[int, float]:
    magnitudes = [abs(zone.net_gex) for zone in zones]
    max_mag = max(magnitudes) if magnitudes else 0.0

    alphas: dict[int, float] = {}
    for idx, magnitude in enumerate(magnitudes):
        if max_mag == 0:
            alphas[idx] = MIN_ALPHA
        else:
            scaled = magnitude / max_mag
            alphas[idx] = MIN_ALPHA + scaled * (MAX_ALPHA - MIN_ALPHA)
    return alphas


def _draw_gex_zones(
    ax: plt.Axes,
    zones: list[GEXZone],
    historical_bars: int,
    total_bars: int,
) -> None:
    """Shade GEX zones across the future gap using axhspan."""
    if not zones or total_bars <= historical_bars:
        return

    xmin = historical_bars / total_bars
    alphas = _normalize_zone_alphas(zones)

    for idx, zone in enumerate(zones):
        color = POSITIVE_GEX_COLOR if zone.net_gex >= 0 else NEGATIVE_GEX_COLOR
        ax.axhspan(
            zone.price_low,
            zone.price_high,
            xmin=xmin,
            xmax=1.0,
            facecolor=color,
            alpha=alphas[idx],
            edgecolor="none",
            zorder=0,
        )


def plot_gex_chart(
    ohlcv: pd.DataFrame,
    zones: list[GEXZone],
    *,
    title: str = "Macro GEX Chart",
    gamma_flip: float | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """
    Render a candlestick chart with GEX zones overlaid on the future gap.

    Returns the matplotlib Figure and Axes list from mplfinance.
    """
    extended = extend_with_future_gap(ohlcv)
    historical_bars = len(ohlcv)
    total_bars = len(extended)

    style = mpf.make_mpf_style(
        marketcolors=mpf.make_marketcolors(
            up="#26a69a",
            down="#ef5350",
            edge="inherit",
            wick="inherit",
            volume="inherit",
        ),
        gridstyle="--",
        y_on_right=False,
    )

    def _overlay(ax: plt.Axes) -> None:
        _draw_gex_zones(ax, zones, historical_bars, total_bars)
        if gamma_flip is not None:
            ax.axhline(
                gamma_flip,
                color="#FFD700",
                linestyle="--",
                linewidth=1.2,
                label="Gamma Flip",
                zorder=2,
            )
            ax.legend(loc="upper left")

    fig, axes = mpf.plot(
        extended,
        type="candle",
        style=style,
        title=title,
        volume=True,
        returnfig=True,
        figsize=(14, 8),
    )

    price_ax = axes[0]
    _overlay(price_ax)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig, axes


def render_analysis_chart(
    ohlcv: pd.DataFrame,
    zones: list[GEXZone],
    gamma_flip: float | None = None,
    ticker: str = "",
    save_path: str | Path | None = None,
    show: bool = True,
) -> dict[str, Any]:
    """Convenience wrapper used by main.py."""
    title = f"{ticker} Macro GEX" if ticker else "Macro GEX"
    fig, axes = plot_gex_chart(
        ohlcv,
        zones,
        title=title,
        gamma_flip=gamma_flip,
        save_path=save_path,
        show=show,
    )
    return {"figure": fig, "axes": axes}
