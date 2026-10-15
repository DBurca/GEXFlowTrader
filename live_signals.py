"""
Pure signal, filter, and paper-structure math for the SPY forward test.

Strategy 1 sells a 0DTE credit spread just outside the boundary that was
tested. An iron condor would be a second structure on the same signal; the
forward test keeps one defined-risk position per strategy account.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, time

import numpy as np
import pandas as pd

# Half-life of about two hours. Below this, intraday mean reversion has faded.
OU_THETA_FLOOR = 0.005
# Combined bid-ask width versus the structure's net mid.
DEFAULT_BA_RATIO = 0.05
PIN_FRACTION = 0.0015
IV_STABLE_PAD = 0.002
IV_SPIKE = 0.005
CDF_FLAT_FRACTION = 0.05
# A one-minute CDF endpoint can look flat after a round trip. The tester
# samples this once a minute, so three confirms is about three minutes.
CDF_FLAT_CONFIRM = 3


@dataclass(frozen=True)
class Leg:
    right: str  # "call" or "put"
    strike: float
    side: str  # "long" or "short"


@dataclass(frozen=True)
class EntrySignal:
    strategy: str
    direction: str  # "bullish" or "bearish"
    structure: str
    anchor: float
    reason: str
    stop: float | None = None


def ornstein_uhlenbeck(prices: np.ndarray | pd.Series) -> tuple[float, float, float] | None:
    """AR(1) OU fit. Theta is per bar and is allowed to be negative."""
    series = np.asarray(prices, dtype=float)
    series = series[np.isfinite(series)]
    if len(series) < 30:
        return None
    slope, intercept = np.polyfit(series[:-1], series[1:], 1)
    theta = float(1.0 - slope)
    if abs(1.0 - slope) < 1e-8:
        mu = float(np.mean(series))
    else:
        mu = float(intercept / (1.0 - slope))
    residuals = series[1:] - (intercept + slope * series[:-1])
    sigma = float(np.std(residuals, ddof=1)) if len(residuals) > 2 else 0.0
    return theta, mu, sigma


def infer_hmm_state(closes: np.ndarray | pd.Series) -> int | None:
    """
    Two-state HMM on 1-minute returns.

    State 0 is the lower-variance state. Returns None until there are enough
    bars or the fit does not converge.
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        return None

    closes = np.asarray(closes, dtype=float)
    returns = np.diff(closes)
    returns = returns[np.isfinite(returns)]
    if len(returns) < 40:
        return None
    model = GaussianHMM(
        n_components=2,
        covariance_type="diag",
        n_iter=200,
        random_state=42,
        tol=1e-3,
    )
    try:
        model.fit(returns.reshape(-1, 1))
    except Exception:
        return None
    if not model.monitor_.converged:
        return None
    states = model.predict(returns.reshape(-1, 1))
    variances = np.asarray(model.covars_, dtype=float).reshape(-1)
    low_variance_state = int(np.argmin(variances))
    return 0 if int(states[-1]) == low_variance_state else 1


def five_minute_bars(minute_bars: pd.DataFrame) -> pd.DataFrame:
    frame = minute_bars.sort_index()
    grouped = frame.resample("5min", label="left", closed="left").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    )
    return grouped.dropna(subset=["Open", "High", "Low", "Close"])


def completed_five_minute(bars_5: pd.DataFrame, now: datetime) -> pd.DataFrame:
    if bars_5.empty:
        return bars_5
    cutoff = pd.Timestamp(now)
    if cutoff.tzinfo is not None and bars_5.index.tz is None:
        cutoff = cutoff.tz_localize(None)
    elif cutoff.tzinfo is None and bars_5.index.tz is not None:
        cutoff = cutoff.tz_localize(bars_5.index.tz)
    finished = bars_5.index + pd.Timedelta(minutes=5) <= cutoff
    return bars_5.loc[finished]


def value_at_or_before(
    history: list[tuple[datetime, float]],
    moment: datetime,
) -> float | None:
    found = [value for stamped, value in history if stamped <= moment]
    if not found:
        return None
    return float(found[-1])


def minutes_before_close(now: datetime, session_close: time, minutes: int) -> bool:
    """True once the clock is inside the last `minutes` before the cash close."""
    now_minutes = now.hour * 60 + now.minute
    close_minutes = session_close.hour * 60 + session_close.minute
    return now_minutes >= close_minutes - minutes


def charm_pin_active(
    now: datetime,
    spot: float,
    pin_strike: float | None,
    session_close: time,
) -> bool:
    """After 60 minutes before the close, block entries when spot is pinned to max OI."""
    charm_start_minutes = session_close.hour * 60 + session_close.minute - 60
    now_minutes = now.hour * 60 + now.minute
    if now_minutes < charm_start_minutes or pin_strike is None or spot <= 0:
        return False
    return abs(spot - pin_strike) / spot <= PIN_FRACTION


def pin_strike_from_chain(chain: pd.DataFrame) -> float | None:
    if chain.empty:
        return None
    open_interest = chain["call_openInterest"].to_numpy(dtype=float) + chain[
        "put_openInterest"
    ].to_numpy(dtype=float)
    if not np.isfinite(open_interest).any() or np.nanmax(open_interest) <= 0:
        return None
    return float(chain["strikePrice"].iloc[int(np.nanargmax(open_interest))])


def _failed_down(bar: pd.Series, level: float) -> bool:
    return bool(bar["Low"] <= level and bar["Close"] > level)


def _failed_up(bar: pd.Series, level: float) -> bool:
    return bool(bar["High"] >= level and bar["Close"] < level)


def _cdf_supports_fade(direction: str, cdf_now: float, cdf_prev: float) -> bool:
    change = cdf_now - cdf_prev
    flat = abs(change) <= CDF_FLAT_FRACTION * max(abs(cdf_now), abs(cdf_prev), 1.0)
    if direction == "bullish":
        return change > 0 or flat
    return change < 0 or flat


def absorption_signal(
    bar: pd.Series,
    *,
    intrad_gex: float,
    macro_gex: float,
    hmm_state: int | None,
    call_wall: float | None,
    put_wall: float | None,
    upper_2: float,
    lower_2: float,
    upper_1: float = float("nan"),
    lower_1: float = float("nan"),
    call_wall_2: float | None = None,
    put_wall_2: float | None = None,
    cdf_now: float | None,
    cdf_prev: float | None,
    iv_now: float | None,
    iv_prev: float | None,
    momentum_slope: float | None = None,
    iv_max_fear_upper: float | None = None,
    iv_max_fear_lower: float | None = None,
) -> EntrySignal | None:
    """+GEX mean reversion: a 5-minute test that fails to close through the boundary."""
    if intrad_gex <= 0 or macro_gex < 0 or hmm_state != 0:
        return None
    if None in (cdf_now, cdf_prev, iv_now, iv_prev):
        return None
    if iv_now > iv_prev + IV_STABLE_PAD:
        return None

    candidates: list[tuple[str, float, str]] = []
    if put_wall is not None and _failed_down(bar, put_wall):
        candidates.append(("bullish", put_wall, "put wall"))
    if np.isfinite(lower_1) and _failed_down(bar, lower_1):
        candidates.append(("bullish", lower_1, "lower 1σ"))
    if np.isfinite(lower_2) and _failed_down(bar, lower_2):
        candidates.append(("bullish", lower_2, "lower 2σ"))
    if call_wall is not None and _failed_up(bar, call_wall):
        candidates.append(("bearish", call_wall, "call wall"))
    if np.isfinite(upper_1) and _failed_up(bar, upper_1):
        candidates.append(("bearish", upper_1, "upper 1σ"))
    if np.isfinite(upper_2) and _failed_up(bar, upper_2):
        candidates.append(("bearish", upper_2, "upper 2σ"))
    if not candidates:
        return None

    # Prefer the wall test when a band and a wall both qualify.
    candidates.sort(
        key=lambda item: (0 if "wall" in item[2] else 1, abs(item[1] - float(bar["Close"])))
    )
    direction, anchor, label = candidates[0]
    if momentum_slope is not None and (
        not np.isfinite(momentum_slope)
        or (direction == "bullish" and momentum_slope < -0.05)
        or (direction == "bearish" and momentum_slope > 0.05)
    ):
        return None
    if not _cdf_supports_fade(direction, float(cdf_now), float(cdf_prev)):
        return None
    structure = "put_credit" if direction == "bullish" else "call_credit"
    stop = None
    if direction == "bearish" and iv_max_fear_upper is not None and iv_max_fear_upper > float(anchor):
        stop = float(iv_max_fear_upper)
    elif direction == "bullish" and iv_max_fear_lower is not None and iv_max_fear_lower < float(anchor):
        stop = float(iv_max_fear_lower)
    elif "1σ" in label:
        outer = put_wall_2 if direction == "bullish" else call_wall_2
        if outer is not None:
            stop = float(outer)
    return EntrySignal(
        strategy="absorption",
        direction=direction,
        structure=structure,
        anchor=float(anchor),
        reason=f"5-minute test of {label} failed, CDF diverged, ATM IV stable",
        stop=stop,
    )


def _broke_up(previous: pd.Series, bar: pd.Series, level: float) -> bool:
    return bool(previous["Close"] <= level and bar["Close"] > level)


def _broke_down(previous: pd.Series, bar: pd.Series, level: float) -> bool:
    return bool(previous["Close"] >= level and bar["Close"] < level)


def cascade_signal(
    previous: pd.Series,
    bar: pd.Series,
    *,
    intrad_gex: float,
    macro_gex: float,
    gamma_flip: float | None,
    spot: float,
    call_wall: float | None,
    put_wall: float | None,
    zero_gamma: float | None,
    call_wall_2: float | None = None,
    put_wall_2: float | None = None,
    vwap: float,
    cdf_now: float | None,
    cdf_15: float | None,
    cdf_30: float | None,
    iv_now: float | None,
    iv_prev: float | None,
    momentum_price: float | None = None,
    momentum_vwap: float | None = None,
    momentum_slope: float | None = None,
) -> EntrySignal | None:
    """-GEX breakout: a 5-minute close through zero gamma or a wall."""
    flip_breached = gamma_flip is not None and spot < gamma_flip
    if intrad_gex >= 0 or not (macro_gex < 0 or flip_breached):
        return None
    if None in (cdf_now, cdf_15, cdf_30, iv_now, iv_prev):
        return None
    spiked = float(iv_now) >= float(iv_prev) + IV_SPIKE or float(iv_now) >= float(iv_prev) * 1.03
    if not spiked:
        return None

    latest_change = float(cdf_now) - float(cdf_15)
    prior_change = float(cdf_15) - float(cdf_30)
    upside: list[tuple[float, str]] = []
    downside: list[tuple[float, str]] = []
    if call_wall_2 is not None:
        if _broke_up(previous, bar, call_wall_2):
            upside.append((call_wall_2, "secondary call wall"))
    else:
        if call_wall is not None and _broke_up(previous, bar, call_wall):
            upside.append((call_wall, "call wall"))
        if zero_gamma is not None and _broke_up(previous, bar, zero_gamma):
            upside.append((zero_gamma, "zero gamma"))
    if put_wall_2 is not None:
        if _broke_down(previous, bar, put_wall_2):
            downside.append((put_wall_2, "secondary put wall"))
    else:
        if put_wall is not None and _broke_down(previous, bar, put_wall):
            downside.append((put_wall, "put wall"))
        if zero_gamma is not None and _broke_down(previous, bar, zero_gamma):
            downside.append((zero_gamma, "zero gamma"))

    if (
        upside
        and bar["Close"] > vwap
        and latest_change > 0
        and latest_change > prior_change
        and _momentum_aligned("bullish", momentum_price, momentum_vwap, momentum_slope)
    ):
        anchor, label = upside[0]
        return EntrySignal(
            strategy="cascade",
            direction="bullish",
            structure="call_debit",
            anchor=float(anchor),
            reason=f"5-minute close through {label} with accelerating CDF and rising IV",
        )
    if (
        downside
        and bar["Close"] < vwap
        and latest_change < 0
        and latest_change < prior_change
        and _momentum_aligned("bearish", momentum_price, momentum_vwap, momentum_slope)
    ):
        anchor, label = downside[0]
        return EntrySignal(
            strategy="cascade",
            direction="bearish",
            structure="put_debit",
            anchor=float(anchor),
            reason=f"5-minute close through {label} with accelerating CDF and rising IV",
        )
    return None


MACRO_ENTRY_START = 11 * 60
MACRO_ENTRY_END = 15 * 60 + 55


def macro_entry_window(now: datetime) -> bool:
    """11:00 through 15:55 ET, using the clock on ``now``."""
    minutes = now.hour * 60 + now.minute
    return MACRO_ENTRY_START <= minutes <= MACRO_ENTRY_END


def completed_fifteen_minute_closes(minute_bars: pd.DataFrame, now: datetime) -> list[float]:
    """Closes of 15-minute bars that have finished by ``now``."""
    if minute_bars.empty or "Close" not in minute_bars.columns:
        return []
    grouped = (
        minute_bars.sort_index()
        .resample("15min", label="left", closed="left")
        .agg({"Close": "last"})
        .dropna(subset=["Close"])
    )
    if grouped.empty:
        return []
    cutoff = pd.Timestamp(now)
    if cutoff.tzinfo is not None and grouped.index.tz is None:
        cutoff = cutoff.tz_localize(None)
    elif cutoff.tzinfo is None and grouped.index.tz is not None:
        cutoff = cutoff.tz_localize(grouped.index.tz)
    finished = grouped.index + pd.Timedelta(minutes=15) <= cutoff
    return [float(value) for value in grouped.loc[finished, "Close"].tolist()]


def _momentum_aligned(
    direction: str,
    price: float | None,
    vwap: float | None,
    slope: float | None,
) -> bool:
    """ES (or the SPY fallback) must agree with the breakout. Omitted inputs stay open for older callers."""
    if price is None and vwap is None and slope is None:
        return True
    if price is None or vwap is None or slope is None:
        return False
    if not (np.isfinite(price) and np.isfinite(vwap) and np.isfinite(slope)):
        return False
    if direction == "bullish":
        return price > vwap and slope > 0
    return price < vwap and slope < 0


def macro_swing_signal(
    *,
    now: datetime,
    macro_gex: float,
    gamma_flip: float | None,
    hmm_state: int | None,
    ou_theta: float | None,
    fifteen_closes: list[float],
) -> EntrySignal | None:
    """Intraday put debit once two 15-minute closes are under the macro flip."""
    if not macro_entry_window(now):
        return None
    if gamma_flip is None or not np.isfinite(gamma_flip) or macro_gex >= 0:
        return None
    if len(fifteen_closes) < 2:
        return None
    if fifteen_closes[-1] >= gamma_flip or fifteen_closes[-2] >= gamma_flip:
        return None
    trend = hmm_state == 1 or (ou_theta is not None and ou_theta <= OU_THETA_FLOOR)
    if not trend:
        return None
    why = "HMM state 1" if hmm_state == 1 else "OU mean reversion faded"
    return EntrySignal(
        strategy="macro",
        direction="bearish",
        structure="put_debit_macro",
        anchor=float(gamma_flip),
        reason=f"ENTER_MACRO_PUT_DEBIT, two 15-minute closes below the flip, {why}",
    )


def _nearest_strike(strikes: np.ndarray, target: float, side: str) -> float | None:
    if strikes.size == 0:
        return None
    if side == "at_or_below":
        eligible = strikes[strikes <= target + 1e-6]
        return float(eligible[-1]) if eligible.size else None
    if side == "at_or_above":
        eligible = strikes[strikes >= target - 1e-6]
        return float(eligible[0]) if eligible.size else None
    return float(strikes[int(np.argmin(np.abs(strikes - target)))])


def credit_legs(chain: pd.DataFrame, structure: str, anchor: float) -> list[Leg] | None:
    strikes = np.sort(chain["strikePrice"].to_numpy(dtype=float))
    if structure == "put_credit":
        short = _nearest_strike(strikes, anchor, "at_or_below")
        if short is None:
            return None
        lower = strikes[strikes < short - 1e-6]
        if lower.size == 0:
            return None
        return [
            Leg("put", short, "short"),
            Leg("put", float(lower[-1]), "long"),
        ]
    short = _nearest_strike(strikes, anchor, "at_or_above")
    if short is None:
        return None
    higher = strikes[strikes > short + 1e-6]
    if higher.size == 0:
        return None
    return [
        Leg("call", short, "short"),
        Leg("call", float(higher[0]), "long"),
    ]


def debit_legs(
    chain: pd.DataFrame,
    structure: str,
    spot: float,
    width: float = 2.0,
) -> list[Leg] | None:
    strikes = np.sort(chain["strikePrice"].to_numpy(dtype=float))
    if structure == "call_debit":
        long_strike = _nearest_strike(strikes, spot, "at_or_below") or _nearest_strike(
            strikes, spot, "nearest"
        )
        if long_strike is None:
            return None
        higher = strikes[strikes >= long_strike + width - 0.2]
        if higher.size == 0:
            higher = strikes[strikes > long_strike + 1e-6]
        if higher.size == 0:
            return None
        return [
            Leg("call", long_strike, "long"),
            Leg("call", float(higher[0]), "short"),
        ]
    long_strike = _nearest_strike(strikes, spot, "at_or_above") or _nearest_strike(
        strikes, spot, "nearest"
    )
    if long_strike is None:
        return None
    lower = strikes[strikes <= long_strike - width + 0.2]
    if lower.size == 0:
        lower = strikes[strikes < long_strike - 1e-6]
    if lower.size == 0:
        return None
    return [
        Leg("put", long_strike, "long"),
        Leg("put", float(lower[-1]), "short"),
    ]


def _quote(chain: pd.DataFrame, leg: Leg) -> tuple[float, float] | None:
    if chain.empty:
        return None
    distance = (chain["strikePrice"] - leg.strike).abs()
    row = chain.loc[distance.idxmin()]
    if abs(float(row["strikePrice"]) - leg.strike) > 0.051:
        return None
    bid = float(row[f"{leg.right}_bid"])
    ask = float(row[f"{leg.right}_ask"])
    if not np.isfinite(bid) or not np.isfinite(ask) or ask < bid or ask < 0 or bid < 0:
        return None
    if ask == 0 and bid == 0:
        return None
    return bid, ask


def structure_nets(chain: pd.DataFrame, legs: list[Leg]) -> tuple[float, float] | None:
    """
    Return (short_mid - long_mid, combined bid-ask width).

    The signed net is positive for a credit and negative for a debit.
    """
    signed = 0.0
    width = 0.0
    for leg in legs:
        quote = _quote(chain, leg)
        if quote is None:
            return None
        bid, ask = quote
        mid = (bid + ask) / 2.0
        signed += mid if leg.side == "short" else -mid
        width += ask - bid
    return signed, width


def executed_fill(signed_mid: float, total_width: float, factor: float) -> float:
    """Worse than the signed mid by ``factor`` times the summed bid-ask width.

    A credit is received for less. A debit, whose signed mid is negative, is
    paid for more. Both are ``signed mid − factor × Σ(ask − bid)``.
    """
    return float(signed_mid) - float(factor) * float(total_width)


def fill_price_blocks(signed_mid: float, fill: float, legs: list[Leg]) -> str | None:
    """Reject a debit that costs at least the strike width, or a credit at or under $0.05."""
    if signed_mid < 0:
        if -fill >= spread_width(legs):
            return "debit fill is at least the strike width"
        return None
    if fill <= 0.05:
        return "credit fill is at or below 0.05"
    return None


def slippage_blocks(chain: pd.DataFrame, legs: list[Leg], max_ratio: float) -> str | None:
    """Reason the order is blocked, or None when the quote is tight enough."""
    nets = structure_nets(chain, legs)
    if nets is None:
        return "missing or crossed bid-ask"
    signed, ba_width = nets
    premium = abs(signed)
    if premium < 0.05:
        return "structure premium below 0.05"
    strikes = [leg.strike for leg in legs]
    strike_width = abs(max(strikes) - min(strikes))
    if signed > 0 and signed >= strike_width:
        return "credit is wider than the strikes"
    if signed < 0 and abs(signed) >= strike_width:
        return "debit is wider than the strikes"
    ratio = ba_width / premium
    if ratio > max_ratio:
        return f"bid-ask {ratio:.1%} of spread value"
    return None


def describe_legs(legs: list[Leg], contracts: int) -> str:
    parts = [
        f"{leg.side} {_strike_text(leg.strike)} {leg.right}"
        for leg in legs
    ]
    return f"{contracts}x " + " / ".join(parts)


def _strike_text(value: float) -> str:
    text = f"{value:.2f}"
    return text.rstrip("0").rstrip(".")


def spread_width(legs: list[Leg]) -> float:
    strikes = [leg.strike for leg in legs]
    return abs(max(strikes) - min(strikes))


def contract_count(balance: float, max_loss_dollars: float, risk_fraction: float = 0.02) -> int:
    """Whole contracts whose max loss fits inside the risk budget, capped at 25."""
    if max_loss_dollars <= 0 or balance <= max_loss_dollars:
        return 0
    count = int((balance * risk_fraction) // max_loss_dollars)
    if count < 1:
        return 0
    return min(count, 25)


def max_loss_per_contract(signed_net: float, legs: list[Leg]) -> float:
    width = spread_width(legs)
    if signed_net > 0:
        return max(width - signed_net, 0.0) * 100.0
    return abs(signed_net) * 100.0


RISK_STOP_FRACTION = 0.50


def intrabar_stop(
    direction: str,
    spot: float,
    stop: float | None,
    entry_net: float,
    marked_net: float | None,
    contracts: int,
    legs: list[Leg],
) -> str | None:
    """Stop on the live spot or once half the defined risk is gone. No 5-minute wait."""
    if stop is not None and math.isfinite(spot) and math.isfinite(stop):
        if direction == "bullish" and spot < stop:
            return "spot through the structural stop"
        if direction == "bearish" and spot > stop:
            return "spot through the structural stop"
    if marked_net is None or contracts < 1:
        return None
    defined = max_loss_per_contract(entry_net, legs) * contracts
    if defined <= 0:
        return None
    if trade_pnl(entry_net, marked_net, contracts, 0.0) <= -RISK_STOP_FRACTION * defined:
        return "half of defined risk lost"
    return None


def dynamic_slippage_factor(live_width: float, baseline_width: float | None, base: float = 0.35) -> float:
    """Widen the calm factor when the live spread is wider than the baseline, up to 0.50."""
    if baseline_width is None or baseline_width <= 0 or not math.isfinite(live_width):
        return float(base)
    expansion = max(0.0, (float(live_width) - float(baseline_width)) / float(baseline_width))
    return min(0.50, float(base) + 0.15 * expansion)


def exit_slippage_factor(
    strategy: str,
    event: str,
    base: float,
    live_width: float | None = None,
    baseline_width: float | None = None,
) -> float:
    """Cascade entries and panic exits cross the spread. Calm fills scale with spread expansion."""
    if event in {"STOP", "MARKET_FLATTEN"} or (event == "OPEN" and strategy == "cascade"):
        return max(float(base), 1.0)
    if live_width is None or baseline_width is None:
        return float(base)
    return dynamic_slippage_factor(live_width, baseline_width, base)


def trade_pnl(
    entry_net: float,
    current_net: float,
    contracts: int,
    fees: float,
) -> float:
    """Dollar PnL. Both nets are short-mid minus long-mid."""
    return (entry_net - current_net) * 100.0 * contracts - fees


def round_trip_fees(contracts: int, n_legs: int, fee_per_contract: float = 0.65) -> float:
    return fee_per_contract * contracts * n_legs * 2


def cdf_flattened(cdf_now: float, cdf_prev: float) -> bool:
    return abs(cdf_now - cdf_prev) <= CDF_FLAT_FRACTION * max(abs(cdf_now), abs(cdf_prev), 1.0)


def next_cdf_flat_streak(streak: int, is_flat: bool) -> tuple[int, bool]:
    """Advance the consecutive minute-sample count. The second value is true once it holds."""
    streak = streak + 1 if is_flat else 0
    return streak, streak >= CDF_FLAT_CONFIRM


def touched_iv_peak(
    direction: str,
    spot: float,
    entry_spot: float,
    peak: float | None,
) -> bool:
    """Spot has reached the IV peak center from the fade entry."""
    if peak is None or not math.isfinite(peak) or not math.isfinite(spot):
        return False
    if direction == "bullish":
        return peak >= entry_spot and spot >= peak
    return peak <= entry_spot and spot <= peak


def reverted_to_mean(
    direction: str,
    spot: float,
    entry_spot: float,
    vwap: float,
    zero_gamma: float | None,
) -> bool:
    targets = [vwap]
    if zero_gamma is not None and np.isfinite(zero_gamma):
        targets.append(zero_gamma)
    if direction == "bullish":
        goals = [level for level in targets if level >= entry_spot]
        return bool(goals) and spot >= min(goals)
    goals = [level for level in targets if level <= entry_spot]
    return bool(goals) and spot <= max(goals)


def breach_confirmed(
    direction: str,
    close: float,
    anchor: float,
    cdf_now: float,
    cdf_prev: float,
) -> bool:
    """5-minute close through the tested level with CDF moving the same way."""
    change = cdf_now - cdf_prev
    if direction == "bullish":
        return close < anchor and change < 0
    return close > anchor and change > 0
