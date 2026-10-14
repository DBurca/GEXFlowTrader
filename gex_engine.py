"""
Net gamma exposure for the SPY paper forward test.

Macro strike GEX uses open interest. Intraday levels use a separate dollar-gamma
book for the regime and zero-gamma line, and an open-interest book for walls.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from event_calendar import session_close_time
from ta_engine import minutes_remaining


def calculate_strike_gex(option_fields: pd.DataFrame, spot_price: float) -> pd.DataFrame:
    """
    Compute per-strike net GEX:
        GEX_strike = (OI_call * gamma_call - OI_put * gamma_put) * 100 * S
    """
    if option_fields.empty:
        raise ValueError("option_fields is empty; cannot calculate GEX.")

    frame = option_fields.copy()
    frame["net_gex"] = (
        (
            frame["call_openInterest"] * frame["call_gamma"]
            - frame["put_openInterest"] * frame["put_gamma"]
        )
        * 100.0
        * spot_price
    )
    return frame.sort_values("strikePrice").reset_index(drop=True)


def calculate_total_net_gex(gex_profile: pd.DataFrame) -> float:
    """Sum net GEX across all strikes."""
    return float(gex_profile["net_gex"].sum())


def classify_regime(total_net_gex: float) -> str:
    """
    Classify the overall market regime from aggregate GEX.

    High positive GEX -> mean-reverting; high negative GEX -> momentum.
    """
    if total_net_gex >= 0:
        return "mean_reverting"
    return "momentum"


# ---------------------------------------------------------------------------
# Intraday 0DTE dollar gamma, walls, delta flow, and ATM IV
# ---------------------------------------------------------------------------

EASTERN = ZoneInfo("America/New_York")
CONTRACT_MULTIPLIER = 100.0
ONE_PERCENT = 0.01

CHAIN_COLUMNS: tuple[str, ...] = (
    "strikePrice",
    "call_openInterest",
    "call_gamma",
    "call_delta",
    "call_volatility",
    "call_volume",
    "put_openInterest",
    "put_gamma",
    "put_delta",
    "put_volatility",
    "put_volume",
)

SNAPSHOT_COLUMNS: tuple[str, ...] = (
    "spot",
    "call_wall",
    "put_wall",
    "zero_gamma",
    "net_gex",
    "call_gex",
    "put_gex",
    "cdf",
    "atm_iv",
)


@dataclass(frozen=True)
class IntradayGEXSnapshot:
    """One 0DTE chain snapshot reduced to dashboard levels."""

    as_of: pd.Timestamp
    spot: float
    call_wall: float | None
    put_wall: float | None
    call_wall_2: float | None
    put_wall_2: float | None
    call_wall_1_gamma: float | None
    call_wall_2_gamma: float | None
    put_wall_1_gamma: float | None
    put_wall_2_gamma: float | None
    zero_gamma: float | None
    net_gex: float
    call_gex: float
    put_gex: float
    cdf: float
    atm_iv: float
    charm_hedge: float
    atm_call_iv: float
    atm_put_iv: float


def _as_eastern(moment: datetime | pd.Timestamp) -> pd.Timestamp:
    """Interpret naive timestamps as US/Eastern and convert aware ones."""
    stamp = pd.Timestamp(moment)
    if stamp.tzinfo is None:
        return stamp.tz_localize(EASTERN)
    return stamp.tz_convert(EASTERN)


def _finite(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


# Smooth blend from open interest at the cash open toward session volume.
# t = 0 is 09:30. By the close the volume share is about 0.76, with no step.
WEIGHT_DECAY = 0.0075
VOL_WEIGHT_CLOSE = 0.8


def positioning_weights(as_of: datetime | pd.Timestamp) -> tuple[float, float]:
    """Return (open-interest weight, volume weight) for the minute since 09:30."""
    stamp = _as_eastern(as_of)
    minutes = (stamp.hour * 60 + stamp.minute) - (9 * 60 + 30)
    elapsed = max(0.0, min(float(minutes), 390.0))
    volume_weight = VOL_WEIGHT_CLOSE * (1.0 - math.exp(-WEIGHT_DECAY * elapsed))
    return 1.0 - volume_weight, volume_weight


def positioning_weight(
    open_interest: Any,
    volume: Any,
    as_of: datetime | pd.Timestamp,
) -> np.ndarray:
    """Continuous blend of open interest and session volume. No 10:30 step."""
    oi = _finite(open_interest)
    traded = _finite(volume)
    if oi.shape != traded.shape:
        raise ValueError("open interest and volume must have the same shape.")
    oi_weight, volume_weight = positioning_weights(as_of)
    return oi_weight * oi + volume_weight * traded


def dollar_gamma(gamma: Any, weight: Any, spot: float) -> np.ndarray:
    """
    Dollar gamma for a 1% spot move.

    Dollar Gamma = Gamma * Positioning Weight * 100 * Spot^2 * 0.01
    """
    if not np.isfinite(spot) or spot <= 0:
        raise ValueError(f"spot must be a positive finite price (got {spot}).")
    return (
        np.abs(_finite(gamma))
        * _finite(weight)
        * CONTRACT_MULTIPLIER
        * float(spot) ** 2
        * ONE_PERCENT
    )


def strike_exposure_frame(
    chain: pd.DataFrame,
    spot: float,
    as_of: datetime | pd.Timestamp,
) -> pd.DataFrame:
    """Vectorized per-strike call, put, and net dollar gamma."""
    missing = [column for column in CHAIN_COLUMNS if column not in chain.columns]
    if missing:
        raise ValueError(f"Option chain is missing columns: {', '.join(missing)}")
    if chain.empty:
        raise ValueError("Option chain is empty; cannot calculate dollar gamma.")

    frame = chain.sort_values("strikePrice")
    call_weight = positioning_weight(
        frame["call_openInterest"], frame["call_volume"], as_of
    )
    put_weight = positioning_weight(
        frame["put_openInterest"], frame["put_volume"], as_of
    )
    call_dg = dollar_gamma(frame["call_gamma"], call_weight, spot)
    put_dg = -dollar_gamma(frame["put_gamma"], put_weight, spot)

    exposure = pd.DataFrame(
        {
            "strikePrice": frame["strikePrice"].to_numpy(dtype=float),
            "call_weight": call_weight,
            "put_weight": put_weight,
            "call_dollar_gamma": call_dg,
            "put_dollar_gamma": put_dg,
            "net_dollar_gamma": call_dg + put_dg,
        }
    )
    return exposure.reset_index(drop=True)


# Secondary wall is kept only when its dollar gamma is at least this
# fraction of the primary strike's dollar gamma.
WALL_SECONDARY_RATIO = 0.80


def oi_dollar_gamma(gamma: Any, open_interest: Any, spot: float) -> np.ndarray:
    """Dollar gamma = Gamma × open interest × 100 × spot."""
    if not np.isfinite(spot) or spot <= 0:
        raise ValueError(f"spot must be a positive finite price (got {spot}).")
    return _finite(gamma) * _finite(open_interest) * 100.0 * float(spot)


def net_gex(chain: pd.DataFrame, spot: float) -> float:
    """Call dollar gamma minus put dollar gamma, summed across strikes."""
    if chain.empty:
        return 0.0
    calls = oi_dollar_gamma(chain["call_gamma"], chain["call_openInterest"], spot)
    puts = oi_dollar_gamma(chain["put_gamma"], chain["put_openInterest"], spot)
    return float(np.sum(calls - puts))


def volume_gex(chain: pd.DataFrame, spot: float) -> float:
    """0DTE gamma × session volume × 100 × spot, calls minus puts."""
    if chain.empty:
        return 0.0
    calls = oi_dollar_gamma(chain["call_gamma"], chain["call_volume"], spot)
    puts = oi_dollar_gamma(chain["put_gamma"], chain["put_volume"], spot)
    return float(np.sum(calls - puts))


def overnight_pml(chain: pd.DataFrame) -> float | None:
    """Strike with the largest long-option dealer liability on the opening 0DTE book.

    Dealers short the customer inventory, so liability at a strike is call
    open interest × call gamma plus put open interest × put gamma. Spot and
    the contract multiplier are the same at every strike and do not move the peak.
    """
    if chain.empty or "strikePrice" not in chain.columns:
        return None
    strikes = chain["strikePrice"].to_numpy(dtype=float)
    call_oi = _finite(chain["call_openInterest"]) if "call_openInterest" in chain.columns else np.zeros(strikes.shape)
    put_oi = _finite(chain["put_openInterest"]) if "put_openInterest" in chain.columns else np.zeros(strikes.shape)
    call_gamma = _finite(chain["call_gamma"]) if "call_gamma" in chain.columns else np.zeros(strikes.shape)
    put_gamma = _finite(chain["put_gamma"]) if "put_gamma" in chain.columns else np.zeros(strikes.shape)
    finite = np.isfinite(strikes)
    strikes = strikes[finite]
    liability = (call_oi * call_gamma + put_oi * put_gamma)[finite]
    if strikes.size == 0 or not np.isfinite(liability).any() or float(np.nanmax(liability)) <= 0:
        return None
    return float(strikes[int(np.nanargmax(liability))])


# A strike stays in the IV peak when it is this close to the highest strike IV.
PEAK_IV_RATIO = 0.98
# Exposure has dropped off once dollar gamma is under this fraction of the primary wall.
FEAR_GAMMA_RATIO = 0.20


def _strike_iv(chain: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    frame = chain.sort_values("strikePrice")
    strikes = frame["strikePrice"].to_numpy(dtype=float)
    call_iv = frame["call_volatility"].to_numpy(dtype=float) if "call_volatility" in frame.columns else np.full(strikes.shape, np.nan)
    put_iv = frame["put_volatility"].to_numpy(dtype=float) if "put_volatility" in frame.columns else np.full(strikes.shape, np.nan)
    stacked = np.vstack([call_iv, put_iv])
    with np.errstate(all="ignore"):
        iv = np.nanmean(stacked, axis=0)
    return strikes, iv


def iv_peak_center(chain: pd.DataFrame) -> float | None:
    """Geometric center of the contiguous 0DTE strikes that hold the IV peak."""
    if chain.empty:
        return None
    strikes, iv = _strike_iv(chain)
    finite = np.isfinite(strikes) & np.isfinite(iv) & (strikes > 0) & (iv > 0)
    if not finite.any():
        return None
    strikes, iv = strikes[finite], iv[finite]
    order = np.argsort(strikes)
    strikes, iv = strikes[order], iv[order]
    peak = float(np.max(iv))
    zone = iv >= peak * PEAK_IV_RATIO
    center_at = int(np.argmax(iv))
    left = center_at
    while left > 0 and zone[left - 1]:
        left -= 1
    right = center_at
    while right + 1 < zone.size and zone[right + 1]:
        right += 1
    cluster = strikes[left : right + 1]
    return float(np.exp(np.mean(np.log(cluster))))


def iv_max_fear_levels(chain: pd.DataFrame, spot: float) -> tuple[float | None, float | None]:
    """Outer strikes beyond the call and put walls where dollar gamma falls below 20% of the primary.

    Returns (iv_max_fear_upper, iv_max_fear_lower).
    """
    if chain.empty or not np.isfinite(spot) or spot <= 0:
        return None, None
    frame = chain.sort_values("strikePrice")
    strikes = frame["strikePrice"].to_numpy(dtype=float)
    call_g = oi_dollar_gamma(frame["call_gamma"], frame["call_openInterest"], float(spot))
    put_g = oi_dollar_gamma(frame["put_gamma"], frame["put_openInterest"], float(spot))
    call_1, call_1_g, call_2, _call_2_g = rank_gamma_walls(strikes, call_g)
    put_1, put_1_g, put_2, _put_2_g = rank_gamma_walls(strikes, put_g)
    upper_wall = max((level for level in (call_1, call_2) if level is not None), default=None)
    lower_wall = min((level for level in (put_1, put_2) if level is not None), default=None)
    upper = _fear_beyond(strikes, call_g, upper_wall, call_1_g, above=True)
    lower = _fear_beyond(strikes, put_g, lower_wall, put_1_g, above=False)
    return upper, lower


def _fear_beyond(
    strikes: np.ndarray,
    exposure: np.ndarray,
    wall: float | None,
    primary: float | None,
    above: bool,
) -> float | None:
    if wall is None or primary is None or not np.isfinite(primary) or primary <= 0:
        return None
    order = np.argsort(strikes)
    strikes = strikes[order]
    exposure = exposure[order]
    threshold = FEAR_GAMMA_RATIO * float(primary)
    if above:
        beyond = np.flatnonzero(strikes > wall + 1e-6)
    else:
        beyond = np.flatnonzero(strikes < wall - 1e-6)[::-1]
    for index in beyond:
        if float(exposure[index]) < threshold:
            return float(strikes[index])
    return None


def normalized_gex(net: float, spot: float) -> float:
    """Dollars of underlying to buy or sell for a 1% move: net GEX × 0.01 × spot."""
    if not np.isfinite(net) or not np.isfinite(spot):
        return float("nan")
    return float(net) * 0.01 * float(spot)


# Close enough for SPY charm and gamma. Rate and dividend barely move either greek.
RISK_FREE = 0.04
DIVIDEND_YIELD = 0.012
MINUTES_PER_YEAR = 365 * 24 * 60


def _year_fraction(minutes_left: float) -> float:
    """Calendar years until the close. One minute is the floor, so 16:00 does not divide by zero."""
    return max(float(minutes_left), 1.0) / MINUTES_PER_YEAR


def _minutes_to_close(as_of: datetime | pd.Timestamp) -> float:
    stamp = _as_eastern(as_of)
    return minutes_remaining(stamp.to_pydatetime(), session_close_time(stamp.date()))


def _d1_d2(spot: float, strike: np.ndarray, t_years: float, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sqrt_t = float(np.sqrt(t_years))
    vol_sqrt = sigma * sqrt_t
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(spot / strike) + (RISK_FREE - DIVIDEND_YIELD + 0.5 * sigma**2) * t_years) / vol_sqrt
    return d1, d1 - vol_sqrt


def black_scholes_gamma(spot: float, strike: Any, t_years: float, sigma: Any) -> np.ndarray:
    """Black-Scholes gamma at the live spot and each strike's own implied vol."""
    strike_array = np.asarray(strike, dtype=float)
    vol = np.asarray(sigma, dtype=float)
    d1, _ = _d1_d2(spot, strike_array, t_years, vol)
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma = np.exp(-DIVIDEND_YIELD * t_years) * norm.pdf(d1) / (spot * vol * np.sqrt(t_years))
    good = np.isfinite(gamma) & (vol > 0) & (strike_array > 0) & (spot > 0)
    return np.where(good, gamma, 0.0)


def black_scholes_charm(spot: float, strike: Any, t_years: float, sigma: Any, option_type: str) -> np.ndarray:
    """Change in delta per year of time passing. Calls and puts use the same d1 and d2."""
    strike_array = np.asarray(strike, dtype=float)
    vol = np.asarray(sigma, dtype=float)
    d1, d2 = _d1_d2(spot, strike_array, t_years, vol)
    sqrt_t = np.sqrt(t_years)
    discount = np.exp(-DIVIDEND_YIELD * t_years)
    cdf = norm.cdf(d1 if option_type == "call" else -d1)
    term1 = DIVIDEND_YIELD * discount * cdf
    numer = 2 * (RISK_FREE - DIVIDEND_YIELD) * t_years - d2 * vol * sqrt_t
    denom = 2 * t_years * sqrt_t * vol
    with np.errstate(divide="ignore", invalid="ignore"):
        term2 = discount * norm.pdf(d1) * (numer / denom)
    charm = (term1 - term2) if option_type == "call" else (-term1 - term2)
    good = np.isfinite(charm) & (vol > 0) & (strike_array > 0) & (spot > 0)
    return np.where(good, charm, 0.0)


def _book_weights(
    chain: pd.DataFrame,
    as_of: datetime | pd.Timestamp,
    minutes_left: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, float]:
    frame = chain.sort_values("strikePrice")
    strikes = frame["strikePrice"].to_numpy(dtype=float)
    call_weight = positioning_weight(frame["call_openInterest"], frame["call_volume"], as_of)
    put_weight = positioning_weight(frame["put_openInterest"], frame["put_volume"], as_of)
    return frame, strikes, call_weight, put_weight, _year_fraction(minutes_left)


def iv_net_dollar_gamma(
    chain: pd.DataFrame,
    spot: float,
    as_of: datetime | pd.Timestamp,
    minutes_left: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Dollar-gamma book rebuilt from live spot and each strike's implied vol.

    Same weights and sign as the regime book. Gamma itself is Black-Scholes
    at the current spot, so the zero-gamma strike moves when IV moves.
    """
    frame, strikes, call_weight, put_weight, t_years = _book_weights(chain, as_of, minutes_left)
    call_gamma = black_scholes_gamma(spot, strikes, t_years, frame["call_volatility"].to_numpy(dtype=float))
    put_gamma = black_scholes_gamma(spot, strikes, t_years, frame["put_volatility"].to_numpy(dtype=float))
    net = dollar_gamma(call_gamma, call_weight, spot) - dollar_gamma(put_gamma, put_weight, spot)
    return strikes, net


def charm_to_close(
    chain: pd.DataFrame,
    spot: float,
    as_of: datetime | pd.Timestamp,
    minutes_left: float,
) -> float:
    """
    Dollars of underlying the signed book must trade by the close as delta decays.

    Charm is Black-Scholes dDelta per year elapsed. Positive means the book's
    delta rises into the close, so a hedge sells shares. Calls minus puts,
    with the same positioning weight as the regime book.
    """
    if chain.empty or not np.isfinite(minutes_left) or minutes_left <= 0 or not np.isfinite(spot) or spot <= 0:
        return 0.0
    frame, strikes, call_weight, put_weight, t_years = _book_weights(chain, as_of, minutes_left)
    call_charm = black_scholes_charm(spot, strikes, t_years, frame["call_volatility"].to_numpy(dtype=float), "call")
    put_charm = black_scholes_charm(spot, strikes, t_years, frame["put_volatility"].to_numpy(dtype=float), "put")
    shares_per_year = np.sum((call_charm * call_weight - put_charm * put_weight) * CONTRACT_MULTIPLIER)
    shares = float(shares_per_year) * (float(minutes_left) / MINUTES_PER_YEAR)
    return shares * float(spot)


def rank_gamma_walls(
    strikes: Any,
    exposure: Any,
    ratio: float = WALL_SECONDARY_RATIO,
) -> tuple[float | None, float | None, float | None, float | None]:
    """
    Primary and secondary walls.

    Returns (strike_1, dollar_gamma_1, strike_2, dollar_gamma_2). The
    secondary strike is None when its dollar gamma is below ``ratio`` times
    the primary.
    """
    strike_array = np.asarray(strikes, dtype=float)
    values = np.asarray(exposure, dtype=float)
    if strike_array.size == 0 or strike_array.size != values.size:
        return None, None, None, None
    finite = np.isfinite(values) & np.isfinite(strike_array) & (values > 0)
    if not finite.any():
        return None, None, None, None
    order = np.argsort(np.where(finite, values, -np.inf))[::-1]
    first = int(order[0])
    primary_strike = float(strike_array[first])
    primary_gamma = float(values[first])
    if order.size < 2:
        return primary_strike, primary_gamma, None, None
    second = int(order[1])
    if not finite[second] or values[second] < ratio * primary_gamma:
        return primary_strike, primary_gamma, None, None
    return primary_strike, primary_gamma, float(strike_array[second]), float(values[second])


def call_wall_strike(exposure: pd.DataFrame) -> float | None:
    """Strike with the single highest positive call dollar gamma."""
    values = exposure["call_dollar_gamma"].to_numpy(dtype=float)
    if values.size == 0 or not np.isfinite(values).any() or np.nanmax(values) <= 0:
        return None
    return float(exposure["strikePrice"].iloc[int(np.nanargmax(values))])


# A one-strike step on SPY is a real wall walk. Anything farther has to
# repeat on the next poll before it replaces the level. One bad gamma quote
# otherwise becomes the wall for a single minute and then snaps back.
LEVEL_JUMP = 1.01
LEVEL_CONFIRM = 2


@dataclass
class LevelState:
    """Last accepted level, plus a jump that is waiting to be confirmed."""

    accepted: float | None = None
    pending: float | None = None
    count: int = 0
    clearing: bool = False


def stabilize_level(
    candidate: float | None,
    state: LevelState,
    jump: float = LEVEL_JUMP,
    confirm: int = LEVEL_CONFIRM,
) -> LevelState:
    """Accept a small move immediately. Hold a larger jump until it repeats."""
    if candidate is None or not np.isfinite(candidate):
        return state
    value = float(candidate)
    if state.accepted is None or abs(value - state.accepted) <= jump:
        return LevelState(accepted=value)
    if state.pending is not None and abs(value - state.pending) <= jump:
        count = state.count + 1
        pending = state.pending
    else:
        count = 1
        pending = value
    if count >= confirm:
        return LevelState(accepted=pending)
    return LevelState(accepted=state.accepted, pending=pending, count=count)


def stabilize_optional_level(
    candidate: float | None,
    state: LevelState,
    jump: float = LEVEL_JUMP,
    confirm: int = LEVEL_CONFIRM,
) -> LevelState:
    """Like ``stabilize_level``, but a missing reading can clear the level.

    A new strike and a disappearance both have to repeat for ``confirm``
    polls. A one-strike step is taken immediately once a level exists.
    """
    value = None if candidate is None or not np.isfinite(candidate) else float(candidate)
    if state.accepted is None:
        if value is None:
            return LevelState()
        if state.pending is not None and abs(value - state.pending) <= jump:
            count = state.count + 1
            pending = state.pending
        else:
            count = 1
            pending = value
        if count >= confirm:
            return LevelState(accepted=pending)
        return LevelState(pending=pending, count=count)
    if value is not None and abs(value - state.accepted) <= jump:
        return LevelState(accepted=value)
    if value is None:
        count = state.count + 1 if state.clearing else 1
        if count >= confirm:
            return LevelState()
        return LevelState(accepted=state.accepted, clearing=True, count=count)
    if (
        not state.clearing
        and state.pending is not None
        and abs(value - state.pending) <= jump
    ):
        count = state.count + 1
        pending = state.pending
    else:
        count = 1
        pending = value
    if count >= confirm:
        return LevelState(accepted=pending)
    return LevelState(accepted=state.accepted, pending=pending, count=count)


def stabilize_optional_series(
    values: Any,
    jump: float = LEVEL_JUMP,
    confirm: int = LEVEL_CONFIRM,
) -> list[float | None]:
    """Causal filter for a secondary wall that is allowed to disappear."""
    state = LevelState()
    cleaned: list[float | None] = []
    for value in values:
        number = None if value is None else float(value)
        state = stabilize_optional_level(number, state, jump, confirm)
        cleaned.append(state.accepted)
    return cleaned


def stabilize_series(
    values: Any,
    jump: float = LEVEL_JUMP,
    confirm: int = LEVEL_CONFIRM,
) -> list[float | None]:
    """Causal filter for a column of wall or zero-gamma readings."""
    state = LevelState()
    cleaned: list[float | None] = []
    for value in values:
        state = stabilize_level(None if value is None else float(value), state, jump, confirm)
        cleaned.append(state.accepted)
    return cleaned


def put_wall_strike(exposure: pd.DataFrame) -> float | None:
    """Strike with the most negative put dollar gamma."""
    values = exposure["put_dollar_gamma"].to_numpy(dtype=float)
    if values.size == 0 or not np.isfinite(values).any() or np.nanmin(values) >= 0:
        return None
    return float(exposure["strikePrice"].iloc[int(np.nanargmin(values))])


def zero_gamma_level(
    strikes: Any,
    net_dollar_gamma: Any,
    spot: float | None = None,
) -> float | None:
    """
    Interpolate the gamma-flip pivot.

    Cumulative net dollar gamma is summed from the lowest strike upward.
    The level is the linearly interpolated price where that cumulative sum
    crosses zero. When several crossings exist, the one closest to spot is used.
    """
    strike_array = np.asarray(strikes, dtype=float)
    net = np.asarray(net_dollar_gamma, dtype=float)
    if strike_array.size == 0 or strike_array.size != net.size:
        return None

    order = np.argsort(strike_array)
    strike_array = strike_array[order]
    cumulative = np.cumsum(np.nan_to_num(net[order], nan=0.0))
    levels: list[np.ndarray] = []

    exact = np.flatnonzero(cumulative == 0)
    if exact.size:
        levels.append(strike_array[exact])

    crossings = np.flatnonzero(cumulative[:-1] * cumulative[1:] < 0)
    if crossings.size:
        left = cumulative[crossings]
        right = cumulative[crossings + 1]
        fraction = -left / (right - left)
        levels.append(
            strike_array[crossings]
            + fraction * (strike_array[crossings + 1] - strike_array[crossings])
        )

    if not levels:
        return None

    found = np.concatenate(levels)
    if spot is None or not np.isfinite(spot):
        return float(found[0])
    return float(found[int(np.argmin(np.abs(found - spot)))])


def classify_aggressor(price: float, bid: float, ask: float, prev_price: float) -> int:
    """Lee-Ready side. +1 bought the offer, -1 sold the bid, 0 is unclassified."""
    if not math.isfinite(price) or price <= 0:
        return 0
    if math.isfinite(bid) and math.isfinite(ask) and ask > bid > 0:
        mid = (bid + ask) / 2.0
        if price > mid:
            return 1
        if price < mid:
            return -1
    if math.isfinite(prev_price) and prev_price > 0:
        if price > prev_price:
            return 1
        if price < prev_price:
            return -1
    return 0


def classified_flow_update(
    chain: pd.DataFrame,
    previous: dict[str, float] | None,
) -> tuple[float, dict[str, float]]:
    """Share-delta added since the previous chain, signed by the latest sale.

    The first chain only sets the volume baseline. A print at the midpoint
    with no tick adds nothing, so unclassified volume is not treated as buying.
    """
    baseline: dict[str, float] = {}
    added = 0.0
    if chain is None or chain.empty:
        return 0.0, dict(previous or {})
    for row in chain.itertuples(index=False):
        strike = float(getattr(row, "strikePrice"))
        for right in ("call", "put"):
            volume = float(getattr(row, f"{right}_volume", 0.0) or 0.0)
            delta = float(getattr(row, f"{right}_delta", 0.0) or 0.0)
            last = float(getattr(row, f"{right}_last", 0.0) or 0.0)
            bid = float(getattr(row, f"{right}_bid", 0.0) or 0.0)
            ask = float(getattr(row, f"{right}_ask", 0.0) or 0.0)
            volume_key = f"{strike:.2f}:{right}:v"
            price_key = f"{strike:.2f}:{right}:p"
            baseline[volume_key] = volume
            baseline[price_key] = last
            if previous is None or volume_key not in previous:
                continue
            growth = volume - float(previous[volume_key])
            if growth <= 0:
                continue
            side = classify_aggressor(last, bid, ask, float(previous.get(price_key, 0.0)))
            if side == 0:
                continue
            signed_delta = delta if right == "call" else -abs(delta)
            added += side * growth * signed_delta
    return added, baseline


def cumulative_delta_flow(
    call_volume: Any,
    call_delta: Any,
    put_volume: Any,
    put_delta: Any,
) -> float:
    """
    Session aggregate options delta flow.

    CDF = sum(Call Volume * Call Delta) - sum(Put Volume * abs(Put Delta))
    """
    calls = _finite(call_volume) * np.clip(_finite(call_delta), -1.0, 1.0)
    puts = _finite(put_volume) * np.abs(np.clip(_finite(put_delta), -1.0, 1.0))
    return float(np.sum(calls) - np.sum(puts))


def atm_implied_vols(
    strikes: Any,
    call_iv: Any,
    put_iv: Any,
    spot: float,
) -> tuple[float, float, float]:
    """Call IV, put IV, and their average at the strike nearest spot. Decimal, not percent."""
    strike_array = np.asarray(strikes, dtype=float)
    if strike_array.size == 0 or not np.isfinite(spot):
        return float("nan"), float("nan"), float("nan")
    index = int(np.nanargmin(np.abs(strike_array - spot)))

    def _one(series: Any) -> float:
        value = float(np.asarray(series, dtype=float)[index])
        if np.isfinite(value) and value > 0:
            return value
        return float("nan")

    call = _one(call_iv)
    put = _one(put_iv)
    readings = [value for value in (call, put) if np.isfinite(value)]
    average = float(np.mean(readings)) if readings else float("nan")
    return call, put, average


def atm_implied_volatility(
    strikes: Any,
    call_iv: Any,
    put_iv: Any,
    spot: float,
) -> float:
    """Average implied vol of the call and put nearest to spot. Decimal, not percent."""
    return atm_implied_vols(strikes, call_iv, put_iv, spot)[2]


def momentum_features(bars: pd.DataFrame) -> tuple[float, float, float]:
    """Last price, session VWAP, and the fitted 15-minute move in percent.

    The slope is a linear regression of the last 15 closes. The percent is
    that slope across the window, divided by the last price.
    """
    if bars.empty or "Close" not in bars.columns:
        return float("nan"), float("nan"), float("nan")
    typical = (bars["High"] + bars["Low"] + bars["Close"]) / 3.0
    volume = bars["Volume"].to_numpy(dtype=float)
    typical_values = typical.to_numpy(dtype=float)
    if np.nansum(volume) > 0:
        vwap = float(np.nansum(typical_values * volume) / np.nansum(volume))
    else:
        vwap = float(np.nanmean(typical_values))
    closes = bars["Close"].to_numpy(dtype=float)
    closes = closes[np.isfinite(closes)]
    if closes.size == 0:
        return float("nan"), vwap, float("nan")
    price = float(closes[-1])
    window = closes[-15:]
    if window.size < 2 or price <= 0:
        return price, vwap, float("nan")
    slope = float(np.polyfit(np.arange(window.size, dtype=float), window, 1)[0])
    percent = slope * float(window.size - 1) / price * 100.0
    return price, vwap, percent


def build_intraday_snapshot(
    chain: pd.DataFrame,
    spot: float,
    as_of: datetime | pd.Timestamp,
) -> IntradayGEXSnapshot:
    """Reduce one 0DTE chain to walls, zero gamma, CDF, ATM IV, and charm."""
    exposure = strike_exposure_frame(chain, spot, as_of)
    net = exposure["net_dollar_gamma"].to_numpy(dtype=float)
    minutes_left = _minutes_to_close(as_of)
    iv_strikes, iv_net = iv_net_dollar_gamma(chain, float(spot), as_of, minutes_left)
    call_exposure = oi_dollar_gamma(chain["call_gamma"], chain["call_openInterest"], float(spot))
    put_exposure = oi_dollar_gamma(chain["put_gamma"], chain["put_openInterest"], float(spot))
    call_1, call_1_g, call_2, call_2_g = rank_gamma_walls(chain["strikePrice"], call_exposure)
    put_1, put_1_g, put_2, put_2_g = rank_gamma_walls(chain["strikePrice"], put_exposure)
    call_iv, put_iv, average_iv = atm_implied_vols(
        chain["strikePrice"],
        chain["call_volatility"],
        chain["put_volatility"],
        float(spot),
    )
    return IntradayGEXSnapshot(
        as_of=_as_eastern(as_of),
        spot=float(spot),
        call_wall=call_1,
        put_wall=put_1,
        call_wall_2=call_2,
        put_wall_2=put_2,
        call_wall_1_gamma=call_1_g,
        call_wall_2_gamma=call_2_g,
        put_wall_1_gamma=put_1_g,
        put_wall_2_gamma=put_2_g,
        zero_gamma=zero_gamma_level(iv_strikes, iv_net, spot=float(spot)),
        net_gex=float(np.nansum(net)),
        call_gex=float(np.nansum(exposure["call_dollar_gamma"])),
        put_gex=float(np.nansum(exposure["put_dollar_gamma"])),
        cdf=cumulative_delta_flow(
            chain["call_volume"],
            chain["call_delta"],
            chain["put_volume"],
            chain["put_delta"],
        ),
        atm_iv=average_iv,
        charm_hedge=charm_to_close(chain, float(spot), as_of, minutes_left),
        atm_call_iv=call_iv,
        atm_put_iv=put_iv,
    )


def empty_snapshot_history() -> pd.DataFrame:
    """Time-indexed frame ready for chronological CDF and wall history."""
    return pd.DataFrame(
        columns=list(SNAPSHOT_COLUMNS),
        index=pd.DatetimeIndex([], tz=EASTERN, name="timestamp"),
    )


def _snapshot_frame(snapshot: IntradayGEXSnapshot) -> pd.DataFrame:
    row = {
        "timestamp": snapshot.as_of,
        "spot": snapshot.spot,
        "call_wall": snapshot.call_wall,
        "put_wall": snapshot.put_wall,
        "zero_gamma": snapshot.zero_gamma,
        "net_gex": snapshot.net_gex,
        "call_gex": snapshot.call_gex,
        "put_gex": snapshot.put_gex,
        "cdf": snapshot.cdf,
        "atm_iv": snapshot.atm_iv,
    }
    return pd.DataFrame([row]).set_index("timestamp")


def append_snapshot(
    history: pd.DataFrame,
    snapshot: IntradayGEXSnapshot,
) -> pd.DataFrame:
    """Store one snapshot. A repeated timestamp replaces the previous row."""
    addition = _snapshot_frame(snapshot)
    if history.empty:
        return addition
    combined = pd.concat([history, addition])
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined.sort_index()


def build_snapshot_history(
    records: list[tuple[pd.Timestamp, float, pd.DataFrame]],
) -> pd.DataFrame:
    """Build a chronological snapshot frame from ``(time, spot, chain)`` rows."""
    history = empty_snapshot_history()
    for moment, spot, chain in records:
        history = append_snapshot(history, build_intraday_snapshot(chain, spot, moment))
    return history
