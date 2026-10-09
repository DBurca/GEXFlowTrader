"""
Net Gamma Exposure (GEX) calculations, zone clustering, and Phase 2 model skeletons.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import config


@dataclass(frozen=True)
class GEXZone:
    """A clustered price band with aggregated net GEX."""

    price_low: float
    price_high: float
    net_gex: float
    regime: str  # "mean_reverting" | "momentum"


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


def cluster_gex_zones(
    gex_profile: pd.DataFrame,
    resolution: int | None = None,
    spot_price: float | None = None,
) -> list[GEXZone]:
    """
    Group strikes into price zones based on RESOLUTION.

    Lower resolution (1) merges strikes into broad major-wall bands.
    Higher resolution (10) preserves finer, more granular zones.
    """
    resolution = config.RESOLUTION if resolution is None else resolution
    if not (1 <= resolution <= 10):
        raise ValueError(f"resolution must be between 1 and 10 (got {resolution}).")

    strikes = gex_profile["strikePrice"].to_numpy(dtype=float)
    gex_values = gex_profile["net_gex"].to_numpy(dtype=float)

    if len(strikes) == 0:
        return []

    spot = float(spot_price if spot_price is not None else strikes[len(strikes) // 2])
    strike_span = float(strikes.max() - strikes.min()) or spot * 0.05
    # Resolution 1 -> wide bands; resolution 10 -> narrow bands.
    band_width = strike_span / max(resolution * 1.5, 1.0)

    zones: list[GEXZone] = []
    band_start_idx = 0

    for idx in range(1, len(strikes)):
        if strikes[idx] - strikes[band_start_idx] >= band_width:
            zones.append(_build_zone(strikes, gex_values, band_start_idx, idx - 1))
            band_start_idx = idx

    zones.append(_build_zone(strikes, gex_values, band_start_idx, len(strikes) - 1))

    max_abs_gex = max((abs(zone.net_gex) for zone in zones), default=0.0)
    if max_abs_gex == 0:
        return zones

    # At low resolution, keep only the dominant walls.
    keep_count = max(1, min(len(zones), resolution + 1))
    if resolution <= 3:
        zones = sorted(zones, key=lambda z: abs(z.net_gex), reverse=True)[:keep_count]
        zones = sorted(zones, key=lambda z: z.price_low)

    return zones


def _build_zone(
    strikes: np.ndarray,
    gex_values: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> GEXZone:
    net_gex = float(gex_values[start_idx : end_idx + 1].sum())
    regime = "mean_reverting" if net_gex >= 0 else "momentum"
    return GEXZone(
        price_low=float(strikes[start_idx]),
        price_high=float(strikes[end_idx]),
        net_gex=net_gex,
        regime=regime,
    )


def build_gex_summary(
    option_fields: pd.DataFrame,
    spot_price: float,
    resolution: int | None = None,
) -> dict[str, Any]:
    """Run the full GEX pipeline and return profile, total, zones, and regime."""
    profile = calculate_strike_gex(option_fields, spot_price)
    total = calculate_total_net_gex(profile)
    zones = cluster_gex_zones(profile, resolution=resolution, spot_price=spot_price)

    return {
        "profile": profile,
        "total_net_gex": total,
        "regime": classify_regime(total),
        "zones": zones,
    }


# ---------------------------------------------------------------------------
# Phase 2 skeletons
# ---------------------------------------------------------------------------


def fit_hmm_regime(
    total_gex_history: pd.Series,
    n_states: int = 2,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Skeleton: Hidden Markov Model for regime classification from Total GEX history.

    Uses hmmlearn.GaussianHMM to label latent momentum vs. mean-reverting states.
    Wire this up once a historical Total GEX time series is available.
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError as exc:
        raise ImportError(
            "hmmlearn is required for HMM regime modeling. "
            "Install with: pip install hmmlearn"
        ) from exc

    if total_gex_history.empty:
        raise ValueError("total_gex_history must contain at least one observation.")

    observations = total_gex_history.dropna().to_numpy(dtype=float).reshape(-1, 1)
    if len(observations) < n_states:
        raise ValueError(
            f"Need at least {n_states} observations for {n_states}-state HMM."
        )

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        random_state=random_state,
    )
    model.fit(observations)
    hidden_states = model.predict(observations)

    return {
        "model": model,
        "hidden_states": hidden_states,
        "n_states": n_states,
        "status": "skeleton_implemented",
    }


def fit_ou_process(
    series: pd.Series,
    dt: float = 1.0,
) -> dict[str, float]:
    """
    Skeleton: Ornstein-Uhlenbeck parameter estimation for mean reversion.

        dX_t = theta * (mu - X_t) * dt + sigma * dW_t

    Intended for use when Total GEX is highly positive (mean-reverting regime).
    """
    clean = series.dropna().astype(float)
    if len(clean) < 3:
        raise ValueError("Need at least 3 observations to estimate OU parameters.")

    x = clean.to_numpy()
    x_lag = x[:-1]
    x_next = x[1:]

    # AR(1) proxy: X_{t+1} = a + b * X_t  =>  theta = (1-b)/dt, mu = a/(1-b)
    b, a = np.polyfit(x_lag, x_next, 1)
    theta = max((1.0 - b) / dt, 1e-8)
    mu = float(a / max(1.0 - b, 1e-8))

    residuals = x_next - (a + b * x_lag)
    sigma = float(np.std(residuals, ddof=1) / np.sqrt(dt))

    return {
        "theta": float(theta),
        "mu": mu,
        "sigma": sigma,
        "dt": dt,
        "status": "skeleton_implemented",
    }


# ---------------------------------------------------------------------------
# Intraday 0DTE dollar gamma, walls, delta flow, and ATM IV
# ---------------------------------------------------------------------------

EASTERN = ZoneInfo("America/New_York")
OI_WEIGHT_UNTIL = time(10, 30)
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
    zero_gamma: float | None
    net_gex: float
    call_gex: float
    put_gex: float
    cdf: float
    atm_iv: float


def _as_eastern(moment: datetime | pd.Timestamp) -> pd.Timestamp:
    """Interpret naive timestamps as US/Eastern and convert aware ones."""
    stamp = pd.Timestamp(moment)
    if stamp.tzinfo is None:
        return stamp.tz_localize(EASTERN)
    return stamp.tz_convert(EASTERN)


def _finite(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def positioning_weight(
    open_interest: Any,
    volume: Any,
    as_of: datetime | pd.Timestamp,
) -> np.ndarray:
    """
    Intraday positioning blend.

    From the cash open through 10:30 ET the weight is open interest.
    After 10:30 ET it is ``0.4 * OI + 0.6 * intraday volume``.
    """
    oi = _finite(open_interest)
    traded = _finite(volume)
    if oi.shape != traded.shape:
        raise ValueError("open interest and volume must have the same shape.")
    if _as_eastern(as_of).time() <= OI_WEIGHT_UNTIL:
        return oi.copy()
    return 0.4 * oi + 0.6 * traded


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


def call_wall_strike(exposure: pd.DataFrame) -> float | None:
    """Strike with the single highest positive call dollar gamma."""
    values = exposure["call_dollar_gamma"].to_numpy(dtype=float)
    if values.size == 0 or not np.isfinite(values).any() or np.nanmax(values) <= 0:
        return None
    return float(exposure["strikePrice"].iloc[int(np.nanargmax(values))])


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


def atm_implied_volatility(
    strikes: Any,
    call_iv: Any,
    put_iv: Any,
    spot: float,
) -> float:
    """Average implied vol of the call and put nearest to spot. Decimal, not percent."""
    strike_array = np.asarray(strikes, dtype=float)
    if strike_array.size == 0 or not np.isfinite(spot):
        return float("nan")

    index = int(np.nanargmin(np.abs(strike_array - spot)))
    readings: list[float] = []
    for series in (call_iv, put_iv):
        value = float(np.asarray(series, dtype=float)[index])
        if np.isfinite(value) and value > 0:
            readings.append(value)
    if not readings:
        return float("nan")
    return float(np.mean(readings))


def build_intraday_snapshot(
    chain: pd.DataFrame,
    spot: float,
    as_of: datetime | pd.Timestamp,
) -> IntradayGEXSnapshot:
    """Reduce one 0DTE chain to walls, zero gamma, CDF, and ATM IV."""
    exposure = strike_exposure_frame(chain, spot, as_of)
    net = exposure["net_dollar_gamma"].to_numpy(dtype=float)
    return IntradayGEXSnapshot(
        as_of=_as_eastern(as_of),
        spot=float(spot),
        call_wall=call_wall_strike(exposure),
        put_wall=put_wall_strike(exposure),
        zero_gamma=zero_gamma_level(exposure["strikePrice"], net, spot=float(spot)),
        net_gex=float(np.nansum(net)),
        call_gex=float(np.nansum(exposure["call_dollar_gamma"])),
        put_gex=float(np.nansum(exposure["put_dollar_gamma"])),
        cdf=cumulative_delta_flow(
            chain["call_volume"],
            chain["call_delta"],
            chain["put_volume"],
            chain["put_delta"],
        ),
        atm_iv=atm_implied_volatility(
            chain["strikePrice"],
            chain["call_volatility"],
            chain["put_volatility"],
            float(spot),
        ),
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
