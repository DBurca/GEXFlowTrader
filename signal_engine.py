"""
Gamma Flip detection and momentum/mean-reversion signal generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

import config
from gex_engine import GEXZone


@dataclass(frozen=True)
class TradingSignal:
    """Output of the signal engine for a single evaluation bar."""

    triggered: bool
    signal_type: str | None  # "momentum_short" | None
    gamma_flip: float | None
    trigger_price: float | None
    message: str


def calculate_gamma_flip(gex_profile: pd.DataFrame) -> float | None:
    """
    Find the price where cumulative net GEX crosses from positive to negative.

    Uses linear interpolation between adjacent strikes when a sign change occurs.
    """
    if gex_profile.empty:
        return None

    frame = gex_profile.sort_values("strikePrice").copy()
    frame["cumulative_gex"] = frame["net_gex"].cumsum()

    strikes = frame["strikePrice"].to_numpy(dtype=float)
    cumulative = frame["cumulative_gex"].to_numpy(dtype=float)

    for idx in range(1, len(cumulative)):
        prev_cum = cumulative[idx - 1]
        curr_cum = cumulative[idx]

        if prev_cum == 0:
            return float(strikes[idx - 1])
        if curr_cum == 0:
            return float(strikes[idx])

        if prev_cum > 0 and curr_cum < 0:
            weight = prev_cum / (prev_cum - curr_cum)
            flip = strikes[idx - 1] + weight * (strikes[idx] - strikes[idx - 1])
            return float(flip)

        if prev_cum < 0 and curr_cum > 0:
            weight = -prev_cum / (curr_cum - prev_cum)
            flip = strikes[idx - 1] + weight * (strikes[idx] - strikes[idx - 1])
            return float(flip)

    return None


def evaluate_gamma_flip_signal(
    ohlcv: pd.DataFrame,
    gamma_flip: float | None,
    require_confirmation: bool | None = None,
) -> TradingSignal:
    """
    Generate a momentum-short signal when price crosses below the Gamma Flip.

    require_confirmation=False -> intraday Low breach triggers immediately.
    require_confirmation=True  -> daily Close must finish below the flip level.
    """
    require_confirmation = (
        config.REQUIRE_CONFIRMATION
        if require_confirmation is None
        else require_confirmation
    )

    if gamma_flip is None:
        return TradingSignal(
            triggered=False,
            signal_type=None,
            gamma_flip=None,
            trigger_price=None,
            message="Gamma Flip could not be determined from the GEX profile.",
        )

    if len(ohlcv) < 2:
        return TradingSignal(
            triggered=False,
            signal_type=None,
            gamma_flip=gamma_flip,
            trigger_price=None,
            message="Insufficient OHLCV history to evaluate a cross.",
        )

    prior = ohlcv.iloc[-2]
    latest = ohlcv.iloc[-1]

    if require_confirmation:
        crossed = float(prior["Close"]) >= gamma_flip and float(latest["Close"]) < gamma_flip
        trigger_price = float(latest["Close"])
        detail = f"daily close {trigger_price:.2f} below Gamma Flip {gamma_flip:.2f}"
    else:
        crossed = float(prior["Low"]) >= gamma_flip and float(latest["Low"]) < gamma_flip
        trigger_price = float(latest["Low"])
        detail = f"intraday low {trigger_price:.2f} crossed below Gamma Flip {gamma_flip:.2f}"

    if crossed:
        return TradingSignal(
            triggered=True,
            signal_type="momentum_short",
            gamma_flip=gamma_flip,
            trigger_price=trigger_price,
            message=f"Momentum short trigger: {detail}",
        )

    return TradingSignal(
        triggered=False,
        signal_type=None,
        gamma_flip=gamma_flip,
        trigger_price=None,
        message=f"No cross below Gamma Flip ({gamma_flip:.2f}) on latest bar.",
    )


def build_signal_summary(
    ohlcv: pd.DataFrame,
    gex_profile: pd.DataFrame,
    zones: list[GEXZone] | None = None,
    require_confirmation: bool | None = None,
) -> dict[str, Any]:
    """Compute Gamma Flip and evaluate the latest trading signal."""
    gamma_flip = calculate_gamma_flip(gex_profile)
    signal = evaluate_gamma_flip_signal(
        ohlcv,
        gamma_flip,
        require_confirmation=require_confirmation,
    )

    return {
        "gamma_flip": gamma_flip,
        "signal": signal,
        "zones": zones or [],
    }
