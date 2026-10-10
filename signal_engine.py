"""Macro gamma-flip level from a strike GEX profile."""

from __future__ import annotations

import pandas as pd


def calculate_gamma_flip(gex_profile: pd.DataFrame) -> float | None:
    """
    Price where cumulative net GEX changes sign.

    Linear interpolation between the two strikes that straddle the cross.
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
