"""
Session VWAP and the expected-move bands the forward test publishes.

VWAP starts at 09:30 ET. The levels drawn on the dashboard are

    width = spot × ATM IV × sqrt(1/252)
    band  = VWAP ± k × width

``calculate_vwap_bands`` still attaches a return-volatility width. The
forward test replaces those four band columns with ``expected_move_bands``.
"""

from __future__ import annotations

import math
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

TRADING_DAYS = 252

EASTERN = ZoneInfo("America/New_York")
RTH_OPEN_MINUTES = 9 * 60 + 30
RTH_CLOSE_MINUTES = 16 * 60


def expected_move(spot: float, atm_iv: float) -> float:
    """One-day one-sigma move: spot × ATM IV × sqrt(1/252).

    ``atm_iv`` is a decimal (0.10 for 10 percent), not a percent.
    """
    if not np.isfinite(spot) or not np.isfinite(atm_iv) or spot <= 0 or atm_iv <= 0:
        return float("nan")
    return float(spot) * float(atm_iv) * math.sqrt(1.0 / TRADING_DAYS)


def expected_move_bands(vwap: float, spot: float, atm_iv: float) -> tuple[float, float, float, float]:
    """VWAP ± 1 and ± 2 expected moves. Returns upper_1, lower_1, upper_2, lower_2."""
    width = expected_move(spot, atm_iv)
    if not np.isfinite(vwap) or not np.isfinite(width):
        nan = float("nan")
        return nan, nan, nan, nan
    return vwap + width, vwap - width, vwap + 2.0 * width, vwap - 2.0 * width


def calculate_vwap_bands(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Return session bars with VWAP and 1σ / 2σ bands.

    Columns added: ``vwap``, ``upper_1``, ``lower_1``, ``upper_2``, ``lower_2``,
    ``return_std``. Premarket and after-hours rows are dropped. Naive timestamps
    are treated as US/Eastern.
    """
    frame = _regular_session(bars)
    typical = (frame["High"] + frame["Low"] + frame["Close"]) / 3.0
    volume = frame["Volume"].astype(float).clip(lower=0.0)
    cumulative_volume = volume.cumsum().replace(0.0, np.nan)
    vwap = ((typical * volume).cumsum() / cumulative_volume).fillna(frame["Close"])

    returns = frame["Close"].pct_change()
    expanding_std = returns.expanding(min_periods=2).std(ddof=1)
    elapsed = np.arange(len(frame), dtype=float)
    cumulative_std = expanding_std.to_numpy(dtype=float) * np.sqrt(elapsed)
    cumulative_std = np.where(np.isfinite(cumulative_std), np.abs(cumulative_std), 0.0)
    price_sigma = vwap.to_numpy(dtype=float) * cumulative_std

    result = frame.copy()
    result["vwap"] = vwap.to_numpy(dtype=float)
    result["return_std"] = np.where(np.isfinite(expanding_std), expanding_std, 0.0)
    result["upper_1"] = result["vwap"] + price_sigma
    result["lower_1"] = result["vwap"] - price_sigma
    result["upper_2"] = result["vwap"] + 2.0 * price_sigma
    result["lower_2"] = result["vwap"] - 2.0 * price_sigma
    return result


def _regular_session(bars: pd.DataFrame) -> pd.DataFrame:
    if bars.empty:
        raise ValueError("bars is empty; cannot calculate session VWAP.")

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [column for column in required if column not in bars.columns]
    if missing:
        raise ValueError(f"bars is missing columns: {', '.join(missing)}")

    frame = bars.loc[:, required].astype(float).copy()
    index = pd.DatetimeIndex(frame.index)
    if index.tz is None:
        index = index.tz_localize(EASTERN)
    else:
        index = index.tz_convert(EASTERN)
    frame.index = index

    session_day = index[-1].date()
    minutes = np.asarray(index.hour) * 60 + np.asarray(index.minute)
    dates = np.array([stamp.date() == session_day for stamp in index])
    frame = frame.loc[dates & (minutes >= RTH_OPEN_MINUTES) & (minutes < RTH_CLOSE_MINUTES)]
    frame = frame.sort_index()
    if frame.empty:
        raise ValueError("No regular-hours bars at or after 09:30 ET.")
    if not np.isfinite(frame["Close"]).all():
        raise ValueError("Close contains non-finite prices.")
    return frame
