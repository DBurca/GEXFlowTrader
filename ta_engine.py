"""
Session VWAP and the expected-move bands the forward test publishes.

VWAP starts at 09:30 ET. The levels drawn on the dashboard are the
expected move still left in the session:

    width = spot × ATM IV × sqrt(minutes left / (390 × 252))
    band  = VWAP ± k × width

``calculate_vwap_bands`` still attaches a return-volatility width. The
forward test replaces those four band columns with ``expected_move_bands``.
"""

from __future__ import annotations

import math
from datetime import datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

TRADING_DAYS = 252
SESSION_MINUTES = 390

EASTERN = ZoneInfo("America/New_York")
RTH_OPEN_MINUTES = 9 * 60 + 30
RTH_CLOSE_MINUTES = 16 * 60


def minutes_remaining(moment: datetime, close_at: time) -> float:
    """Whole minutes from ``moment`` until the session close, never negative."""
    now_minutes = moment.hour * 60 + moment.minute
    close_minutes = close_at.hour * 60 + close_at.minute
    return float(max(0, close_minutes - now_minutes))


def expected_move(spot: float, atm_iv: float, minutes_left: float = SESSION_MINUTES) -> float:
    """One-sigma move still left in the session.

    ``spot × ATM IV × sqrt(minutes_left / (390 × 252))``. A full 390-minute
    session is the old one-day move. ``atm_iv`` is a decimal.
    """
    if (
        not np.isfinite(spot)
        or not np.isfinite(atm_iv)
        or not np.isfinite(minutes_left)
        or spot <= 0
        or atm_iv <= 0
        or minutes_left <= 0
    ):
        return 0.0 if np.isfinite(minutes_left) and minutes_left <= 0 else float("nan")
    fraction = max(float(minutes_left), 0.0) / SESSION_MINUTES
    return float(spot) * float(atm_iv) * math.sqrt(fraction / TRADING_DAYS)


def expected_move_bands(
    vwap: float,
    spot: float,
    atm_iv: float,
    minutes_left: float = SESSION_MINUTES,
) -> tuple[float, float, float, float]:
    """VWAP ± 1 and ± 2 remaining-session moves. Returns upper_1, lower_1, upper_2, lower_2."""
    width = expected_move(spot, atm_iv, minutes_left)
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
