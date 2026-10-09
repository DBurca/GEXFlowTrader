"""
User inputs and app-wide constants for the Macro GEX Prediction & Regime Tool.
"""

from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path
from typing import Final, TypedDict

# ---------------------------------------------------------------------------
# Schwab API credentials
# Set via environment variables. Do not hardcode secrets in this file.
# ---------------------------------------------------------------------------
SCHWAB_API_KEY: str = os.getenv("SCHWAB_API_KEY", "")
SCHWAB_API_SECRET: str = os.getenv("SCHWAB_API_SECRET", "")
SCHWAB_ACCESS_TOKEN: str = os.getenv("SCHWAB_ACCESS_TOKEN", "")
SCHWAB_CALLBACK_URL: str = os.getenv("SCHWAB_CALLBACK_URL", "https://127.0.0.1:8080")
TOKEN_FILE_PATH: str = os.getenv(
    "SCHWAB_TOKEN_FILE",
    str(Path(__file__).resolve().parent / "tokens.json"),
)

# ---------------------------------------------------------------------------
# Schwab Market Data API
# Logical names from the project spec map to Schwab's REST paths.
# ---------------------------------------------------------------------------
SCHWAB_BASE_URL: Final[str] = "https://api.schwabapi.com/marketdata/v1"
PRICES_ENDPOINT: Final[str] = "/pricehistory"  # spec: /markets/prices
OPTIONS_CHAINS_ENDPOINT: Final[str] = "/chains"  # spec: /markets/options/chains

# ---------------------------------------------------------------------------
# User inputs
# ---------------------------------------------------------------------------
TICKER: str = "SPCX"
TIME_INTERVAL: str = "15m"  # e.g. '1D', '1W', '60m'

DATE_RANGE: dict[str, date | str] = {
    "start": "2026-06-01",
    "end": date.today().isoformat(),
}

# 1 = major GEX walls only, 10 = highly granular strike clustering
RESOLUTION: int = 5

# True  -> signals require a daily close beyond a GEX wall
# False -> signals trigger on a simple touch
REQUIRE_CONFIRMATION: bool = True

# ---------------------------------------------------------------------------
# Options chain filtering (30–90 DTE removes near-term intraday noise)
# ---------------------------------------------------------------------------
MIN_DTE: Final[int] = 30
MAX_DTE: Final[int] = 90

# ---------------------------------------------------------------------------
# Internal mappings (not user-facing)
# ---------------------------------------------------------------------------
class _IntervalParams(TypedDict):
    frequency_type: str
    frequency: int


TIME_INTERVAL_MAP: dict[str, _IntervalParams] = {
    "1m": {"frequency_type": "minute", "frequency": 1},
    "5m": {"frequency_type": "minute", "frequency": 5},
    "15m": {"frequency_type": "minute", "frequency": 15},
    "30m": {"frequency_type": "minute", "frequency": 30},
    "60m": {"frequency_type": "minute", "frequency": 60},
    "1D": {"frequency_type": "daily", "frequency": 1},
    "1W": {"frequency_type": "weekly", "frequency": 1},
    "1M": {"frequency_type": "monthly", "frequency": 1},
}


def parse_date(value: date | str | datetime) -> date:
    """Normalize config date values to datetime.date."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(value, "%Y-%m-%d").date()


def validate_config() -> None:
    """Raise ValueError when required settings are missing or invalid."""
    import auth_tokens

    try:
        auth_tokens.get_access_token()
    except ValueError as exc:
        raise ValueError(str(exc)) from exc

    if not (1 <= RESOLUTION <= 10):
        raise ValueError(f"RESOLUTION must be between 1 and 10 (got {RESOLUTION}).")

    if TIME_INTERVAL not in TIME_INTERVAL_MAP:
        supported = ", ".join(sorted(TIME_INTERVAL_MAP))
        raise ValueError(
            f"Unsupported TIME_INTERVAL '{TIME_INTERVAL}'. Supported: {supported}"
        )

    start = parse_date(DATE_RANGE["start"])
    end = parse_date(DATE_RANGE["end"])
    if start > end:
        raise ValueError(
            f"DATE_RANGE start ({start}) must be on or before end ({end})."
        )
