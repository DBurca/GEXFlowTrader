"""
Schwab API integration for historical OHLCV and options chain data.

All requests hit the API directly on each call — no caching layer.
"""

from __future__ import annotations

import time
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

import pandas as pd
import requests
from requests.exceptions import ConnectionError, ReadTimeout

import config
from auth_tokens import get_access_token

PRICE_HISTORY_TIMEOUT_SECONDS = 60
OPTIONS_CHAIN_TIMEOUT_SECONDS = 120
MAX_REQUEST_RETRIES = 3


class SchwabAPIError(RuntimeError):
    """Raised when a Schwab API request fails."""


def _auth_headers() -> dict[str, str]:
    config.validate_config()
    return {
        "Authorization": f"Bearer {get_access_token()}",
        "Accept": "application/json",
    }


def _request_json(
    endpoint: str,
    params: dict[str, Any] | None = None,
    *,
    timeout: int = PRICE_HISTORY_TIMEOUT_SECONDS,
    retries: int = MAX_REQUEST_RETRIES,
) -> dict[str, Any]:
    url = f"{config.SCHWAB_BASE_URL}{endpoint}"
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(
                url,
                headers=_auth_headers(),
                params=params,
                timeout=timeout,
            )
        except (ReadTimeout, ConnectionError) as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(2 ** (attempt - 1))
            continue

        if not response.ok:
            raise SchwabAPIError(
                f"Schwab API error {response.status_code} for {endpoint}: {response.text}"
            )

        payload = response.json()
        if not isinstance(payload, dict):
            raise SchwabAPIError(
                f"Unexpected response type from {endpoint}: {type(payload)}"
            )
        return payload

    raise SchwabAPIError(
        f"Schwab API request timed out for {endpoint} after {retries} attempts."
    ) from last_error


def _to_epoch_ms(value: date | datetime, *, end_of_day: bool = False) -> int:
    if isinstance(value, datetime):
        dt = value
    elif end_of_day:
        dt = datetime.combine(value, time(23, 59, 59), tzinfo=timezone.utc)
    else:
        dt = datetime.combine(value, time.min, tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _build_price_history_params(
    symbol: str,
    start_date: date,
    end_date: date,
    interval: str,
) -> dict[str, Any]:
    """
    Build Schwab-compatible /pricehistory query parameters.

    Schwab validates periodType against frequencyType. Date-only requests can
    default to periodType=day, which rejects daily bars. We follow schwab-py's
    pairings and bound results with startDate/endDate.
    """
    interval_params = config.TIME_INTERVAL_MAP[interval]
    frequency_type = interval_params["frequency_type"]
    frequency = interval_params["frequency"]

    params: dict[str, Any] = {
        "symbol": symbol,
        "frequencyType": frequency_type,
        "frequency": frequency,
        "startDate": _to_epoch_ms(start_date),
        "endDate": _to_epoch_ms(end_date, end_of_day=True),
        "needExtendedHoursData": False,
    }

    if frequency_type == "minute":
        params["periodType"] = "day"
        params["period"] = 1
    else:
        years_span = max(1, (end_date - start_date).days // 365 + 1)
        params["periodType"] = "year"
        params["period"] = min(20, years_span)

    return params


def _candles_to_ohlcv_frame(candles: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(candles)
    frame["datetime"] = pd.to_datetime(frame["datetime"], unit="ms", utc=True)
    frame = frame.set_index("datetime").sort_index()
    frame = frame.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    return frame[["Open", "High", "Low", "Close", "Volume"]]


def fetch_historical_ohlcv(
    ticker: str | None = None,
    start: date | str | None = None,
    end: date | str | None = None,
    interval: str | None = None,
) -> pd.DataFrame:
    """
    Pull historical OHLCV candles from Schwab /pricehistory (spec: /markets/prices).

    Returns a DataFrame indexed by datetime with columns:
    Open, High, Low, Close, Volume.
    """
    ticker = ticker or config.TICKER
    start_date = config.parse_date(start or config.DATE_RANGE["start"])
    end_date = config.parse_date(end or config.DATE_RANGE["end"])
    interval = interval or config.TIME_INTERVAL

    if interval not in config.TIME_INTERVAL_MAP:
        raise ValueError(f"Unsupported interval '{interval}'.")

    params = _build_price_history_params(
        symbol=ticker,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
    )
    payload = _request_json(config.PRICES_ENDPOINT, params=params)

    candles = payload.get("candles", [])
    if not candles:
        raise SchwabAPIError(f"No OHLCV candles returned for {ticker}.")

    frame = _candles_to_ohlcv_frame(candles)

    start_ts = pd.Timestamp(_to_epoch_ms(start_date), unit="ms", tz="UTC")
    end_ts = pd.Timestamp(_to_epoch_ms(end_date, end_of_day=True), unit="ms", tz="UTC")
    frame = frame.loc[start_ts:end_ts]

    if frame.empty:
        raise SchwabAPIError(
            f"No OHLCV candles for {ticker} between {start_date} and {end_date}."
        )

    return frame


def _dte_date_window(
    min_dte: int | None = None,
    max_dte: int | None = None,
) -> tuple[str, str]:
    """Return Schwab fromDate/toDate strings for the configured DTE window."""
    min_dte = config.MIN_DTE if min_dte is None else min_dte
    max_dte = config.MAX_DTE if max_dte is None else max_dte
    today = date.today()
    from_date = today + timedelta(days=min_dte)
    to_date = today + timedelta(days=max_dte)
    return from_date.isoformat(), to_date.isoformat()


def fetch_options_chain(
    ticker: str | None = None,
    min_dte: int | None = None,
    max_dte: int | None = None,
) -> dict[str, Any]:
    """
    Pull the options chain from Schwab /chains (spec: /markets/options/chains).

    Requests only the 30-90 DTE expiration window at the API level so large
    symbols like SPY return in reasonable time. Uses strategy=ANALYTICAL for gamma.
    """
    ticker = ticker or config.TICKER
    from_date, to_date = _dte_date_window(min_dte=min_dte, max_dte=max_dte)

    return _request_json(
        config.OPTIONS_CHAINS_ENDPOINT,
        params={
            "symbol": ticker,
            "contractType": "ALL",
            "strategy": "ANALYTICAL",
            "range": "ALL",
            "includeUnderlyingQuote": True,
            "fromDate": from_date,
            "toDate": to_date,
        },
        timeout=OPTIONS_CHAIN_TIMEOUT_SECONDS,
    )


def _parse_expiration_key(expiration_key: str) -> tuple[date, int | None]:
    """
    Parse Schwab expiration map keys like '2025-07-18:45'.

    Returns (expiration_date, days_to_expiration or None).
    """
    date_part, _, dte_part = expiration_key.partition(":")
    expiration = datetime.strptime(date_part, "%Y-%m-%d").date()
    dte = int(dte_part) if dte_part.isdigit() else None
    return expiration, dte


def _contract_dte(contract: dict[str, Any], expiration_key: str) -> int | None:
    dte = contract.get("daysToExpiration")
    if dte is not None:
        return int(dte)

    _, key_dte = _parse_expiration_key(expiration_key)
    return key_dte


def filter_options_by_dte(
    chain: dict[str, Any],
    min_dte: int | None = None,
    max_dte: int | None = None,
) -> dict[str, Any]:
    """
    Return a shallow copy of the chain containing only expirations within [min_dte, max_dte].
    """
    min_dte = config.MIN_DTE if min_dte is None else min_dte
    max_dte = config.MAX_DTE if max_dte is None else max_dte

    filtered = dict(chain)
    for map_name in ("callExpDateMap", "putExpDateMap"):
        exp_map = chain.get(map_name, {})
        filtered_map: dict[str, Any] = {}

        for expiration_key, strikes in exp_map.items():
            sample_contract = next(
                (contracts[0] for contracts in strikes.values() if contracts),
                None,
            )
            if sample_contract is None:
                continue

            dte = _contract_dte(sample_contract, expiration_key)
            if dte is None or not (min_dte <= dte <= max_dte):
                continue
            filtered_map[expiration_key] = strikes

        filtered[map_name] = filtered_map

    return filtered


def _aggregate_strike_rows(
    exp_date_map: dict[str, Any],
    option_type: str,
) -> dict[float, dict[str, float]]:
    """
    Aggregate open interest and gamma by strike across all expirations in a map.
    """
    aggregated: dict[float, dict[str, float]] = {}

    for expiration_key, strike_map in exp_date_map.items():
        for strike_key, contracts in strike_map.items():
            if not contracts:
                continue

            contract = contracts[0]
            strike = float(contract.get("strikePrice", strike_key))
            open_interest = float(contract.get("openInterest") or 0.0)
            gamma = float(contract.get("gamma") or 0.0)

            if strike not in aggregated:
                aggregated[strike] = {
                    "openInterest": 0.0,
                    "gamma": 0.0,
                    "gamma_oi_weighted_sum": 0.0,
                }

            bucket = aggregated[strike]
            bucket["openInterest"] += open_interest
            bucket["gamma_oi_weighted_sum"] += gamma * open_interest

    for strike, bucket in aggregated.items():
        oi = bucket["openInterest"]
        bucket["gamma"] = (
            bucket["gamma_oi_weighted_sum"] / oi if oi > 0 else 0.0
        )
        bucket["option_type"] = option_type  # type: ignore[assignment]

    return aggregated


def extract_option_fields(
    chain: dict[str, Any],
    min_dte: int | None = None,
    max_dte: int | None = None,
) -> pd.DataFrame:
    """
    Filter the chain to 30–90 DTE and extract strike-level call/put inputs for GEX.

    Returns a DataFrame with columns:
        strikePrice, call_openInterest, call_gamma, put_openInterest, put_gamma
    One row per strike, aggregated across qualifying expirations.
    """
    filtered_chain = filter_options_by_dte(chain, min_dte=min_dte, max_dte=max_dte)

    calls = _aggregate_strike_rows(filtered_chain.get("callExpDateMap", {}), "call")
    puts = _aggregate_strike_rows(filtered_chain.get("putExpDateMap", {}), "put")

    all_strikes = sorted(set(calls) | set(puts))
    rows: list[dict[str, float]] = []

    for strike in all_strikes:
        call_data = calls.get(strike, {})
        put_data = puts.get(strike, {})
        rows.append(
            {
                "strikePrice": strike,
                "call_openInterest": call_data.get("openInterest", 0.0),
                "call_gamma": call_data.get("gamma", 0.0),
                "put_openInterest": put_data.get("openInterest", 0.0),
                "put_gamma": put_data.get("gamma", 0.0),
            }
        )

    return pd.DataFrame(rows)


def get_underlying_spot(chain: dict[str, Any]) -> float:
    """Return the current underlying spot price from an options chain payload."""
    underlying_price = chain.get("underlyingPrice")
    if underlying_price is not None:
        return float(underlying_price)

    underlying = chain.get("underlying") or {}
    for field in ("mark", "last", "close"):
        value = underlying.get(field)
        if value is not None:
            return float(value)

    raise SchwabAPIError("Could not determine underlying spot price from options chain.")


def load_market_data(
    ticker: str | None = None,
) -> dict[str, Any]:
    """
    Convenience loader used by downstream engines.

    Returns:
        ohlcv: Historical OHLCV DataFrame
        options_chain: Raw filtered options chain (30–90 DTE expirations only)
        option_fields: Strike-level call/put OI and gamma DataFrame
        spot_price: Current underlying price
    """
    ticker = ticker or config.TICKER
    print(f"  Fetching OHLCV for {ticker}...")
    ohlcv = fetch_historical_ohlcv(ticker=ticker)
    print(f"  Fetching options chain for {ticker} ({config.MIN_DTE}-{config.MAX_DTE} DTE)...")
    raw_chain = fetch_options_chain(ticker=ticker)
    filtered_chain = filter_options_by_dte(raw_chain)
    option_fields = extract_option_fields(raw_chain)
    spot_price = get_underlying_spot(raw_chain)

    return {
        "ohlcv": ohlcv,
        "options_chain": filtered_chain,
        "option_fields": option_fields,
        "spot_price": spot_price,
    }
