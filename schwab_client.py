"""
Schwab authentication and intraday market-data fetchers.

Tokens are loaded from ``tokens.json``. An access token older than 28 minutes
is refreshed before the next request. Option chains are polled at most once
every 180 seconds, and only during regular trading hours. One-minute spot
bars are polled at most once every 30 seconds.

``generate_simulated_session`` builds a full cash session locally so the
dashboard can be checked when the market is closed.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import date, datetime, time as clock_time
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from requests.exceptions import RequestException

import config
from auth_tokens import load_token_file, save_token_file

EASTERN = ZoneInfo("America/New_York")
SCHWAB_TOKEN_URL = "https://api.schwabapi.com/v1/oauth/token"
ACCESS_TOKEN_MAX_AGE_SECONDS = 28 * 60
CHAIN_POLL_SECONDS = 180
SPOT_POLL_SECONDS = 30
STRIKE_WINDOW = 0.02
RTH_OPEN = clock_time(9, 30)
RTH_CLOSE = clock_time(16, 0)
MAX_RETRIES = 3
QUOTE_TIMEOUT_SECONDS = 30
BARS_TIMEOUT_SECONDS = 45
CHAIN_TIMEOUT_SECONDS = 60
RETRYABLE_STATUS = {429, 500, 502, 503, 504}

INDEX_STRIKE_STEP = 5.0
DEFAULT_STRIKE_STEP = 1.0
INDEX_SYMBOLS = {"SPX", "$SPX", "SPXW", "$SPXW", "NDX", "$NDX", "RUT", "$RUT"}

CHAIN_COLUMNS = (
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


class SchwabClientError(RuntimeError):
    """Base error for intraday Schwab requests."""


class SchwabAuthError(SchwabClientError):
    """Raised when a usable access token cannot be obtained."""


class SchwabConnectionError(SchwabClientError):
    """Raised when the API cannot be reached after retries."""


class EmptyOptionChainError(SchwabClientError):
    """Raised when the 0DTE chain is missing or has no strikes in range."""


class NoSessionBarsError(SchwabClientError):
    """Raised when today's regular-hours 1-minute bars are not available."""


class MarketClosedError(SchwabClientError):
    """Raised when a chain poll is requested outside regular trading hours."""


@dataclass(frozen=True)
class OptionChainFetch:
    """A 0DTE chain plus the spot used to window it."""

    contracts: pd.DataFrame
    spot: float
    refreshed: bool


def in_regular_trading_hours(moment: datetime | None = None) -> bool:
    """True from 09:30 through 16:00 ET on weekdays, inclusive of the close."""
    current = moment or datetime.now(EASTERN)
    if current.tzinfo is None:
        current = current.replace(tzinfo=EASTERN)
    else:
        current = current.astimezone(EASTERN)
    if current.weekday() >= 5:
        return False
    return RTH_OPEN <= current.time() <= RTH_CLOSE


def _eastern_now() -> datetime:
    return datetime.now(EASTERN)


def _session_date(moment: datetime | None = None) -> date:
    current = moment or _eastern_now()
    if current.tzinfo is None:
        return current.date()
    return current.astimezone(EASTERN).date()


def _strike_step(symbol: str) -> float:
    if symbol.upper() in INDEX_SYMBOLS:
        return INDEX_STRIKE_STEP
    return DEFAULT_STRIKE_STEP


def _number(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def _first_contract(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                return item
        return None
    if isinstance(raw, dict):
        return raw
    return None


def _parse_expiration_key(expiration_key: str) -> tuple[date, int | None]:
    date_part, _, dte_part = expiration_key.partition(":")
    expiration = datetime.strptime(date_part, "%Y-%m-%d").date()
    dte = int(dte_part) if dte_part.isdigit() else None
    return expiration, dte


class TokenManager:
    """Load ``tokens.json`` and refresh access tokens older than 28 minutes."""

    def __init__(self, max_age_seconds: int = ACCESS_TOKEN_MAX_AGE_SECONDS) -> None:
        self.max_age_seconds = max_age_seconds

    def get_access_token(self, *, force: bool = False) -> str:
        record = load_token_file()
        if record is None:
            if config.SCHWAB_ACCESS_TOKEN and not force:
                return config.SCHWAB_ACCESS_TOKEN
            raise SchwabAuthError(
                "No Schwab token file found. Run `python keygetter.py` "
                "or export SCHWAB_ACCESS_TOKEN."
            )

        access_token = record.get("access_token")
        if not force and access_token and not self._needs_refresh(record):
            return str(access_token)

        refresh_token = record.get("refresh_token")
        if not refresh_token:
            if access_token and not self._is_expired(record) and not force:
                return str(access_token)
            raise SchwabAuthError(
                "Access token is older than 28 minutes and no refresh token "
                "is available. Run `python keygetter.py` again."
            )

        try:
            payload = self._refresh(str(refresh_token))
        except SchwabAuthError:
            if access_token and not self._is_expired(record) and not force:
                return str(access_token)
            raise
        return str(payload["access_token"])

    def _needs_refresh(self, record: dict[str, Any]) -> bool:
        updated_at = record.get("updated_at")
        if updated_at is not None:
            age = time.time() - float(updated_at)
            return age >= self.max_age_seconds
        return self._is_expired(record)

    @staticmethod
    def _is_expired(record: dict[str, Any]) -> bool:
        expires_at = record.get("expires_at")
        if expires_at is None:
            return False
        return time.time() >= float(expires_at) - 30

    def _refresh(self, refresh_token: str) -> dict[str, Any]:
        if not config.SCHWAB_API_KEY or not config.SCHWAB_API_SECRET:
            raise SchwabAuthError(
                "SCHWAB_API_KEY and SCHWAB_API_SECRET are required to refresh tokens."
            )

        credentials = _basic_credentials(config.SCHWAB_API_KEY, config.SCHWAB_API_SECRET)
        try:
            response = requests.post(
                SCHWAB_TOKEN_URL,
                headers={
                    "Authorization": f"Basic {credentials}",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                },
                timeout=QUOTE_TIMEOUT_SECONDS,
            )
        except RequestException as exc:
            raise SchwabConnectionError(
                f"Token refresh request failed: {exc.__class__.__name__}"
            ) from exc

        if not response.ok:
            raise SchwabAuthError(
                f"Token refresh failed ({response.status_code}). "
                "Run `python keygetter.py` to re-authenticate."
            )

        payload = response.json()
        if not isinstance(payload, dict) or "access_token" not in payload:
            raise SchwabAuthError("Token refresh returned an unexpected payload.")
        save_token_file(payload)
        return payload


def _basic_credentials(client_id: str, client_secret: str) -> str:
    import base64

    return base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()


class SchwabIntradayClient:
    """Rate-limited quote, 1-minute bar, and 0DTE chain client."""

    def __init__(self, tokens: TokenManager | None = None) -> None:
        self.tokens = tokens or TokenManager()
        self.session = requests.Session()
        self._bars_cache: dict[str, pd.DataFrame] = {}
        self._bars_mono: dict[str, float] = {}
        self._chain_cache: dict[str, tuple[pd.DataFrame, float]] = {}
        self._chain_mono: dict[str, float] = {}

    def fetch_spot_quote(self, symbol: str) -> float:
        """Latest last price for ``symbol``."""
        symbol = symbol.upper()
        payload = self._request_json(
            "/quotes",
            params={"symbols": symbol},
            timeout=QUOTE_TIMEOUT_SECONDS,
        )
        return _extract_last_price(payload, symbol)

    def fetch_intraday_bars(self, symbol: str, *, force: bool = False) -> pd.DataFrame:
        """
        Today's regular-hours 1-minute bars.

        Repeat calls inside the 30-second window return the cached frame.
        """
        symbol = symbol.upper()
        cached = self._bars_cache.get(symbol)
        fetched_at = self._bars_mono.get(symbol, 0.0)
        if cached is not None and not force and time.monotonic() - fetched_at < SPOT_POLL_SECONDS:
            return cached.copy()

        payload = self._request_json(
            config.PRICES_ENDPOINT,
            params={
                "symbol": symbol,
                "periodType": "day",
                "period": 1,
                "frequencyType": "minute",
                "frequency": 1,
                "needExtendedHoursData": False,
            },
            timeout=BARS_TIMEOUT_SECONDS,
        )
        candles = payload.get("candles") or []
        if not isinstance(candles, list) or not candles:
            raise NoSessionBarsError(f"No 1-minute candles returned for {symbol}.")

        frame = _filter_rth_today(_bars_from_candles(candles))
        self._bars_cache[symbol] = frame
        self._bars_mono[symbol] = time.monotonic()
        return frame.copy()

    def fetch_0dte_chain(
        self,
        symbol: str,
        spot: float | None = None,
        *,
        force: bool = False,
    ) -> OptionChainFetch:
        """
        Today's expiration, limited to strikes within ±2% of spot.

        Network polls are skipped outside 09:30–16:00 ET and when the previous
        chain is younger than 180 seconds. A cached chain is returned instead,
        with ``refreshed=False``.
        """
        symbol = symbol.upper()
        cached = self._chain_cache.get(symbol)
        fetched_at = self._chain_mono.get(symbol, 0.0)
        market_open = in_regular_trading_hours()

        if not market_open:
            if cached is not None:
                frame, cached_spot = cached
                return OptionChainFetch(frame.copy(), cached_spot, refreshed=False)
            raise MarketClosedError(
                "Option-chain polling runs during regular trading hours only "
                "(09:30–16:00 ET)."
            )

        if cached is not None and not force and time.monotonic() - fetched_at < CHAIN_POLL_SECONDS:
            frame, cached_spot = cached
            return OptionChainFetch(frame.copy(), cached_spot, refreshed=False)

        if spot is None or not math.isfinite(spot) or spot <= 0:
            spot = self.fetch_spot_quote(symbol)

        session_day = _session_date().isoformat()
        strike_count = int(math.ceil((spot * STRIKE_WINDOW) / _strike_step(symbol))) + 3
        payload = self._request_json(
            config.OPTIONS_CHAINS_ENDPOINT,
            params={
                "symbol": symbol,
                "contractType": "ALL",
                "strategy": "SINGLE",
                "range": "ALL",
                "includeUnderlyingQuote": True,
                "strikeCount": strike_count,
                "fromDate": session_day,
                "toDate": session_day,
            },
            timeout=CHAIN_TIMEOUT_SECONDS,
        )
        status = str(payload.get("status") or "SUCCESS").upper()
        if status not in {"SUCCESS"}:
            raise SchwabClientError(f"Option chain status for {symbol}: {status}")

        chain_spot = _underlying_price(payload) or float(spot)
        contracts = _parse_0dte_chain(payload, chain_spot, session_day=_session_date())
        self._chain_cache[symbol] = (contracts, chain_spot)
        self._chain_mono[symbol] = time.monotonic()
        return OptionChainFetch(contracts.copy(), chain_spot, refreshed=True)

    def _request_json(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: int,
    ) -> dict[str, Any]:
        url = f"{config.SCHWAB_BASE_URL}{endpoint}"
        refreshed = False
        last_error: Exception | None = None

        attempt = 0
        while attempt < MAX_RETRIES:
            attempt += 1
            try:
                response = self.session.get(
                    url,
                    headers={
                        "Authorization": f"Bearer {self.tokens.get_access_token()}",
                        "Accept": "application/json",
                    },
                    params=params,
                    timeout=timeout,
                )
            except RequestException as exc:
                last_error = exc
                if attempt == MAX_RETRIES:
                    break
                time.sleep(2 ** (attempt - 1))
                continue

            if response.status_code == 401 and not refreshed:
                self.tokens.get_access_token(force=True)
                refreshed = True
                attempt -= 1
                continue

            if response.status_code in RETRYABLE_STATUS and attempt < MAX_RETRIES:
                time.sleep(2 ** (attempt - 1))
                continue

            if not response.ok:
                detail = response.text[:240].replace("\n", " ")
                raise SchwabClientError(
                    f"Schwab API error {response.status_code} for {endpoint}: {detail}"
                )

            try:
                payload = response.json()
            except ValueError as exc:
                raise SchwabClientError(
                    f"Schwab returned a non-JSON response for {endpoint}."
                ) from exc

            if not isinstance(payload, dict):
                raise SchwabClientError(
                    f"Unexpected response type from {endpoint}: {type(payload).__name__}"
                )
            return payload

        raise SchwabConnectionError(
            f"Schwab API request failed for {endpoint} after {MAX_RETRIES} attempts "
            f"({last_error.__class__.__name__ if last_error else 'unknown'})."
        ) from last_error


def _extract_last_price(payload: dict[str, Any], symbol: str) -> float:
    node: Any = payload.get(symbol)
    if node is None:
        for key, value in payload.items():
            if str(key).upper() == symbol.upper():
                node = value
                break
    if not isinstance(node, dict):
        node = payload

    candidates: list[Any] = []
    quote = node.get("quote")
    if isinstance(quote, dict):
        candidates.extend(quote.get(key) for key in ("lastPrice", "mark", "last"))
    regular = node.get("regular")
    if isinstance(regular, dict):
        candidates.extend(
            regular.get(key) for key in ("regularMarketLastPrice", "lastPrice")
        )
    extended = node.get("extended")
    if isinstance(extended, dict):
        candidates.append(extended.get("lastPrice"))

    for value in candidates:
        price = _number(value, default=float("nan"))
        if math.isfinite(price) and price > 0:
            return price
    raise SchwabClientError(f"Quote payload for {symbol} did not include a last price.")


def _underlying_price(payload: dict[str, Any]) -> float | None:
    direct = _number(payload.get("underlyingPrice"), default=float("nan"))
    if math.isfinite(direct) and direct > 0:
        return direct
    underlying = payload.get("underlying") or {}
    if isinstance(underlying, dict):
        for key in ("mark", "last", "close"):
            price = _number(underlying.get(key), default=float("nan"))
            if math.isfinite(price) and price > 0:
                return price
    return None


def _bars_from_candles(candles: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(candles)
    required = {"open", "high", "low", "close", "volume", "datetime"}
    if not required.issubset(frame.columns):
        raise SchwabClientError("Price history payload is missing candle fields.")
    index = pd.to_datetime(frame["datetime"], unit="ms", utc=True).dt.tz_convert(EASTERN)
    frame = frame.set_index(index).sort_index()
    frame = frame.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    return frame[["Open", "High", "Low", "Close", "Volume"]].astype(float)


def _filter_rth_today(frame: pd.DataFrame) -> pd.DataFrame:
    today = _session_date()
    index = pd.DatetimeIndex(frame.index)
    minutes = np.asarray(index.hour) * 60 + np.asarray(index.minute)
    dates = np.array([stamp.date() == today for stamp in index])
    filtered = frame.loc[dates & (minutes >= 9 * 60 + 30) & (minutes < 16 * 60)]
    if filtered.empty:
        raise NoSessionBarsError(
            f"No regular-hours 1-minute bars for {today.isoformat()} yet."
        )
    return filtered


def _empty_side(prefix: str) -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "strikePrice",
            f"{prefix}_openInterest",
            f"{prefix}_gamma",
            f"{prefix}_delta",
            f"{prefix}_volatility",
            f"{prefix}_volume",
        ]
    )


def _side_table(
    exp_map: dict[str, Any] | None,
    prefix: str,
    session_day: date,
) -> pd.DataFrame:
    records: list[dict[str, float]] = []
    for expiration_key, strike_map in (exp_map or {}).items():
        try:
            expiration, key_dte = _parse_expiration_key(str(expiration_key))
        except ValueError:
            continue
        if expiration != session_day or key_dte not in (None, 0):
            continue
        if not isinstance(strike_map, dict):
            continue
        for strike_key, raw in strike_map.items():
            contract = _first_contract(raw)
            if contract is None:
                continue
            dte = contract.get("daysToExpiration", key_dte)
            if dte is not None and int(dte) != 0:
                continue
            records.append(
                {
                    "strikePrice": _number(contract.get("strikePrice"), _number(strike_key)),
                    f"{prefix}_openInterest": _number(contract.get("openInterest")),
                    f"{prefix}_gamma": _number(contract.get("gamma")),
                    f"{prefix}_delta": _number(contract.get("delta")),
                    # Schwab reports implied volatility as a percent (18.4 = 18.4%).
                    f"{prefix}_volatility": _number(contract.get("volatility")) / 100.0,
                    f"{prefix}_volume": _number(contract.get("totalVolume")),
                }
            )

    if not records:
        return _empty_side(prefix)

    frame = pd.DataFrame(records)
    interest = f"{prefix}_openInterest"
    frame = frame.sort_values(interest, ascending=False)
    return frame.drop_duplicates("strikePrice", keep="first")


def _parse_0dte_chain(
    payload: dict[str, Any],
    spot: float,
    session_day: date,
) -> pd.DataFrame:
    calls = _side_table(payload.get("callExpDateMap"), "call", session_day)
    puts = _side_table(payload.get("putExpDateMap"), "put", session_day)
    if calls.empty and puts.empty:
        raise EmptyOptionChainError(
            f"No 0DTE contracts expiring {session_day.isoformat()} were returned."
        )

    merged = calls.merge(puts, on="strikePrice", how="outer").fillna(0.0)
    lower = spot * (1.0 - STRIKE_WINDOW)
    upper = spot * (1.0 + STRIKE_WINDOW)
    merged = merged[
        (merged["strikePrice"] >= lower) & (merged["strikePrice"] <= upper)
    ]
    if merged.empty:
        raise EmptyOptionChainError(
            f"No 0DTE strikes inside ±2% of spot {spot:.2f} "
            f"({lower:.2f}–{upper:.2f})."
        )

    merged = merged.sort_values("strikePrice").reset_index(drop=True)
    return merged.loc[:, list(CHAIN_COLUMNS)].astype(float)


@dataclass(frozen=True)
class SimulatedSession:
    """A locally generated regular-hours session."""

    symbol: str
    bars: pd.DataFrame
    chains: list[tuple[pd.Timestamp, float, pd.DataFrame]]


def generate_simulated_session(
    symbol: str = "SPY",
    *,
    seed: int = 11,
    anchor: float = 572.40,
) -> SimulatedSession:
    """
    Build one regular-hours session of 1-minute bars and 3-minute 0DTE chains.

    The path, open-interest spikes, and volume mix are deterministic for ``seed``.
    Chains use Black-Scholes greeks so the analytics engine can be checked
    without a live token.
    """
    from scipy.stats import norm

    symbol = symbol.upper()
    rng = np.random.default_rng(seed)
    session_day = _session_date()
    start = datetime.combine(session_day, RTH_OPEN, tzinfo=EASTERN)
    index = pd.date_range(start, periods=390, freq="min")
    progress = np.linspace(0.0, 1.0, len(index))
    drift = 3.1 * np.sin(np.linspace(0.0, np.pi * 0.92, len(index)))
    walk = np.cumsum(rng.normal(0.0, 0.11, len(index)))
    close = anchor + drift + walk
    open_ = np.empty(len(index))
    open_[0] = anchor
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) + np.abs(rng.normal(0.0, 0.05, len(index)))
    low = np.minimum(open_, close) - np.abs(rng.normal(0.0, 0.05, len(index)))
    volume_shape = (
        1.0
        + 1.5 * np.exp(-((progress - 0.02) / 0.10) ** 2)
        + 1.3 * np.exp(-((progress - 1.0) / 0.08) ** 2)
    )
    volume = rng.integers(180_000, 420_000, len(index)) * volume_shape

    bars = pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=index,
    )
    bars.index.name = "datetime"

    chains: list[tuple[pd.Timestamp, float, pd.DataFrame]] = []
    for position in range(0, len(index), 3):
        moment = pd.Timestamp(index[position])
        spot = float(close[position])
        chains.append(
            (
                moment,
                spot,
                _synthetic_chain(
                    spot=spot,
                    progress=float(progress[position]),
                    moment=moment,
                    norm=norm,
                ),
            )
        )

    return SimulatedSession(symbol=symbol, bars=bars, chains=chains)


def _synthetic_chain(
    *,
    spot: float,
    progress: float,
    moment: pd.Timestamp,
    norm: Any,
) -> pd.DataFrame:
    low = math.floor(spot * (1.0 - STRIKE_WINDOW))
    high = math.ceil(spot * (1.0 + STRIKE_WINDOW))
    strikes = np.arange(low, high + 1.0, 1.0)

    # Sticky strikes inside the session's ±2% window so the walls step
    # a few times instead of shadowing every tick of spot.
    if progress < 0.33:
        call_anchor, put_anchor = 578.0, 567.0
        call_spike, put_spike = 90_000.0, 32_000.0
    elif progress < 0.66:
        call_anchor, put_anchor = 581.0, 566.0
        call_spike, put_spike = 70_000.0, 34_000.0
    else:
        call_anchor, put_anchor = 576.0, 568.0
        call_spike, put_spike = 140_000.0, 60_000.0

    call_anchor = float(np.clip(call_anchor, strikes[0], strikes[-1]))
    put_anchor = float(np.clip(put_anchor, strikes[0], strikes[-1]))
    atm_iv = 0.122 + 0.050 * progress + 0.008 * math.sin(progress * 10.0)
    moneyness = (strikes - spot) / spot
    call_iv = np.clip(
        atm_iv * (1.0 + 5.0 * np.maximum(moneyness, 0.0) ** 2 - 1.2 * moneyness),
        0.05,
        1.5,
    )
    put_iv = np.clip(
        atm_iv * (1.0 + 8.0 * np.minimum(moneyness, 0.0) ** 2 - 1.8 * moneyness),
        0.05,
        1.5,
    )
    mid_iv = (call_iv + put_iv) / 2.0

    minutes_left = max(16 * 60 - (moment.hour * 60 + moment.minute), 20)
    t_years = minutes_left / (365.0 * 24.0 * 60.0)
    vol_sqrt = np.maximum(mid_iv * math.sqrt(t_years), 1e-6)
    d1 = (np.log(spot / strikes) + 0.5 * mid_iv**2 * t_years) / vol_sqrt
    call_delta = norm.cdf(d1)
    put_delta = call_delta - 1.0
    # Wide gamma keeps dollar-gamma peaks on the open-interest spikes.
    # A pure 0DTE Black-Scholes gamma is so peaked at the money that the
    # wall collapses onto spot and the step lines disappear.
    gamma = np.maximum(0.05 * np.exp(-0.5 * ((strikes - spot) / 10.0) ** 2), 0.012)

    call_oi = 6_000.0 * np.exp(-0.5 * ((strikes - (spot + 2.0)) / 6.0) ** 2)
    put_oi = 6_000.0 * np.exp(-0.5 * ((strikes - (spot - 2.0)) / 6.0) ** 2)
    call_oi = call_oi + call_spike * np.exp(-0.5 * ((strikes - call_anchor) / 0.55) ** 2)
    put_oi = put_oi + put_spike * np.exp(-0.5 * ((strikes - put_anchor) / 0.55) ** 2)
    if progress >= 0.66:
        # Put interest above the call wall pulls total gamma negative after
        # the cumulative sum has already crossed zero.
        put_oi = put_oi + 45_000.0 * np.exp(
            -0.5 * ((strikes - min(call_anchor + 6.0, strikes[-1])) / 1.1) ** 2
        )
    near_atm = np.exp(-0.5 * ((strikes - spot) / 2.5) ** 2)
    call_volume = call_oi * (0.30 + 1.7 * (1.0 - progress)) * (0.40 + near_atm)
    put_volume = put_oi * (0.30 + 2.1 * progress) * (0.40 + near_atm)

    return pd.DataFrame(
        {
            "strikePrice": strikes,
            "call_openInterest": call_oi,
            "call_gamma": gamma,
            "call_delta": call_delta,
            "call_volatility": call_iv,
            "call_volume": call_volume,
            "put_openInterest": put_oi,
            "put_gamma": gamma,
            "put_delta": put_delta,
            "put_volatility": put_iv,
            "put_volume": put_volume,
        }
    )
