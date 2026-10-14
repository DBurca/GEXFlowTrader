"""
Schwab session client for the SPY forward test.

Credentials come from the environment, matching LiveBacktestGEX:
SCHWAB_APP_KEY, SCHWAB_APP_SECRET, SCHWAB_REFRESH_TOKEN.
The access token is refreshed when it is older than 28 minutes.
"""

from __future__ import annotations

import base64
import math
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
import requests
from requests import RequestException

from event_calendar import EASTERN

MARKET_BASE = "https://api.schwabapi.com/marketdata/v1"
TOKEN_URL = "https://api.schwabapi.com/v1/oauth/token"
TOKEN_AGE_SECONDS = 28 * 60


def _refresh_token_path():
    from pathlib import Path

    raw = os.environ.get("SCHWAB_REFRESH_TOKEN_PATH", "")
    return Path(raw) if raw else None


def _read_refresh_token(path) -> str:
    if path is None or not path.is_file():
        return ""
    return path.read_text().strip()


def _write_refresh_token(path, token: str) -> None:
    if path is None or not token:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(token + "\n")
    path.chmod(0o600)


def _auth_error(response: requests.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return "unreadable response"
    if not isinstance(body, dict):
        return "unexpected response"
    detail = body.get("error_description") or body.get("error") or "request rejected"
    return " ".join(str(detail).split())[:180]


class MarketDataError(RuntimeError):
    """Raised when a required Schwab read fails or comes back empty."""


class AuthError(MarketDataError):
    """Raised when the refresh token cannot mint an access token."""


@dataclass
class MacroBook:
    gex_fields: pd.DataFrame
    trade_chain: pd.DataFrame
    expiration: date | None
    dte: int | None


class SchwabSession:
    def __init__(self, app_key: str, app_secret: str, refresh_token: str) -> None:
        self.app_key = app_key
        self.app_secret = app_secret
        self.refresh_token = refresh_token
        self.access_token: str | None = None
        self.token_expiry = 0.0
        self.refresh_token_path = _refresh_token_path()
        stored = _read_refresh_token(self.refresh_token_path)
        if stored:
            self.refresh_token = stored
        self.http = requests.Session()

    def _authenticate(self) -> None:
        if self.access_token and time.time() < self.token_expiry:
            return
        credentials = base64.b64encode(
            f"{self.app_key}:{self.app_secret}".encode()
        ).decode()
        try:
            response = self.http.post(
                TOKEN_URL,
                headers={
                    "Authorization": f"Basic {credentials}",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": self.refresh_token,
                },
                timeout=30,
            )
        except RequestException as exc:
            raise MarketDataError(f"token request failed: {exc.__class__.__name__}") from exc
        if response.status_code != 200:
            raise AuthError(f"Schwab auth failed ({response.status_code}): {_auth_error(response)}")
        payload = response.json()
        token = payload.get("access_token")
        if not token:
            raise AuthError("Schwab auth response did not include an access token")
        self.access_token = str(token)
        rotated = payload.get("refresh_token")
        if rotated:
            self.refresh_token = str(rotated)
            _write_refresh_token(self.refresh_token_path, self.refresh_token)
        self.token_expiry = time.time() + TOKEN_AGE_SECONDS

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        self._authenticate()
        url = f"{MARKET_BASE}{path}"
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                response = self.http.get(
                    url,
                    headers={"Authorization": f"Bearer {self.access_token}", "Accept": "application/json"},
                    params=params,
                    timeout=45,
                )
            except RequestException as exc:
                last_error = exc
                time.sleep(attempt)
                continue
            if response.status_code == 401 and attempt == 1:
                self.token_expiry = 0.0
                self._authenticate()
                continue
            if response.status_code in {429, 500, 502, 503, 504} and attempt < 3:
                time.sleep(attempt)
                continue
            if response.status_code != 200:
                raise MarketDataError(f"{path} returned {response.status_code}")
            payload = response.json()
            if not isinstance(payload, dict):
                raise MarketDataError(f"{path} returned an unexpected payload")
            return payload
        raise MarketDataError(
            f"{path} failed after retries ({last_error.__class__.__name__ if last_error else 'status'})"
        )

    def spot(self, symbol: str = "SPY") -> float:
        payload = self._get("/quotes", {"symbols": symbol})
        node = payload.get(symbol) or payload.get(symbol.upper())
        if not isinstance(node, dict):
            raise MarketDataError(f"no quote for {symbol}")
        quote = node.get("quote") if isinstance(node.get("quote"), dict) else {}
        for key in ("lastPrice", "mark", "last"):
            value = _number(quote.get(key))
            if value > 0:
                return value
        raise MarketDataError(f"quote for {symbol} had no last price")

    def minute_bars(self, symbol: str = "SPY") -> pd.DataFrame:
        # period=1 returns the last completed session. Today's bars require
        # an explicit start and end in epoch milliseconds.
        now = datetime.now(EASTERN)
        session_start = now.replace(hour=9, minute=30, second=0, microsecond=0)
        payload = self._get(
            "/pricehistory",
            {
                "symbol": symbol,
                "periodType": "day",
                "frequencyType": "minute",
                "frequency": 1,
                "needExtendedHoursData": False,
                "startDate": int(session_start.timestamp() * 1000),
                "endDate": int(now.timestamp() * 1000),
            },
        )
        candles = payload.get("candles") or []
        if not candles:
            raise MarketDataError(f"no 1-minute bars for {symbol}")
        frame = pd.DataFrame(candles)
        required = {"open", "high", "low", "close", "volume", "datetime"}
        if not required.issubset(frame.columns):
            raise MarketDataError("price history is missing candle fields")
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

    def option_chain(
        self,
        symbol: str,
        *,
        start: date,
        end: date,
        strike_count: int,
    ) -> dict[str, Any]:
        payload = self._get(
            "/chains",
            {
                "symbol": symbol,
                "contractType": "ALL",
                "strategy": "SINGLE",
                "range": "ALL",
                "includeUnderlyingQuote": "TRUE",
                "strikeCount": strike_count,
                "fromDate": start.isoformat(),
                "toDate": end.isoformat(),
            },
        )
        status = str(payload.get("status") or "SUCCESS").upper()
        if status != "SUCCESS":
            raise MarketDataError(f"option chain status {status}")
        return payload


ES_SYMBOL = "/ES"


def fetch_momentum_bars(
    client: SchwabSession,
    session_day: date,
    close_at: datetime,
    spy_bars: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    """1-minute /ES bars for the cash session. SPY closes stand in when the futures feed fails."""
    try:
        return session_bars(client.minute_bars(ES_SYMBOL), session_day, close_at), ES_SYMBOL
    except MarketDataError:
        return spy_bars, "SPY"


def session_bars(bars: pd.DataFrame, session_day: date, close_at: datetime) -> pd.DataFrame:
    index = pd.DatetimeIndex(bars.index)
    if index.tz is None:
        index = index.tz_localize(EASTERN)
    else:
        index = index.tz_convert(EASTERN)
    frame = bars.copy()
    frame.index = index
    minutes = index.hour * 60 + index.minute
    close_minutes = close_at.hour * 60 + close_at.minute
    dates = np_dates(index, session_day)
    kept = frame.loc[dates & (minutes >= 9 * 60 + 30) & (minutes < close_minutes)]
    if kept.empty:
        raise MarketDataError(f"no regular-hours bars for {session_day.isoformat()}")
    return kept.sort_index()


def np_dates(index: pd.DatetimeIndex, session_day: date):
    import numpy as np

    return np.array([stamp.date() == session_day for stamp in index])


def fetch_near_term_gex(client: SchwabSession, symbol: str, session_day: date, spot: float) -> float:
    """Net GEX of the 0 DTE and 1 DTE books. Gamma × OI × 100 × spot."""
    payload = client.option_chain(
        symbol,
        start=session_day,
        end=session_day + timedelta(days=4),
        strike_count=30,
    )
    total = 0.0
    found = False
    for _exp, dte, contract in _iter_contracts(payload):
        if dte is None or dte < 0 or dte > 1:
            continue
        found = True
        gamma = _number(contract.get("gamma"))
        oi = _number(contract.get("openInterest"))
        sign = 1.0 if str(contract.get("putCall") or "").upper() == "CALL" else -1.0
        total += sign * gamma * oi * 100.0 * float(spot)
    if not found:
        raise MarketDataError(f"no 0–1 DTE contracts for {session_day.isoformat()}")
    return total


def fetch_intraday_chain(client: SchwabSession, symbol: str, session_day: date) -> pd.DataFrame:
    payload = client.option_chain(
        symbol,
        start=session_day,
        end=session_day,
        strike_count=30,
    )
    frame = _chain_frame(payload, lambda dte, exp: exp == session_day and dte == 0)
    if frame.empty:
        raise MarketDataError(f"no 0DTE contracts for {session_day.isoformat()}")
    return frame


def fetch_macro_book(client: SchwabSession, symbol: str, session_day: date) -> MacroBook:
    payload = client.option_chain(
        symbol,
        start=session_day + timedelta(days=28),
        end=session_day + timedelta(days=95),
        strike_count=50,
    )
    contracts = list(_iter_contracts(payload))
    window = [
        item
        for item in contracts
        if item[1] is not None and 30 <= item[1] <= 90
    ]
    if not window:
        raise MarketDataError("no 30–90 DTE contracts in the macro chain")
    gex_fields = _aggregate_gex_fields(window)
    trade_exp = _select_trade_expiration(window)
    trade_rows = [item for item in window if item[0] == trade_exp[0]] if trade_exp else []
    trade_chain = _rows_to_frame(trade_rows) if trade_rows else _empty_chain()
    return MacroBook(
        gex_fields=gex_fields,
        trade_chain=trade_chain,
        expiration=trade_exp[0] if trade_exp else None,
        dte=trade_exp[1] if trade_exp else None,
    )


def _select_trade_expiration(
    window: list[tuple[date, int | None, dict[str, Any]]],
) -> tuple[date, int] | None:
    expirations = {
        (item[0], int(item[1]))
        for item in window
        if item[1] is not None and 30 <= int(item[1]) <= 60
    }
    if not expirations:
        return None
    return min(expirations, key=lambda item: (abs(item[1] - 45), item[1]))


def _aggregate_gex_fields(
    window: list[tuple[date, int | None, dict[str, Any]]],
) -> pd.DataFrame:
    calls: dict[float, dict[str, float]] = {}
    puts: dict[float, dict[str, float]] = {}
    for _exp, _dte, contract in window:
        strike = _number(contract.get("strikePrice"))
        if strike <= 0:
            continue
        right = str(contract.get("putCall") or "").upper()
        bucket = calls if right == "CALL" else puts
        slot = bucket.setdefault(strike, {"oi": 0.0, "gamma_oi": 0.0})
        oi = _number(contract.get("openInterest"))
        slot["oi"] += oi
        slot["gamma_oi"] += _number(contract.get("gamma")) * oi
    strikes = sorted(set(calls) | set(puts))
    rows = []
    for strike in strikes:
        call = calls.get(strike, {"oi": 0.0, "gamma_oi": 0.0})
        put = puts.get(strike, {"oi": 0.0, "gamma_oi": 0.0})
        rows.append(
            {
                "strikePrice": strike,
                "call_openInterest": call["oi"],
                "call_gamma": call["gamma_oi"] / call["oi"] if call["oi"] else 0.0,
                "put_openInterest": put["oi"],
                "put_gamma": put["gamma_oi"] / put["oi"] if put["oi"] else 0.0,
            }
        )
    if not rows:
        raise MarketDataError("macro chain aggregated to zero strikes")
    return pd.DataFrame(rows)


def _chain_frame(payload: dict[str, Any], keep) -> pd.DataFrame:
    rows = []
    for exp, dte, contract in _iter_contracts(payload):
        if dte is None or not keep(dte, exp):
            continue
        rows.append((exp, dte, contract))
    return _rows_to_frame(rows)


def _rows_to_frame(rows: list[tuple[date, int | None, dict[str, Any]]]) -> pd.DataFrame:
    per_strike: dict[float, dict[str, float]] = {}
    for _exp, _dte, contract in rows:
        strike = _number(contract.get("strikePrice"))
        if strike <= 0:
            continue
        right = "call" if str(contract.get("putCall") or "").upper() == "CALL" else "put"
        slot = per_strike.setdefault(strike, {})
        slot[f"{right}_openInterest"] = _number(contract.get("openInterest"))
        slot[f"{right}_gamma"] = _number(contract.get("gamma"))
        slot[f"{right}_delta"] = _number(contract.get("delta"))
        slot[f"{right}_volatility"] = _number(contract.get("volatility")) / 100.0
        slot[f"{right}_volume"] = _number(contract.get("totalVolume"))
        slot[f"{right}_bid"] = _number(contract.get("bid"))
        slot[f"{right}_ask"] = _number(contract.get("ask"))
    if not per_strike:
        return _empty_chain()
    frame = pd.DataFrame(
        [{"strikePrice": strike, **fields} for strike, fields in per_strike.items()]
    )
    for column in _empty_chain().columns:
        if column not in frame.columns:
            frame[column] = 0.0
    return frame.sort_values("strikePrice").reset_index(drop=True)


def _empty_chain() -> pd.DataFrame:
    columns = ["strikePrice"]
    for right in ("call", "put"):
        for field in ("openInterest", "gamma", "delta", "volatility", "volume", "bid", "ask"):
            columns.append(f"{right}_{field}")
    return pd.DataFrame(columns=columns)


def _iter_contracts(payload: dict[str, Any]):
    for map_name in ("callExpDateMap", "putExpDateMap"):
        exp_map = payload.get(map_name) or {}
        if not isinstance(exp_map, dict):
            continue
        for exp_key, strikes in exp_map.items():
            try:
                exp, key_dte = _parse_expiration(str(exp_key))
            except ValueError:
                continue
            if not isinstance(strikes, dict):
                continue
            for _strike_key, raw in strikes.items():
                contract = raw[0] if isinstance(raw, list) and raw else raw
                if not isinstance(contract, dict):
                    continue
                if not contract.get("putCall"):
                    contract = dict(contract)
                    contract["putCall"] = "CALL" if map_name.startswith("call") else "PUT"
                dte = contract.get("daysToExpiration", key_dte)
                dte_value = int(dte) if dte is not None and str(dte).lstrip("-").isdigit() else key_dte
                yield exp, dte_value, contract


def _parse_expiration(key: str) -> tuple[date, int | None]:
    date_part, _, dte_part = key.partition(":")
    expiration = datetime.strptime(date_part, "%Y-%m-%d").date()
    dte = int(dte_part) if dte_part.isdigit() else None
    return expiration, dte


def _number(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(parsed):
        return 0.0
    return parsed
