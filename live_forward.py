"""
SPY paper forward test.

Three strategy accounts plus a cumulative balance. No broker orders.
Snapshots and trades commit together; a failed poll stores nothing and
does not change a paper position. Errors back off in-process and notify
Discord. Weekends and NYSE holidays wait for the next open.
"""

from __future__ import annotations

import json
import logging
import math
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Any

import duckdb
import pandas as pd
import requests

from event_calendar import (
    in_regular_session,
    macro_blackout,
    next_session_open,
    now_eastern,
    session_close_time,
)
from gex_engine import (
    LevelState,
    build_intraday_snapshot,
    calculate_strike_gex,
    calculate_total_net_gex,
    classify_regime,
    classified_flow_update,
    iv_max_fear_levels,
    iv_peak_center,
    momentum_features,
    overnight_pml,
    stabilize_level,
    stabilize_optional_level,
    volume_gex,
)
from live_market import (
    AuthError,
    MacroBook,
    SchwabSession,
    fetch_intraday_chain,
    fetch_macro_book,
    fetch_near_term_gex,
    session_bars,
)
from live_signals import (
    EntrySignal,
    Leg,
    absorption_signal,
    cascade_signal,
    cdf_flattened,
    charm_pin_active,
    next_cdf_flat_streak,
    contract_count,
    credit_legs,
    debit_legs,
    describe_legs,
    infer_hmm_state,
    macro_swing_signal,
    max_loss_per_contract,
    minutes_before_close,
    ornstein_uhlenbeck,
    pin_strike_from_chain,
    reverted_to_mean,
    touched_iv_peak,
    round_trip_fees,
    slippage_blocks,
    structure_nets,
    trade_pnl,
    completed_fifteen_minute_closes,
    completed_five_minute,
    executed_fill,
    exit_slippage_factor,
    fill_price_blocks,
    intrabar_stop,
    five_minute_bars,
    macro_entry_window,
    value_at_or_before,
)
from live_view import publish_live_view
from signal_engine import calculate_gamma_flip
from ta_engine import calculate_vwap_bands, expected_move_bands, minutes_remaining

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

SYMBOL = "SPY"
STRATEGIES = ("absorption", "cascade", "macro")
POLL_SECONDS = 60
MACRO_REFRESH_SECONDS = 15 * 60
HMM_REFRESH_SECONDS = 5 * 60
QUOTE_SKEW_WARN_SECONDS = 3.0
FEE_PER_CONTRACT = 0.65
RISK_FRACTION = 0.02
OPEN_COLOR = 31487
TAKE_PROFIT_COLOR = 5763719
STOP_COLOR = 15548997
EXIT_COLOR = 3447003
FLATTEN_COLOR = 16753920


@dataclass
class MarketSnapshot:
    now: datetime
    spot: float
    intrad_gex: float
    call_wall: float | None
    put_wall: float | None
    call_wall_2: float | None
    put_wall_2: float | None
    call_wall_1_gamma: float | None
    call_wall_2_gamma: float | None
    put_wall_1_gamma: float | None
    put_wall_2_gamma: float | None
    gex_0_1: float | None
    total_net_gex: float | None
    gex_volume: float | None
    charm_hedge: float | None
    bar_open: float
    bar_high: float
    bar_low: float
    bar_close: float
    zero_gamma: float | None
    cdf: float | None
    atm_iv: float | None
    atm_call_iv: float | None
    atm_put_iv: float | None
    vwap: float
    upper_1: float
    lower_1: float
    upper_2: float
    lower_2: float
    macro_gex: float
    gamma_flip: float | None
    macro_regime: str
    hmm: int | None
    ou_theta: float | None
    ou_mu: float | None
    ou_sigma: float | None
    pin_strike: float | None
    blackout: str | None
    charm: bool
    entry_block: str | None
    chain: pd.DataFrame
    macro_chain: pd.DataFrame
    macro_expiration: str | None
    macro_dte: int | None
    bars_5: pd.DataFrame
    minute_bars: pd.DataFrame
    momentum_price: float | None
    momentum_vwap: float | None
    momentum_slope: float | None
    iv_peak_center: float | None
    iv_max_fear_upper: float | None
    iv_max_fear_lower: float | None
    static_open_call_wall_1: float | None
    static_open_put_wall_1: float | None
    static_open_pml: float | None


class ForwardTester:
    def __init__(self) -> None:
        missing = [
            name
            for name in ("SCHWAB_APP_KEY", "SCHWAB_APP_SECRET", "SCHWAB_REFRESH_TOKEN")
            if not os.environ.get(name)
        ]
        if missing:
            raise SystemExit(
                "Missing environment variables: " + ", ".join(missing)
            )
        self.initial = float(os.environ.get("INITIAL_BALANCE", "100000"))
        self.max_ba_ratio = float(os.environ.get("SPREAD_BA_MAX_RATIO", "0.05"))
        self.slippage_factor = float(os.environ.get("SLIPPAGE_SPREAD_FACTOR", "0.35"))
        self.webhook = os.environ.get("DISCORD_WEBHOOK_URL", "")
        self.db = duckdb.connect(os.environ.get("DUCKDB_PATH", "forward_test.db"))
        self.client = SchwabSession(
            os.environ["SCHWAB_APP_KEY"],
            os.environ["SCHWAB_APP_SECRET"],
            os.environ["SCHWAB_REFRESH_TOKEN"],
        )
        self.balances: dict[str, float] = {}
        self.positions: dict[str, dict[str, Any]] = {}
        self.cooldowns: dict[str, datetime] = {}
        self.cdf_history: list[tuple[datetime, float]] = []
        self.iv_history: list[tuple[datetime, float]] = []
        self.macro_book: MacroBook | None = None
        self.macro_fetched_at = 0.0
        self.hmm: int | None = None
        self.hmm_fitted_at = 0.0
        self.prior_hmm: int | None = None
        self.closing_hmm: int | None = None
        self.previous_close: float | None = None
        self.previous_close_day: str | None = None
        self.closing_spot: float | None = None
        self.closing_day: str | None = None
        self.macro_checked_day: str | None = None
        self.last_signal_bar: str | None = None
        self.backoff = 15
        self._level_day = None
        self._call_wall_state = LevelState()
        self._put_wall_state = LevelState()
        self._call_wall_2_state = LevelState()
        self._put_wall_2_state = LevelState()
        self._zero_gamma_state = LevelState()
        self._cdf_flat_streak = 0
        self._es_fallback_logged = False
        self.last_snap: MarketSnapshot | None = None
        self._flow_day: str | None = None
        self._flow_state: dict[str, float] | None = None
        self.classified_cdf = 0.0
        self.static_open_day: str | None = None
        self.static_open_call_wall: float | None = None
        self.static_open_put_wall: float | None = None
        self.static_open_pml: float | None = None
        self._setup_database()
        self._load_state()

    def _setup_database(self) -> None:
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS snapshots (
                ts VARCHAR,
                spot DOUBLE,
                intrad_net_gex DOUBLE,
                call_wall DOUBLE,
                put_wall DOUBLE,
                zero_gamma DOUBLE,
                cdf DOUBLE,
                atm_iv DOUBLE,
                vwap DOUBLE,
                upper_1 DOUBLE,
                lower_1 DOUBLE,
                upper_2 DOUBLE,
                lower_2 DOUBLE,
                macro_net_gex DOUBLE,
                macro_gamma_flip DOUBLE,
                macro_regime VARCHAR,
                hmm_state INTEGER,
                ou_theta DOUBLE,
                ou_mu DOUBLE,
                ou_sigma DOUBLE,
                pin_strike DOUBLE,
                blackout VARCHAR,
                charm_guard BOOLEAN,
                entry_block_reason VARCHAR
            )
            """
        )
        for name, kind in (
            ("call_wall_1_strike", "DOUBLE"),
            ("call_wall_1_gamma", "DOUBLE"),
            ("call_wall_2_strike", "DOUBLE"),
            ("call_wall_2_gamma", "DOUBLE"),
            ("put_wall_1_strike", "DOUBLE"),
            ("put_wall_1_gamma", "DOUBLE"),
            ("put_wall_2_strike", "DOUBLE"),
            ("put_wall_2_gamma", "DOUBLE"),
            ("gex_0_1", "DOUBLE"),
            ("total_net_gex", "DOUBLE"),
            ("gex_volume", "DOUBLE"),
            ("bar_open", "DOUBLE"),
            ("bar_high", "DOUBLE"),
            ("bar_low", "DOUBLE"),
            ("bar_close", "DOUBLE"),
            ("charm_hedge", "DOUBLE"),
            ("atm_call_iv", "DOUBLE"),
            ("atm_put_iv", "DOUBLE"),
        ):
            self.db.execute(f"ALTER TABLE snapshots ADD COLUMN IF NOT EXISTS {name} {kind}")
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                ts VARCHAR,
                strategy VARCHAR,
                event VARCHAR,
                structure VARCHAR,
                contracts INTEGER,
                spot DOUBLE,
                pnl DOUBLE,
                strategy_balance DOUBLE,
                cumulative_balance DOUBLE,
                detail VARCHAR,
                metrics VARCHAR
            )
            """
        )
        for name in ("mid_price", "executed_fill_price"):
            self.db.execute(f"ALTER TABLE trades ADD COLUMN IF NOT EXISTS {name} DOUBLE")
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS price_bars (
                ts VARCHAR PRIMARY KEY,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE
            )
            """
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS balances (strategy VARCHAR PRIMARY KEY, balance DOUBLE)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS open_positions (strategy VARCHAR PRIMARY KEY, payload VARCHAR)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS session_state (key VARCHAR PRIMARY KEY, value VARCHAR)"
        )

    def _load_state(self) -> None:
        for name in (*STRATEGIES, "cumulative"):
            row = self.db.execute(
                "SELECT balance FROM balances WHERE strategy = ?", [name]
            ).fetchone()
            if row is None:
                starting = self.initial * (3 if name == "cumulative" else 1)
                self.db.execute(
                    "INSERT INTO balances VALUES (?, ?)", [name, starting]
                )
                self.balances[name] = starting
            else:
                self.balances[name] = float(row[0])
        self.balances["cumulative"] = sum(self.balances[name] for name in STRATEGIES)

        self.positions = {}
        for strategy, payload in self.db.execute(
            "SELECT strategy, payload FROM open_positions"
        ).fetchall():
            self.positions[str(strategy)] = _position_from_json(str(payload))

        saved = {
            str(key): str(value)
            for key, value in self.db.execute(
                "SELECT key, value FROM session_state"
            ).fetchall()
        }
        self.prior_hmm = _optional_int(saved.get("prior_hmm"))
        self.closing_hmm = _optional_int(saved.get("closing_hmm"))
        self.previous_close = _optional_float(saved.get("previous_close"))
        self.previous_close_day = saved.get("previous_close_day") or None
        self.closing_spot = _optional_float(saved.get("closing_spot"))
        self.closing_day = saved.get("closing_day") or None
        self.macro_checked_day = saved.get("macro_checked_day") or None
        self._promote_prior_session(now_eastern().date())
        for name in STRATEGIES:
            stamped = saved.get(f"cooldown_{name}")
            if stamped:
                self.cooldowns[name] = datetime.fromisoformat(stamped)
        self.static_open_day = saved.get("static_open_day") or None
        self.static_open_call_wall = _optional_float(saved.get("static_open_call_wall_1"))
        self.static_open_put_wall = _optional_float(saved.get("static_open_put_wall_1"))
        self.static_open_pml = _optional_float(saved.get("static_open_pml"))
        self._flow_day = saved.get("flow_day") or None
        self.classified_cdf = _optional_float(saved.get("flow_total")) or 0.0
        raw_flow = saved.get("flow_state") or ""
        try:
            loaded = json.loads(raw_flow) if raw_flow else None
        except json.JSONDecodeError:
            loaded = None
        self._flow_state = loaded if isinstance(loaded, dict) else None

        start = now_eastern().replace(hour=9, minute=30, second=0, microsecond=0)
        try:
            rows = self.db.execute(
                "SELECT ts, cdf, atm_iv FROM snapshots WHERE ts >= ? ORDER BY ts",
                [start.isoformat()],
            ).fetchall()
        except duckdb.Error:
            rows = []
        for stamped, cdf, iv in rows:
            moment = datetime.fromisoformat(str(stamped))
            if cdf is not None and math.isfinite(float(cdf)):
                self.cdf_history.append((moment, float(cdf)))
            if iv is not None and math.isfinite(float(iv)):
                self.iv_history.append((moment, float(iv)))

    def run(self) -> None:
        self._publish_view()
        self._discord_text(
            "SPY forward test started. Paper only. "
            + ", ".join(f"{name} ${self.balances[name]:,.2f}" for name in (*STRATEGIES, "cumulative"))
        )
        while True:
            try:
                if not in_regular_session():
                    self._sleep_until_open()
                    continue
                started = time.monotonic()
                self._iteration()
                self.backoff = 15
                remaining = POLL_SECONDS - (time.monotonic() - started)
                if remaining > 0:
                    time.sleep(remaining)
            except AuthError as exc:
                self._pause_on_error(exc, everyone=True)
            except Exception as exc:
                logging.exception("Forward iteration failed")
                self._pause_on_error(exc, everyone=False)

    def _iteration(self) -> None:
        snap = self._collect()
        checkpoint = self._checkpoint()
        try:
            events = self._decide(snap)
            self._persist(snap, events)
        except Exception:
            self._restore(checkpoint)
            raise
        self.last_snap = snap
        self._remember_flow(snap)
        self._publish_view()
        logging.info(
            "STATE | spot=%.2f | iGEX=%s | mGEX=%s | HMM=%s | OU=%s | blackout=%s | charm=%s | charm$=%s",
            snap.spot,
            _compact(snap.intrad_gex),
            _compact(snap.macro_gex),
            snap.hmm if snap.hmm is not None else "—",
            f"{snap.ou_theta:.4f}" if snap.ou_theta is not None else "—",
            snap.blackout or "—",
            "on" if snap.charm else "off",
            _compact(snap.charm_hedge),
        )
        for event in events:
            self._discord_event(snap, event)

    def _stable_levels(
        self,
        now: datetime,
        call_wall: float | None,
        put_wall: float | None,
        zero_gamma: float | None,
        call_wall_2: float | None,
        put_wall_2: float | None,
    ) -> tuple[float | None, float | None, float | None, float | None, float | None]:
        """Drop a multi-strike jump that does not survive three polls."""
        if self._level_day != now.date():
            self._level_day = now.date()
            self._call_wall_state = LevelState()
            self._put_wall_state = LevelState()
            self._call_wall_2_state = LevelState()
            self._put_wall_2_state = LevelState()
            self._zero_gamma_state = LevelState()
        self._call_wall_state = self._hold_level("call wall", call_wall, self._call_wall_state)
        self._put_wall_state = self._hold_level("put wall", put_wall, self._put_wall_state)
        self._call_wall_2_state = self._hold_optional("secondary call wall", call_wall_2, self._call_wall_2_state)
        self._put_wall_2_state = self._hold_optional("secondary put wall", put_wall_2, self._put_wall_2_state)
        self._zero_gamma_state = self._hold_level("zero gamma", zero_gamma, self._zero_gamma_state)
        return (
            self._call_wall_state.accepted,
            self._put_wall_state.accepted,
            self._zero_gamma_state.accepted,
            self._call_wall_2_state.accepted,
            self._put_wall_2_state.accepted,
        )

    def _hold_level(self, name: str, candidate: float | None, state: LevelState) -> LevelState:
        updated = stabilize_level(candidate, state)
        if (
            candidate is not None
            and updated.accepted is not None
            and abs(float(candidate) - updated.accepted) > 1.01
        ):
            logging.info(
                "Unstable %s %.2f held at %.2f",
                name,
                float(candidate),
                updated.accepted,
            )
        return updated

    def _hold_optional(self, name: str, candidate: float | None, state: LevelState) -> LevelState:
        updated = stabilize_optional_level(candidate, state)
        if candidate is not None and updated.accepted != candidate and (
            updated.accepted is None or abs(float(candidate) - updated.accepted) > 1.01
        ):
            logging.info(
                "Unstable %s %.2f held at %s",
                name,
                float(candidate),
                "none" if updated.accepted is None else f"{updated.accepted:.2f}",
            )
        return updated

    def _lock_open_levels(
        self,
        now: datetime,
        call_wall: float | None,
        put_wall: float | None,
        chain: pd.DataFrame,
    ) -> None:
        """Freeze the 09:30 walls and overnight PML for the rest of the session."""
        day = now.date().isoformat()
        if self.static_open_day == day:
            return
        if now.hour != 9 or now.minute != 30:
            return
        self.static_open_day = day
        self.static_open_call_wall = call_wall
        self.static_open_put_wall = put_wall
        self.static_open_pml = overnight_pml(chain)
        logging.info(
            "Opening levels locked: call %.2f put %.2f PML %s",
            float("nan") if call_wall is None else call_wall,
            float("nan") if put_wall is None else put_wall,
            "none" if self.static_open_pml is None else f"{self.static_open_pml:.2f}",
        )

    def _fetch_market(self, now: datetime):
        """Spot, SPY bars, the 0DTE chain, and /ES bars at the same time."""
        close_at = session_close_time(now.date())

        def timed(fn):
            try:
                return fn(), None, time.time()
            except Exception as exc:
                return None, exc, time.time()

        with ThreadPoolExecutor(max_workers=4) as pool:
            spot_f = pool.submit(timed, lambda: self.client.spot(SYMBOL))
            bars_f = pool.submit(timed, lambda: self.client.minute_bars(SYMBOL))
            chain_f = pool.submit(timed, lambda: fetch_intraday_chain(self.client, SYMBOL, now.date()))
            es_f = pool.submit(timed, lambda: self.client.minute_bars("/ES"))
            spot, spot_error, spot_at = spot_f.result()
            raw_bars, bars_error, bars_at = bars_f.result()
            chain, chain_error, chain_at = chain_f.result()
            es_raw, es_error, es_at = es_f.result()
        for error in (spot_error, bars_error, chain_error):
            if error is not None:
                raise error
        skew = max(spot_at, bars_at, chain_at, es_at) - min(spot_at, bars_at, chain_at, es_at)
        if skew > QUOTE_SKEW_WARN_SECONDS:
            logging.warning("Quote time skew %.1fs across spot, bars, and the chain", skew)
        return spot, raw_bars, chain, es_raw, es_error, close_at

    def _collect(self) -> MarketSnapshot:
        now = now_eastern()
        spot, raw_bars, chain, es_raw, es_error, close_at = self._fetch_market(now)
        bars = session_bars(raw_bars, now.date(), close_at)
        bands = calculate_vwap_bands(bars)
        latest = bands.iloc[-1]
        if self._flow_day != now.date().isoformat():
            self._flow_day = now.date().isoformat()
            self._flow_state = None
            self.classified_cdf = 0.0
        added, self._flow_state = classified_flow_update(chain, self._flow_state)
        self.classified_cdf += added
        intrad = replace(build_intraday_snapshot(chain, spot, now), cdf=self.classified_cdf)
        self._lock_open_levels(now, intrad.call_wall, intrad.put_wall, chain)
        peak = iv_peak_center(chain)
        fear_upper, fear_lower = iv_max_fear_levels(chain, spot)
        gex_0_1 = fetch_near_term_gex(self.client, SYMBOL, now.date(), spot)
        call_wall, put_wall, zero_gamma, call_wall_2, put_wall_2 = self._stable_levels(
            now,
            intrad.call_wall,
            intrad.put_wall,
            intrad.zero_gamma,
            intrad.call_wall_2,
            intrad.put_wall_2,
        )
        minutes_left = minutes_remaining(now, close_at)
        upper_1, lower_1, upper_2, lower_2 = expected_move_bands(
            float(latest["vwap"]),
            spot,
            float(intrad.atm_iv),
            minutes_left,
            call_iv=intrad.atm_call_iv,
            put_iv=intrad.atm_put_iv,
        )
        momentum_bars, momentum_source = bars, "SPY"
        if es_error is None and es_raw is not None:
            try:
                filtered = session_bars(es_raw, now.date(), close_at)
            except (TypeError, ValueError, KeyError):
                filtered = None
            if filtered is not None and not filtered.empty:
                momentum_bars, momentum_source = filtered, "/ES"
        if momentum_source != "/ES" and not self._es_fallback_logged:
            logging.info("ES minute bars unavailable; momentum filter is using SPY closes")
            self._es_fallback_logged = True
        momentum_price, momentum_vwap, momentum_slope = momentum_features(momentum_bars)
        candle = bars.iloc[-1]
        book = self._macro_book(now)
        profile = calculate_strike_gex(book.gex_fields, spot)
        macro_gex = calculate_total_net_gex(profile)
        gamma_flip = calculate_gamma_flip(profile)
        hmm = self._hmm(bars["Close"])
        ou = ornstein_uhlenbeck(bars["Close"].to_numpy(dtype=float))
        pin = pin_strike_from_chain(chain)
        blackout = macro_blackout(now)
        charm = charm_pin_active(now, spot, pin, session_close_time(now.date()))
        block = self._entry_block(now, blackout, charm)
        theta, mu, sigma = ou if ou is not None else (None, None, None)
        return MarketSnapshot(
            now=now,
            spot=spot,
            intrad_gex=float(intrad.net_gex),
            call_wall=call_wall,
            put_wall=put_wall,
            call_wall_2=call_wall_2,
            put_wall_2=put_wall_2,
            call_wall_1_gamma=_finite(intrad.call_wall_1_gamma),
            call_wall_2_gamma=_finite(intrad.call_wall_2_gamma) if call_wall_2 is not None else None,
            put_wall_1_gamma=_finite(intrad.put_wall_1_gamma),
            put_wall_2_gamma=_finite(intrad.put_wall_2_gamma) if put_wall_2 is not None else None,
            gex_0_1=gex_0_1,
            total_net_gex=gex_0_1 + float(macro_gex),
            gex_volume=volume_gex(chain, spot),
            charm_hedge=_finite(intrad.charm_hedge),
            bar_open=float(candle["Open"]),
            bar_high=float(candle["High"]),
            bar_low=float(candle["Low"]),
            bar_close=float(candle["Close"]),
            zero_gamma=zero_gamma,
            cdf=_finite(intrad.cdf),
            atm_iv=_finite(intrad.atm_iv),
            atm_call_iv=_finite(intrad.atm_call_iv),
            atm_put_iv=_finite(intrad.atm_put_iv),
            vwap=float(latest["vwap"]),
            upper_1=upper_1,
            lower_1=lower_1,
            upper_2=upper_2,
            lower_2=lower_2,
            macro_gex=float(macro_gex),
            gamma_flip=gamma_flip,
            macro_regime=classify_regime(macro_gex),
            hmm=hmm,
            ou_theta=theta,
            ou_mu=mu,
            ou_sigma=sigma,
            pin_strike=pin,
            blackout=blackout,
            charm=charm,
            entry_block=block,
            chain=chain,
            macro_chain=book.trade_chain,
            macro_expiration=book.expiration.isoformat() if book.expiration else None,
            macro_dte=book.dte,
            bars_5=completed_five_minute(five_minute_bars(bars), now),
            minute_bars=bars,
            momentum_price=momentum_price,
            momentum_vwap=momentum_vwap,
            momentum_slope=momentum_slope,
            iv_peak_center=peak,
            iv_max_fear_upper=fear_upper,
            iv_max_fear_lower=fear_lower,
            static_open_call_wall_1=self.static_open_call_wall,
            static_open_put_wall_1=self.static_open_put_wall,
            static_open_pml=self.static_open_pml,
        )

    def _macro_book(self, now: datetime) -> MacroBook:
        if self.macro_book is not None and time.monotonic() - self.macro_fetched_at < self._macro_refresh_seconds():
            return self.macro_book
        book = fetch_macro_book(self.client, SYMBOL, now.date())
        self.macro_book = book
        self.macro_fetched_at = time.monotonic()
        return book

    def _macro_refresh_seconds(self) -> float:
        if "macro" in self.positions:
            return POLL_SECONDS
        return MACRO_REFRESH_SECONDS

    def _hmm(self, closes: pd.Series) -> int | None:
        values = closes.to_numpy(dtype=float)
        shock = (
            values.size >= 2
            and math.isfinite(values[-1])
            and math.isfinite(values[-2])
            and values[-2] != 0
            and abs(values[-1] / values[-2] - 1.0) >= 0.003
        )
        if (
            not shock
            and self.hmm is not None
            and time.monotonic() - self.hmm_fitted_at < HMM_REFRESH_SECONDS
        ):
            return self.hmm
        fitted = infer_hmm_state(closes.to_numpy(dtype=float))
        if fitted is None and self.hmm is not None:
            return self.hmm
        self.hmm = fitted
        self.hmm_fitted_at = time.monotonic()
        return fitted

    def _entry_block(self, now: datetime, blackout: str | None, charm: bool) -> str | None:
        if blackout:
            return f"macro blackout ({blackout})"
        close = session_close_time(now.date())
        if minutes_before_close(now, close, 15):
            return "no new 0DTE in the last 15 minutes"
        if charm:
            return "late-session charm pin"
        return None

    def _decide(self, snap: MarketSnapshot) -> list[dict[str, Any]]:
        self._promote_prior_session(snap.now.date())
        events: list[dict[str, Any]] = []
        events.extend(self._manage_exits(snap))
        events.extend(self._manage_entries(snap))
        self._roll_session(snap)
        return events

    def _manage_exits(self, snap: MarketSnapshot) -> list[dict[str, Any]]:
        events = []
        for name in ("absorption", "cascade"):
            if name not in self.positions:
                continue
            closed = self._odte_exit(name, snap)
            if closed is not None:
                events.append(closed)
        if "cascade" not in self.positions:
            self._cdf_flat_streak = 0
        if "macro" in self.positions and self._macro_should_exit(snap):
            closed = self._close_position(
                "macro",
                snap,
                "EXIT",
                "macro book reverted or HMM returned to state 0",
            )
            if closed is not None:
                self.macro_checked_day = snap.now.date().isoformat()
                events.append(closed)
        return events

    def _odte_exit(self, name: str, snap: MarketSnapshot) -> dict[str, Any] | None:
        position = self.positions.get(name)
        if position is None:
            return None
        close_at = session_close_time(snap.now.date())
        if minutes_before_close(snap.now, close_at, 5):
            return self._close_position(name, snap, "MARKET_FLATTEN", "session flatten", allow_intrinsic=True)
        stop_level = snap.zero_gamma if name == "cascade" else float(position.get("stop") or position["anchor"])
        marked = self._exit_mark(position, snap, "STOP")
        risk = intrabar_stop(
            position["direction"],
            snap.spot,
            stop_level,
            float(position["entry_net"]),
            marked,
            int(position["contracts"]),
            position["legs"],
        )
        if risk is not None:
            return self._close_position(name, snap, "STOP", risk, allow_intrinsic=True)
        reason = self._profit_reason(name, position, snap)
        if reason is None or not self._exit_is_profitable(name, snap):
            return None
        return self._close_position(name, snap, "TAKE_PROFIT", reason, allow_intrinsic=False)

    def _profit_reason(self, name: str, position: dict[str, Any], snap: MarketSnapshot) -> str | None:
        marked = self._marked_net(position, snap.chain)
        if name == "absorption":
            if marked is None or position["entry_net"] <= 0:
                return None
            if marked <= 0.5 * position["entry_net"]:
                return "50% of credit captured"
            if reverted_to_mean(
                position["direction"],
                snap.spot,
                float(position["entry_spot"]),
                snap.vwap,
                snap.zero_gamma,
            ):
                return "spot reverted to VWAP or zero gamma"
            if touched_iv_peak(
                position["direction"],
                snap.spot,
                float(position["entry_spot"]),
                snap.iv_peak_center,
            ):
                return "spot touched the IV peak center"
            return None
        if marked is not None and position["direction"] == "bullish" and math.isfinite(snap.upper_2) and snap.spot >= snap.upper_2:
            return "spot reached the upper 2σ band"
        if marked is not None and position["direction"] == "bearish" and math.isfinite(snap.lower_2) and snap.spot <= snap.lower_2:
            return "spot reached the lower 2σ band"
        previous = value_at_or_before(self.cdf_history, snap.now - timedelta(minutes=15))
        flat = snap.cdf is not None and previous is not None and cdf_flattened(snap.cdf, previous)
        self._cdf_flat_streak, confirmed = next_cdf_flat_streak(self._cdf_flat_streak, flat)
        if confirmed and marked is not None:
            return "cumulative delta flow flattened"
        return None

    def _exit_mark(self, position: dict[str, Any], snap: MarketSnapshot, event: str) -> float | None:
        chain = snap.macro_chain if position["strategy"] == "macro" else snap.chain
        priced = self._priced(chain, position["legs"], closing=True, event=event, strategy=position["strategy"])
        if priced is None:
            return None
        return priced[1]

    def _priced(
        self,
        chain: pd.DataFrame,
        legs: list[Leg],
        closing: bool,
        event: str = "TAKE_PROFIT",
        strategy: str = "absorption",
    ) -> tuple[float, float] | None:
        """Signed mid and the fill. Closing adds the spread back; opening subtracts it."""
        nets = structure_nets(chain, legs)
        if nets is None:
            return None
        mid, width = nets
        factor = exit_slippage_factor(strategy, "OPEN" if not closing else event, self.slippage_factor)
        slip = factor * width
        return mid, mid + slip if closing else mid - slip

    def _exit_is_profitable(self, name: str, snap: MarketSnapshot) -> bool:
        """A take-profit close has to be green after round-trip fees and exit slippage."""
        position = self.positions[name]
        priced = self._priced(snap.chain, position["legs"], closing=True)
        if priced is None:
            return False
        fees = round_trip_fees(int(position["contracts"]), len(position["legs"]), FEE_PER_CONTRACT)
        return trade_pnl(float(position["entry_net"]), priced[1], int(position["contracts"]), fees) > 0

    def _macro_should_exit(self, snap: MarketSnapshot) -> bool:
        position = self.positions["macro"]
        if position["entry_day"] >= snap.now.date().isoformat():
            return False
        if snap.hmm == 0:
            return True
        if not self._in_settle(snap.now):
            return False
        back_above_flip = snap.gamma_flip is not None and snap.spot > snap.gamma_flip
        return back_above_flip or snap.macro_gex >= 0

    def _manage_entries(self, snap: MarketSnapshot) -> list[dict[str, Any]]:
        events = []
        bar_pair = self._new_five_minute(snap)
        if bar_pair is not None and snap.entry_block is None:
            previous, bar, bar_key = bar_pair
            events.extend(self._intraday_entries(snap, previous, bar, bar_key))
        if (
            macro_entry_window(snap.now)
            and "macro" not in self.positions
            and self.macro_checked_day != snap.now.date().isoformat()
            and snap.blackout is None
        ):
            opened = self._open_macro(snap)
            if opened is not None:
                events.append(opened)
        return events

    def _new_five_minute(
        self, snap: MarketSnapshot
    ) -> tuple[pd.Series | None, pd.Series, str] | None:
        if snap.bars_5.empty:
            return None
        bar_key = pd.Timestamp(snap.bars_5.index[-1]).isoformat()
        if self.last_signal_bar is None:
            self.last_signal_bar = bar_key
            return None
        if bar_key <= self.last_signal_bar:
            return None
        self.last_signal_bar = bar_key
        previous = snap.bars_5.iloc[-2] if len(snap.bars_5) > 1 else None
        return previous, snap.bars_5.iloc[-1], bar_key

    def _intraday_entries(
        self,
        snap: MarketSnapshot,
        previous: pd.Series | None,
        bar: pd.Series,
        bar_key: str,
    ) -> list[dict[str, Any]]:
        events = []
        if "absorption" not in self.positions and not self._cooling("absorption", snap.now):
            opened = self._open_absorption(snap, bar, bar_key)
            if opened is not None:
                events.append(opened)
        if (
            previous is not None
            and "cascade" not in self.positions
            and not self._cooling("cascade", snap.now)
        ):
            opened = self._open_cascade(snap, previous, bar, bar_key)
            if opened is not None:
                events.append(opened)
        return events

    def _open_absorption(self, snap: MarketSnapshot, bar: pd.Series, bar_key: str) -> dict[str, Any] | None:
        signal = absorption_signal(
            bar,
            intrad_gex=snap.intrad_gex,
            macro_gex=snap.macro_gex,
            hmm_state=snap.hmm,
            call_wall=snap.call_wall,
            put_wall=snap.put_wall,
            upper_2=snap.upper_2,
            lower_2=snap.lower_2,
            upper_1=snap.upper_1,
            lower_1=snap.lower_1,
            call_wall_2=snap.call_wall_2,
            put_wall_2=snap.put_wall_2,
            cdf_now=snap.cdf,
            cdf_prev=value_at_or_before(self.cdf_history, snap.now - timedelta(minutes=15)),
            iv_now=snap.atm_iv,
            iv_prev=value_at_or_before(self.iv_history, snap.now - timedelta(minutes=15)),
            momentum_slope=snap.momentum_slope,
            iv_max_fear_upper=snap.iv_max_fear_upper,
            iv_max_fear_lower=snap.iv_max_fear_lower,
        )
        if signal is None:
            return None
        legs = credit_legs(snap.chain, signal.structure, signal.anchor)
        return self._open_from_signal(signal, legs, snap, snap.chain, bar_key)

    def _open_cascade(
        self,
        snap: MarketSnapshot,
        previous: pd.Series,
        bar: pd.Series,
        bar_key: str,
    ) -> dict[str, Any] | None:
        signal = cascade_signal(
            previous,
            bar,
            intrad_gex=snap.intrad_gex,
            macro_gex=snap.macro_gex,
            gamma_flip=snap.gamma_flip,
            spot=snap.spot,
            call_wall=snap.call_wall,
            put_wall=snap.put_wall,
            zero_gamma=snap.zero_gamma,
            call_wall_2=snap.call_wall_2,
            put_wall_2=snap.put_wall_2,
            vwap=snap.vwap,
            cdf_now=snap.cdf,
            cdf_15=value_at_or_before(self.cdf_history, snap.now - timedelta(minutes=15)),
            cdf_30=value_at_or_before(self.cdf_history, snap.now - timedelta(minutes=30)),
            iv_now=snap.atm_iv,
            iv_prev=value_at_or_before(self.iv_history, snap.now - timedelta(minutes=15)),
            momentum_price=snap.momentum_price,
            momentum_vwap=snap.momentum_vwap,
            momentum_slope=snap.momentum_slope,
        )
        if signal is None:
            return None
        legs = debit_legs(snap.chain, signal.structure, snap.spot, width=2.0)
        return self._open_from_signal(signal, legs, snap, snap.chain, bar_key)

    def _open_macro(self, snap: MarketSnapshot) -> dict[str, Any] | None:
        signal = macro_swing_signal(
            now=snap.now,
            macro_gex=snap.macro_gex,
            gamma_flip=snap.gamma_flip,
            hmm_state=snap.hmm,
            ou_theta=snap.ou_theta,
            fifteen_closes=completed_fifteen_minute_closes(snap.minute_bars, snap.now),
        )
        if signal is None:
            return None
        if snap.macro_chain.empty:
            logging.info("Macro swing skipped: no 30–60 DTE chain to trade")
            return None
        legs = debit_legs(snap.macro_chain, "put_debit", snap.spot, width=5.0)
        opened = self._open_from_signal(signal, legs, snap, snap.macro_chain, "")
        if opened is not None:
            self.macro_checked_day = snap.now.date().isoformat()
        return opened

    def _open_from_signal(
        self,
        signal: EntrySignal,
        legs: list[Leg] | None,
        snap: MarketSnapshot,
        chain: pd.DataFrame,
        bar_key: str,
    ) -> dict[str, Any] | None:
        if not legs:
            logging.info("%s skipped: no listed strikes for the structure", signal.strategy)
            return None
        blocked = slippage_blocks(chain, legs, self.max_ba_ratio)
        if blocked:
            logging.info("%s blocked by slippage guard: %s", signal.strategy, blocked)
            return None
        nets = structure_nets(chain, legs)
        if nets is None:
            return None
        signed, width = nets
        fill = executed_fill(signed, width, exit_slippage_factor(signal.strategy, "OPEN", self.slippage_factor))
        rejected = fill_price_blocks(signed, fill, legs)
        if rejected:
            logging.info("%s blocked by fill: %s", signal.strategy, rejected)
            return None
        loss = max_loss_per_contract(fill, legs)
        contracts = contract_count(self.balances[signal.strategy], loss, RISK_FRACTION)
        if contracts < 1:
            logging.info("%s skipped: max loss $%.2f does not fit the 2%% budget", signal.strategy, loss)
            return None
        self.positions[signal.strategy] = {
            "strategy": signal.strategy,
            "structure": signal.structure,
            "direction": signal.direction,
            "legs": legs,
            "contracts": contracts,
            "entry_net": fill,
            "entry_mid": signed,
            "entry_spot": snap.spot,
            "entry_time": snap.now,
            "entry_day": snap.now.date().isoformat(),
            "entry_bar": bar_key,
            "anchor": signal.anchor,
            "stop": signal.stop if signal.stop is not None else signal.anchor,
            "reason": signal.reason,
            "expiration": snap.macro_expiration if signal.strategy == "macro" else snap.now.date().isoformat(),
        }
        logging.info(
            "OPEN %s %s @ fill %.2f mid %.2f | %s",
            signal.strategy,
            describe_legs(legs, contracts),
            fill,
            signed,
            signal.reason,
        )
        return self._event(
            signal.strategy,
            "OPEN",
            snap,
            pnl=None,
            detail=signal.reason,
            mid_price=signed,
            executed_fill_price=fill,
        )

    def _close_position(
        self,
        name: str,
        snap: MarketSnapshot,
        event: str,
        detail: str,
        allow_intrinsic: bool = False,
    ) -> dict[str, Any] | None:
        position = self.positions[name]
        chain = snap.macro_chain if name == "macro" else snap.chain
        priced = self._priced(chain, position["legs"], closing=True, event=event, strategy=name)
        used_intrinsic = False
        if priced is None and allow_intrinsic:
            intrinsic = _intrinsic_net(position["legs"], snap.spot)
            priced = (intrinsic, intrinsic)
            used_intrinsic = True
        if priced is None:
            logging.info("%s %s deferred: option quotes unavailable", name, event)
            return None
        mid, fill = priced
        fees = round_trip_fees(position["contracts"], len(position["legs"]), FEE_PER_CONTRACT)
        pnl = trade_pnl(float(position["entry_net"]), fill, int(position["contracts"]), fees)
        self.balances[name] += pnl
        self.balances["cumulative"] = sum(self.balances[item] for item in STRATEGIES)
        if event == "STOP":
            self.cooldowns[name] = snap.now + timedelta(minutes=5)
        del self.positions[name]
        if used_intrinsic:
            detail = f"{detail}; marked intrinsic because quotes were unavailable"
        logging.info(
            "CLOSE %s %s | PnL $%.2f | balance $%.2f",
            name,
            event,
            pnl,
            self.balances[name],
        )
        return self._event(
            name, event, snap, pnl=pnl, detail=detail, closed=position,
            mid_price=mid, executed_fill_price=fill,
        )

    def _event(
        self,
        strategy: str,
        event: str,
        snap: MarketSnapshot,
        pnl: float | None,
        detail: str,
        closed: dict[str, Any] | None = None,
        mid_price: float | None = None,
        executed_fill_price: float | None = None,
    ) -> dict[str, Any]:
        position = closed or self.positions[strategy]
        return {
            "ts": snap.now.isoformat(),
            "strategy": strategy,
            "event": event,
            "structure": position["structure"],
            "contracts": int(position["contracts"]),
            "spot": snap.spot,
            "pnl": pnl,
            "strategy_balance": self.balances[strategy],
            "cumulative_balance": self.balances["cumulative"],
            "detail": f"{describe_legs(position['legs'], position['contracts'])} | {detail}",
            "mid_price": mid_price,
            "executed_fill_price": executed_fill_price,
            "direction": position["direction"],
            "entry_net": float(position["entry_net"]),
            "entry_spot": float(position["entry_spot"]),
            "expiration": position.get("expiration") or "",
        }

    def _roll_session(self, snap: MarketSnapshot) -> None:
        """Remember today's close without changing the prior session used for signals."""
        if not self._in_settle(snap.now):
            return
        self.closing_spot = snap.spot
        self.closing_day = snap.now.date().isoformat()
        if snap.hmm is not None:
            self.closing_hmm = snap.hmm

    def _promote_prior_session(self, today) -> None:
        """Yesterday's settlement becomes the prior close once a new session starts."""
        if self.closing_day and self.closing_day < today.isoformat():
            if self.closing_spot is not None:
                self.previous_close = self.closing_spot
                self.previous_close_day = self.closing_day
            if self.closing_hmm is not None:
                self.prior_hmm = self.closing_hmm
            self.closing_spot = None
            self.closing_hmm = None
            self.closing_day = None

    def _latest_bar(self, snap: MarketSnapshot) -> tuple[str, pd.Series] | None:
        if snap.bars_5.empty:
            return None
        return pd.Timestamp(snap.bars_5.index[-1]).isoformat(), snap.bars_5.iloc[-1]

    def _marked_net(self, position: dict[str, Any], chain: pd.DataFrame) -> float | None:
        nets = structure_nets(chain, position["legs"])
        if nets is None:
            return None
        return nets[0]

    def _cooling(self, strategy: str, now: datetime) -> bool:
        until = self.cooldowns.get(strategy)
        return until is not None and now < until

    def _in_settle(self, now: datetime) -> bool:
        return minutes_before_close(now, session_close_time(now.date()), 5)

    def _persist(self, snap: MarketSnapshot, events: list[dict[str, Any]]) -> None:
        metrics = _metrics_json(snap)
        self.db.execute("BEGIN TRANSACTION")
        try:
            columns = [
                "ts", "spot", "intrad_net_gex", "call_wall", "put_wall", "zero_gamma",
                "cdf", "atm_iv", "vwap", "upper_1", "lower_1", "upper_2", "lower_2",
                "macro_net_gex", "macro_gamma_flip", "macro_regime", "hmm_state",
                "ou_theta", "ou_mu", "ou_sigma", "pin_strike", "blackout",
                "charm_guard", "entry_block_reason",
                "call_wall_1_strike", "call_wall_1_gamma", "call_wall_2_strike", "call_wall_2_gamma",
                "put_wall_1_strike", "put_wall_1_gamma", "put_wall_2_strike", "put_wall_2_gamma",
                "gex_0_1", "total_net_gex", "gex_volume", "charm_hedge",
                "bar_open", "bar_high", "bar_low", "bar_close",
                "atm_call_iv", "atm_put_iv",
            ]
            self.db.execute(
                "INSERT INTO snapshots (" + ", ".join(columns) + ") VALUES ("
                + ",".join(["?"] * len(columns)) + ")",
                [
                    snap.now.isoformat(),
                    snap.spot,
                    snap.intrad_gex,
                    _db_float(snap.call_wall),
                    _db_float(snap.put_wall),
                    _db_float(snap.zero_gamma),
                    _db_float(snap.cdf),
                    _db_float(snap.atm_iv),
                    _db_float(snap.vwap),
                    _db_float(snap.upper_1),
                    _db_float(snap.lower_1),
                    _db_float(snap.upper_2),
                    _db_float(snap.lower_2),
                    snap.macro_gex,
                    _db_float(snap.gamma_flip),
                    snap.macro_regime,
                    snap.hmm,
                    _db_float(snap.ou_theta),
                    _db_float(snap.ou_mu),
                    _db_float(snap.ou_sigma),
                    _db_float(snap.pin_strike),
                    snap.blackout,
                    snap.charm,
                    snap.entry_block,
                    _db_float(snap.call_wall),
                    _db_float(snap.call_wall_1_gamma),
                    _db_float(snap.call_wall_2),
                    _db_float(snap.call_wall_2_gamma),
                    _db_float(snap.put_wall),
                    _db_float(snap.put_wall_1_gamma),
                    _db_float(snap.put_wall_2),
                    _db_float(snap.put_wall_2_gamma),
                    _db_float(snap.gex_0_1),
                    _db_float(snap.total_net_gex),
                    _db_float(snap.gex_volume),
                    _db_float(snap.charm_hedge),
                    _db_float(snap.bar_open),
                    _db_float(snap.bar_high),
                    _db_float(snap.bar_low),
                    _db_float(snap.bar_close),
                    _db_float(snap.atm_call_iv),
                    _db_float(snap.atm_put_iv),
                ],
            )
            trade_columns = [
                "ts", "strategy", "event", "structure", "contracts", "spot", "pnl",
                "strategy_balance", "cumulative_balance", "detail", "metrics",
                "mid_price", "executed_fill_price",
            ]
            for event in events:
                self.db.execute(
                    "INSERT INTO trades (" + ", ".join(trade_columns) + ") VALUES ("
                    + ",".join(["?"] * len(trade_columns)) + ")",
                    [
                        event["ts"],
                        event["strategy"],
                        event["event"],
                        event["structure"],
                        event["contracts"],
                        event["spot"],
                        event["pnl"],
                        event["strategy_balance"],
                        event["cumulative_balance"],
                        event["detail"],
                        metrics,
                        _db_float(event.get("mid_price")),
                        _db_float(event.get("executed_fill_price")),
                    ],
                )
            for name in (*STRATEGIES, "cumulative"):
                self.db.execute("DELETE FROM balances WHERE strategy = ?", [name])
                self.db.execute(
                    "INSERT INTO balances VALUES (?, ?)",
                    [name, self.balances[name]],
                )
            self.db.execute("DELETE FROM open_positions")
            for strategy, position in self.positions.items():
                self.db.execute(
                    "INSERT INTO open_positions VALUES (?, ?)",
                    [strategy, _position_to_json(position)],
                )
            state = {
                "prior_hmm": "" if self.prior_hmm is None else str(self.prior_hmm),
                "closing_hmm": "" if self.closing_hmm is None else str(self.closing_hmm),
                "previous_close": "" if self.previous_close is None else str(self.previous_close),
                "previous_close_day": self.previous_close_day or "",
                "closing_spot": "" if self.closing_spot is None else str(self.closing_spot),
                "closing_day": self.closing_day or "",
                "macro_checked_day": self.macro_checked_day or "",
                "static_open_day": self.static_open_day or "",
                "static_open_call_wall_1": "" if self.static_open_call_wall is None else str(self.static_open_call_wall),
                "static_open_put_wall_1": "" if self.static_open_put_wall is None else str(self.static_open_put_wall),
                "static_open_pml": "" if self.static_open_pml is None else str(self.static_open_pml),
                "flow_day": self._flow_day or "",
                "flow_total": str(self.classified_cdf),
                "flow_state": json.dumps(self._flow_state or {}),
            }
            for name in STRATEGIES:
                until = self.cooldowns.get(name)
                state[f"cooldown_{name}"] = until.isoformat() if until else ""
            for key, value in state.items():
                self.db.execute("DELETE FROM session_state WHERE key = ?", [key])
                self.db.execute("INSERT INTO session_state VALUES (?, ?)", [key, value])
            self._replace_price_bars(snap)
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    def _replace_price_bars(self, snap: MarketSnapshot) -> None:
        """Replace this session's minute bars with the latest history from Schwab."""
        day = snap.now.date().isoformat()
        self.db.execute("DELETE FROM price_bars WHERE starts_with(ts, ?)", [day])
        bars = snap.minute_bars
        if bars is None or bars.empty:
            return
        for ts, row in bars.iterrows():
            self.db.execute(
                "INSERT INTO price_bars VALUES (?, ?, ?, ?, ?)",
                [
                    pd.Timestamp(ts).isoformat(),
                    float(row["Open"]),
                    float(row["High"]),
                    float(row["Low"]),
                    float(row["Close"]),
                ],
            )

    def _publish_view(self) -> None:
        try:
            publish_live_view(self.db)
        except Exception:
            logging.exception("Live view publish failed")

    def _remember_flow(self, snap: MarketSnapshot) -> None:
        if snap.cdf is not None:
            self.cdf_history.append((snap.now, snap.cdf))
        if snap.atm_iv is not None:
            self.iv_history.append((snap.now, snap.atm_iv))

    def _checkpoint(self) -> dict[str, Any]:
        return {
            "balances": dict(self.balances),
            "positions": json.loads(json.dumps({
                name: json.loads(_position_to_json(position))
                for name, position in self.positions.items()
            })),
            "cooldowns": dict(self.cooldowns),
            "last_signal_bar": self.last_signal_bar,
            "cdf_flat_streak": self._cdf_flat_streak,
            "prior_hmm": self.prior_hmm,
            "closing_hmm": self.closing_hmm,
            "previous_close": self.previous_close,
            "previous_close_day": self.previous_close_day,
            "closing_spot": self.closing_spot,
            "closing_day": self.closing_day,
            "macro_checked_day": self.macro_checked_day,
        }

    def _restore(self, checkpoint: dict[str, Any]) -> None:
        self.balances = dict(checkpoint["balances"])
        self.positions = {
            name: _position_from_json(json.dumps(payload))
            for name, payload in checkpoint["positions"].items()
        }
        self.cooldowns = dict(checkpoint["cooldowns"])
        self.last_signal_bar = checkpoint["last_signal_bar"]
        self._cdf_flat_streak = checkpoint["cdf_flat_streak"]
        self.prior_hmm = checkpoint["prior_hmm"]
        self.closing_hmm = checkpoint["closing_hmm"]
        self.previous_close = checkpoint["previous_close"]
        self.previous_close_day = checkpoint["previous_close_day"]
        self.closing_spot = checkpoint["closing_spot"]
        self.closing_day = checkpoint["closing_day"]
        self.macro_checked_day = checkpoint["macro_checked_day"]

    def _discord_event(self, snap: MarketSnapshot, event: dict[str, Any]) -> None:
        colors = {
            "OPEN": OPEN_COLOR,
            "TAKE_PROFIT": TAKE_PROFIT_COLOR,
            "STOP": STOP_COLOR,
            "EXIT": EXIT_COLOR,
            "MARKET_FLATTEN": FLATTEN_COLOR,
        }
        fields = [
            {"name": "Strategy balance", "value": f"${event['strategy_balance']:,.2f}", "inline": True},
            {"name": "Cumulative balance", "value": f"${event['cumulative_balance']:,.2f}", "inline": True},
            {"name": "Position", "value": event["detail"][:1000], "inline": False},
            {"name": "Spot", "value": f"{event['spot']:.2f} (entry {event['entry_spot']:.2f})", "inline": True},
            {
                "name": "Structure net",
                "value": f"{event['entry_net']:.2f} | exp {event['expiration'] or '—'}",
                "inline": True,
            },
        ]
        if event["pnl"] is not None:
            fields.append({"name": "PnL", "value": f"${event['pnl']:,.2f}", "inline": True})
        fields.append({"name": "Metrics", "value": _metrics_text(snap)[:1024], "inline": False})
        self._discord_embed(
            title=f"SPY {event['strategy']} {event['event']}",
            color=colors.get(event["event"], OPEN_COLOR),
            fields=fields,
        )

    def _discord_text(self, content: str) -> None:
        self._post_discord({"content": content[:1900]})

    def _discord_embed(self, title: str, color: int, fields: list[dict[str, Any]]) -> None:
        self._post_discord(
            {
                "content": None,
                "embeds": [{"title": title[:256], "color": color, "fields": fields}],
            }
        )

    def _post_discord(self, payload: dict[str, Any]) -> None:
        if not self.webhook:
            return
        try:
            requests.post(self.webhook, json=payload, timeout=5)
        except requests.RequestException as exc:
            logging.error("Discord webhook failed: %s", exc.__class__.__name__)

    def _pause_on_error(self, exc: Exception, everyone: bool) -> None:
        self._flatten_if_unmanaged()
        wait = self.backoff
        cap = 30 if any(name in self.positions for name in ("absorption", "cascade")) else 300
        self.backoff = min(self.backoff * 2, cap)
        reason = " ".join(f"{exc.__class__.__name__}: {exc}".split())[:300]
        message = (
            f"SPY forward test paused: {reason}. Next retry in {wait}s. "
            "No snapshot stored and no orders simulated."
        )
        if everyone:
            message = "@everyone " + message
        logging.error(message)
        self._discord_text(message)
        time.sleep(wait)

    def _flatten_if_unmanaged(self) -> None:
        """Paper-flatten 0DTE positions when a failed poll lands inside the last 10 minutes."""
        snap = self.last_snap
        if snap is None:
            return
        now = now_eastern()
        if not minutes_before_close(now, session_close_time(now.date()), 10):
            return
        if not any(name in self.positions for name in ("absorption", "cascade")):
            return
        snap.now = now
        events = []
        for name in ("absorption", "cascade"):
            if name not in self.positions:
                continue
            closed = self._close_position(
                name, snap, "MARKET_FLATTEN", "poll outage inside the flatten window", allow_intrinsic=True,
            )
            if closed is not None:
                events.append(closed)
                self._discord_event(snap, closed)
        if not events:
            return
        try:
            self._persist(snap, events)
            self._publish_view()
        except Exception:
            logging.exception("Outage flatten could not be stored")

    def _sleep_until_open(self) -> None:
        nxt = next_session_open()
        seconds = max(1.0, (nxt - now_eastern()).total_seconds())
        logging.info("Market closed. Next open %s (%.2f hours).", nxt.isoformat(), seconds / 3600)
        time.sleep(seconds)

    def shutdown(self) -> None:
        logging.info("Shutting down forward test.")
        try:
            self.db.close()
        except duckdb.Error:
            pass


def _finite(value: float) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def _db_float(value: float | None) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def _optional_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _intrinsic_net(legs: list[Leg], spot: float) -> float:
    total = 0.0
    for leg in legs:
        if leg.right == "call":
            amount = max(spot - leg.strike, 0.0)
        else:
            amount = max(leg.strike - spot, 0.0)
        total += amount if leg.side == "short" else -amount
    return total


def _position_to_json(position: dict[str, Any]) -> str:
    payload = {
        "strategy": position["strategy"],
        "structure": position["structure"],
        "direction": position["direction"],
        "legs": [
            {"right": leg.right, "strike": leg.strike, "side": leg.side}
            for leg in position["legs"]
        ],
        "contracts": position["contracts"],
        "entry_net": position["entry_net"],
        "entry_spot": position["entry_spot"],
        "entry_time": position["entry_time"].isoformat(),
        "entry_day": position["entry_day"],
        "entry_bar": position["entry_bar"],
        "anchor": position["anchor"],
        "stop": position.get("stop", position["anchor"]),
        "reason": position["reason"],
        "expiration": position.get("expiration") or "",
    }
    return json.dumps(payload)


def _position_from_json(payload: str) -> dict[str, Any]:
    raw = json.loads(payload)
    raw["legs"] = [Leg(**leg) for leg in raw["legs"]]
    raw["entry_time"] = datetime.fromisoformat(raw["entry_time"])
    raw["contracts"] = int(raw["contracts"])
    raw["stop"] = raw.get("stop", raw["anchor"])
    return raw


def _metrics_json(snap: MarketSnapshot) -> str:
    return json.dumps(
        {
            "spot": snap.spot,
            "intrad_net_gex": snap.intrad_gex,
            "call_wall": snap.call_wall,
            "put_wall": snap.put_wall,
            "call_wall_2": snap.call_wall_2,
            "put_wall_2": snap.put_wall_2,
            "gex_0_1": snap.gex_0_1,
            "total_net_gex": snap.total_net_gex,
            "zero_gamma": snap.zero_gamma,
            "cdf": snap.cdf,
            "atm_iv": snap.atm_iv,
            "vwap": snap.vwap,
            "upper_2": snap.upper_2,
            "lower_2": snap.lower_2,
            "macro_net_gex": snap.macro_gex,
            "macro_gamma_flip": snap.gamma_flip,
            "macro_regime": snap.macro_regime,
            "hmm_state": snap.hmm,
            "ou_theta": snap.ou_theta,
            "ou_mu": snap.ou_mu,
            "ou_sigma": snap.ou_sigma,
            "pin_strike": snap.pin_strike,
            "blackout": snap.blackout,
            "charm_guard": snap.charm,
            "charm_hedge": snap.charm_hedge,
            "macro_expiration": snap.macro_expiration,
            "macro_dte": snap.macro_dte,
        },
        default=str,
    )


def _metrics_text(snap: MarketSnapshot) -> str:
    walls = (
        f"Call {_level(snap.call_wall)} | Put {_level(snap.put_wall)} | "
        f"Zero γ {_level(snap.zero_gamma)}"
    )
    return "\n".join(
        [
            f"Spot {snap.spot:.2f} | VWAP {snap.vwap:.2f} | 2σ {_level(snap.lower_2)} / {_level(snap.upper_2)}",
            f"Intraday GEX {_compact(snap.intrad_gex)} | {walls}",
            f"CDF {_compact(snap.cdf)} | ATM IV {_percent(snap.atm_iv)}",
            (
                f"Macro GEX {_compact(snap.macro_gex)} ({snap.macro_regime}) | "
                f"Flip {_level(snap.gamma_flip)} | HMM {snap.hmm if snap.hmm is not None else '—'} | "
                f"OU θ {_theta(snap.ou_theta)}"
            ),
            (
                f"Pin {_level(snap.pin_strike)} | Blackout {snap.blackout or '—'} | "
                f"Charm pin {'on' if snap.charm else 'off'} | Charm to close {_compact(snap.charm_hedge)}"
            ),
        ]
    )


def _level(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):.2f}"


def _percent(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{float(value) * 100:.1f}%"


def _theta(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):.4f}"


def _compact(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    number = float(value)
    sign = "-" if number < 0 else ""
    magnitude = abs(number)
    for scale, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "k")):
        if magnitude >= scale:
            return f"{sign}{magnitude / scale:.2f}{suffix}"
    return f"{sign}{magnitude:.2f}"


def main() -> None:
    bot = ForwardTester()

    def _handle_signal(_signum, _frame):
        bot.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    logging.info("Starting SPY paper forward test.")
    bot.run()


if __name__ == "__main__":
    main()
