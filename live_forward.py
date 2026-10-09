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
from dataclasses import dataclass
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
from gex_engine import build_intraday_snapshot, calculate_strike_gex, calculate_total_net_gex, classify_regime
from live_market import AuthError, MacroBook, SchwabSession, fetch_intraday_chain, fetch_macro_book, session_bars
from live_signals import (
    EntrySignal,
    Leg,
    absorption_signal,
    breach_confirmed,
    cascade_signal,
    cdf_flattened,
    charm_pin_active,
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
    round_trip_fees,
    slippage_blocks,
    structure_nets,
    trade_pnl,
    completed_five_minute,
    five_minute_bars,
    value_at_or_before,
)
from signal_engine import calculate_gamma_flip
from ta_engine import calculate_vwap_bands

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

SYMBOL = "SPY"
STRATEGIES = ("absorption", "cascade", "macro")
POLL_SECONDS = 60
MACRO_REFRESH_SECONDS = 15 * 60
HMM_REFRESH_SECONDS = 30 * 60
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
    zero_gamma: float | None
    cdf: float | None
    atm_iv: float | None
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
        self._remember_flow(snap)
        logging.info(
            "STATE | spot=%.2f | iGEX=%s | mGEX=%s | HMM=%s | OU=%s | blackout=%s | charm=%s",
            snap.spot,
            _compact(snap.intrad_gex),
            _compact(snap.macro_gex),
            snap.hmm if snap.hmm is not None else "—",
            f"{snap.ou_theta:.4f}" if snap.ou_theta is not None else "—",
            snap.blackout or "—",
            "on" if snap.charm else "off",
        )
        for event in events:
            self._discord_event(snap, event)

    def _collect(self) -> MarketSnapshot:
        now = now_eastern()
        spot = self.client.spot(SYMBOL)
        raw_bars = self.client.minute_bars(SYMBOL)
        bars = session_bars(raw_bars, now.date(), session_close_time(now.date()))
        bands = calculate_vwap_bands(bars)
        latest = bands.iloc[-1]
        chain = fetch_intraday_chain(self.client, SYMBOL, now.date())
        intrad = build_intraday_snapshot(chain, spot, now)
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
            call_wall=intrad.call_wall,
            put_wall=intrad.put_wall,
            zero_gamma=intrad.zero_gamma,
            cdf=_finite(intrad.cdf),
            atm_iv=_finite(intrad.atm_iv),
            vwap=float(latest["vwap"]),
            upper_1=float(latest["upper_1"]),
            lower_1=float(latest["lower_1"]),
            upper_2=float(latest["upper_2"]),
            lower_2=float(latest["lower_2"]),
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
        if self.hmm is not None and time.monotonic() - self.hmm_fitted_at < HMM_REFRESH_SECONDS:
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
        bar = self._latest_bar(snap)
        if bar is not None and bar[0] > position["entry_bar"]:
            if name == "absorption" and self._absorption_stopped(position, snap, bar[1]):
                return self._close_position(
                    name, snap, "STOP", "5-minute close through the tested level with CDF confirmation",
                    allow_intrinsic=True,
                )
            if name == "cascade" and self._cascade_stopped(position, snap, bar[1]):
                return self._close_position(
                    name, snap, "STOP", "5-minute close back through zero gamma",
                    allow_intrinsic=True,
                )
        reason = self._profit_reason(name, position, snap)
        if reason is None:
            return None
        return self._close_position(name, snap, "TAKE_PROFIT", reason, allow_intrinsic=False)

    def _absorption_stopped(self, position: dict[str, Any], snap: MarketSnapshot, bar: pd.Series) -> bool:
        if snap.cdf is None:
            return False
        previous = value_at_or_before(self.cdf_history, snap.now - timedelta(minutes=15))
        if previous is None:
            return False
        return breach_confirmed(
            position["direction"],
            float(bar["Close"]),
            float(position["anchor"]),
            snap.cdf,
            previous,
        )

    def _cascade_stopped(self, position: dict[str, Any], snap: MarketSnapshot, bar: pd.Series) -> bool:
        if snap.zero_gamma is None:
            return False
        close = float(bar["Close"])
        if position["direction"] == "bullish":
            return close <= snap.zero_gamma
        return close >= snap.zero_gamma

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
            return None
        if marked is None:
            return None
        if position["direction"] == "bullish" and math.isfinite(snap.upper_2) and snap.spot >= snap.upper_2:
            return "spot reached the upper 2σ band"
        if position["direction"] == "bearish" and math.isfinite(snap.lower_2) and snap.spot <= snap.lower_2:
            return "spot reached the lower 2σ band"
        previous = value_at_or_before(self.cdf_history, snap.now - timedelta(minutes=15))
        if snap.cdf is not None and previous is not None and cdf_flattened(snap.cdf, previous):
            return "cumulative delta flow flattened"
        return None

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
            self._in_settle(snap.now)
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
            cdf_now=snap.cdf,
            cdf_prev=value_at_or_before(self.cdf_history, snap.now - timedelta(minutes=15)),
            iv_now=snap.atm_iv,
            iv_prev=value_at_or_before(self.iv_history, snap.now - timedelta(minutes=15)),
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
            vwap=snap.vwap,
            cdf_now=snap.cdf,
            cdf_15=value_at_or_before(self.cdf_history, snap.now - timedelta(minutes=15)),
            cdf_30=value_at_or_before(self.cdf_history, snap.now - timedelta(minutes=30)),
            iv_now=snap.atm_iv,
            iv_prev=value_at_or_before(self.iv_history, snap.now - timedelta(minutes=15)),
        )
        if signal is None:
            return None
        legs = debit_legs(snap.chain, signal.structure, snap.spot, width=2.0)
        return self._open_from_signal(signal, legs, snap, snap.chain, bar_key)

    def _open_macro(self, snap: MarketSnapshot) -> dict[str, Any] | None:
        signal = macro_swing_signal(
            spot=snap.spot,
            previous_close=self.previous_close,
            macro_gex=snap.macro_gex,
            gamma_flip=snap.gamma_flip,
            hmm_state=snap.hmm,
            previous_hmm=self.prior_hmm,
            ou_theta=snap.ou_theta,
        )
        if signal is None:
            return None
        self.macro_checked_day = snap.now.date().isoformat()
        if snap.macro_chain.empty:
            logging.info("Macro swing skipped: no 30–60 DTE chain to trade")
            return None
        legs = debit_legs(snap.macro_chain, "put_debit", snap.spot, width=5.0)
        return self._open_from_signal(signal, legs, snap, snap.macro_chain, "")

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
        signed, _width = nets
        loss = max_loss_per_contract(signed, legs)
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
            "entry_net": signed,
            "entry_spot": snap.spot,
            "entry_time": snap.now,
            "entry_day": snap.now.date().isoformat(),
            "entry_bar": bar_key,
            "anchor": signal.anchor,
            "reason": signal.reason,
            "expiration": snap.macro_expiration if signal.strategy == "macro" else snap.now.date().isoformat(),
        }
        logging.info(
            "OPEN %s %s @ net %.2f | %s",
            signal.strategy,
            describe_legs(legs, contracts),
            signed,
            signal.reason,
        )
        return self._event(signal.strategy, "OPEN", snap, pnl=None, detail=signal.reason)

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
        marked = self._marked_net(position, chain)
        used_intrinsic = False
        if marked is None and allow_intrinsic:
            marked = _intrinsic_net(position["legs"], snap.spot)
            used_intrinsic = True
        if marked is None:
            logging.info("%s %s deferred: option quotes unavailable", name, event)
            return None
        fees = round_trip_fees(position["contracts"], len(position["legs"]), FEE_PER_CONTRACT)
        pnl = trade_pnl(float(position["entry_net"]), marked, int(position["contracts"]), fees)
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
        return self._event(name, event, snap, pnl=pnl, detail=detail, closed=position)

    def _event(
        self,
        strategy: str,
        event: str,
        snap: MarketSnapshot,
        pnl: float | None,
        detail: str,
        closed: dict[str, Any] | None = None,
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
            self.db.execute(
                "INSERT INTO snapshots VALUES (" + ",".join(["?"] * 24) + ")",
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
                ],
            )
            for event in events:
                self.db.execute(
                    "INSERT INTO trades VALUES (" + ",".join(["?"] * 11) + ")",
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
            }
            for name in STRATEGIES:
                until = self.cooldowns.get(name)
                state[f"cooldown_{name}"] = until.isoformat() if until else ""
            for key, value in state.items():
                self.db.execute("DELETE FROM session_state WHERE key = ?", [key])
                self.db.execute("INSERT INTO session_state VALUES (?, ?)", [key, value])
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK")
            raise

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
        wait = self.backoff
        self.backoff = min(self.backoff * 2, 300)
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
        "reason": position["reason"],
        "expiration": position.get("expiration") or "",
    }
    return json.dumps(payload)


def _position_from_json(payload: str) -> dict[str, Any]:
    raw = json.loads(payload)
    raw["legs"] = [Leg(**leg) for leg in raw["legs"]]
    raw["entry_time"] = datetime.fromisoformat(raw["entry_time"])
    raw["contracts"] = int(raw["contracts"])
    return raw


def _metrics_json(snap: MarketSnapshot) -> str:
    return json.dumps(
        {
            "spot": snap.spot,
            "intrad_net_gex": snap.intrad_gex,
            "call_wall": snap.call_wall,
            "put_wall": snap.put_wall,
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
                f"Charm {'on' if snap.charm else 'off'}"
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
