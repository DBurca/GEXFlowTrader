"""Offline checks for the SPY paper-forward rules. No network and no broker."""

from __future__ import annotations

import os
import tempfile
from datetime import date, datetime, time

import pandas as pd

from event_calendar import (
    EASTERN,
    is_trading_day,
    macro_blackout,
    next_session_open,
    session_close_time,
)
from live_signals import (
    Leg,
    absorption_signal,
    breach_confirmed,
    cascade_signal,
    charm_pin_active,
    contract_count,
    credit_legs,
    debit_legs,
    macro_swing_signal,
    max_loss_per_contract,
    minutes_before_close,
    ornstein_uhlenbeck,
    reverted_to_mean,
    round_trip_fees,
    slippage_blocks,
    structure_nets,
    trade_pnl,
    completed_five_minute,
    five_minute_bars,
)


def _bar(open_, high, low, close) -> pd.Series:
    return pd.Series({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": 1})


def _chain(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_calendar() -> None:
    assert not is_trading_day(date(2026, 1, 1))
    assert not is_trading_day(date(2026, 7, 3))
    assert not is_trading_day(date(2026, 10, 10))
    assert is_trading_day(date(2026, 10, 8))
    assert session_close_time(date(2026, 11, 27)) == time(13, 0)
    assert session_close_time(date(2026, 10, 8)) == time(16, 0)

    fomc = datetime(2026, 10, 28, 13, 50, tzinfo=EASTERN)
    assert macro_blackout(fomc) == "FOMC"
    assert macro_blackout(datetime(2026, 10, 28, 14, 20, tzinfo=EASTERN)) is None

    cpi_morning = datetime(2026, 10, 14, 9, 35, tzinfo=EASTERN)
    assert macro_blackout(cpi_morning) == "CPI"
    assert macro_blackout(datetime(2026, 10, 14, 9, 0, tzinfo=EASTERN)) is None
    assert macro_blackout(datetime(2026, 10, 14, 8, 20, tzinfo=EASTERN)) == "CPI"

    saturday = datetime(2026, 10, 10, 12, 0, tzinfo=EASTERN)
    opened = next_session_open(saturday)
    assert opened.date() == date(2026, 10, 12)
    assert opened.hour == 9 and opened.minute == 30


def test_absorption_and_cascade() -> None:
    held = _bar(560, 561, 555, 558)
    signal = absorption_signal(
        held,
        intrad_gex=1,
        macro_gex=1,
        hmm_state=0,
        call_wall=570,
        put_wall=556,
        upper_2=580,
        lower_2=540,
        cdf_now=10,
        cdf_prev=0,
        iv_now=0.15,
        iv_prev=0.16,
    )
    assert signal is not None and signal.structure == "put_credit"

    rejected = absorption_signal(
        held,
        intrad_gex=-1,
        macro_gex=1,
        hmm_state=0,
        call_wall=570,
        put_wall=556,
        upper_2=580,
        lower_2=540,
        cdf_now=10,
        cdf_prev=0,
        iv_now=0.15,
        iv_prev=0.16,
    )
    assert rejected is None

    previous = _bar(550, 552, 549, 550)
    broke = _bar(550, 556, 550, 555)
    cascade = cascade_signal(
        previous,
        broke,
        intrad_gex=-1,
        macro_gex=-1,
        gamma_flip=560,
        spot=555,
        call_wall=554,
        put_wall=540,
        zero_gamma=548,
        vwap=551,
        cdf_now=30,
        cdf_15=10,
        cdf_30=5,
        iv_now=0.20,
        iv_prev=0.15,
    )
    assert cascade is not None and cascade.direction == "bullish"
    assert cascade.anchor == 554

    quiet = cascade_signal(
        previous,
        broke,
        intrad_gex=-1,
        macro_gex=-1,
        gamma_flip=560,
        spot=555,
        call_wall=554,
        put_wall=540,
        zero_gamma=548,
        vwap=551,
        cdf_now=30,
        cdf_15=10,
        cdf_30=5,
        iv_now=0.151,
        iv_prev=0.15,
    )
    assert quiet is None


def test_macro_signal_and_exits() -> None:
    signal = macro_swing_signal(
        spot=500,
        previous_close=510,
        macro_gex=-1,
        gamma_flip=505,
        hmm_state=1,
        previous_hmm=0,
        ou_theta=0.2,
    )
    assert signal is not None and signal.structure == "put_debit_macro"
    assert macro_swing_signal(
        spot=510,
        previous_close=500,
        macro_gex=-1,
        gamma_flip=505,
        hmm_state=1,
        previous_hmm=0,
        ou_theta=0.0,
    ) is None
    assert breach_confirmed("bullish", 549, 550, 1, 5)
    assert not breach_confirmed("bullish", 549, 550, 6, 5)
    assert reverted_to_mean("bullish", 560, 550, 558, 570)


def test_structures_slippage_and_pnl() -> None:
    chain = _chain(
        [
            {
                "strikePrice": 558,
                "put_bid": 0.40,
                "put_ask": 0.42,
                "call_bid": 0,
                "call_ask": 0,
            },
            {
                "strikePrice": 560,
                "put_bid": 1.20,
                "put_ask": 1.22,
                "call_bid": 0,
                "call_ask": 0,
            },
        ]
    )
    legs = credit_legs(chain, "put_credit", 560)
    assert legs == [Leg("put", 560, "short"), Leg("put", 558, "long")]
    assert slippage_blocks(chain, legs, 0.05) is None
    signed, _width = structure_nets(chain, legs)
    assert abs(signed - 0.80) < 1e-9
    loss = max_loss_per_contract(signed, legs)
    assert abs(loss - 120.0) < 1e-9
    assert contract_count(100_000, loss) == 16

    wide = _chain(
        [
            {"strikePrice": 558, "put_bid": 0.40, "put_ask": 0.70, "call_bid": 0, "call_ask": 0},
            {"strikePrice": 560, "put_bid": 1.00, "put_ask": 1.40, "call_bid": 0, "call_ask": 0},
        ]
    )
    wide_legs = credit_legs(wide, "put_credit", 560)
    assert slippage_blocks(wide, wide_legs, 0.05) is not None

    calls = _chain(
        [
            {"strikePrice": 560, "call_bid": 2.00, "call_ask": 2.04, "put_bid": 0, "put_ask": 0},
            {"strikePrice": 562, "call_bid": 0.80, "call_ask": 0.84, "put_bid": 0, "put_ask": 0},
        ]
    )
    debit = debit_legs(calls, "call_debit", 560.4, width=2)
    assert debit[0] == Leg("call", 560, "long")
    assert debit[1] == Leg("call", 562, "short")
    debit_net, _ba = structure_nets(calls, debit)
    pnl = trade_pnl(debit_net, debit_net - 0.50, 2, round_trip_fees(2, 2))
    assert abs(pnl - (100.0 - 5.20)) < 1e-9
    assert contract_count(100_000, 5_000) == 0


def test_charm_and_five_minute_clock() -> None:
    pinned = datetime(2026, 10, 8, 15, 5, tzinfo=EASTERN)
    assert charm_pin_active(pinned, 500, 500.5, time(16, 0))
    assert not charm_pin_active(datetime(2026, 10, 8, 14, 30, tzinfo=EASTERN), 500, 500.5, time(16, 0))
    assert minutes_before_close(datetime(2026, 11, 27, 12, 45), time(13, 0), 15)
    assert not minutes_before_close(datetime(2026, 11, 27, 12, 44), time(13, 0), 15)

    index = pd.date_range("2026-10-08 09:30", periods=12, freq="1min", tz=EASTERN)
    bars = pd.DataFrame(
        {"Open": 1, "High": 2, "Low": 0.5, "Close": 1.2, "Volume": 10},
        index=index,
    )
    finished = completed_five_minute(
        five_minute_bars(bars),
        datetime(2026, 10, 8, 9, 41, tzinfo=EASTERN),
    )
    assert len(finished) == 2
    assert finished.index[0].minute == 30


def test_ou_and_persistence() -> None:
    anchored = [10 + ((-1) ** i) * 0.01 for i in range(40)]
    theta, _mu, _sigma = ornstein_uhlenbeck(anchored)
    assert theta > 0.2

    os.environ.setdefault("SCHWAB_APP_KEY", "test-key")
    os.environ.setdefault("SCHWAB_APP_SECRET", "test-secret")
    os.environ.setdefault("SCHWAB_REFRESH_TOKEN", "test-token")
    os.environ["DISCORD_WEBHOOK_URL"] = ""
    with tempfile.TemporaryDirectory() as folder:
        os.environ["DUCKDB_PATH"] = os.path.join(folder, "forward_test.db")
        from live_forward import ForwardTester, MarketSnapshot

        bot = ForwardTester()
        assert bot.balances["cumulative"] == 300_000
        snap = _snapshot()
        bot._persist(snap, [])
        rows = bot.db.execute("SELECT COUNT(*) FROM snapshots").fetchone()
        assert rows[0] == 1
        try:
            bot._persist(snap, [{"strategy": "absorption"}])
        except Exception:
            pass
        rows = bot.db.execute("SELECT COUNT(*) FROM snapshots").fetchone()
        assert rows[0] == 1
        bot.shutdown()


def _snapshot() -> "MarketSnapshot":
    from live_forward import MarketSnapshot

    now = datetime(2026, 10, 8, 10, 0, tzinfo=EASTERN)
    return MarketSnapshot(
        now=now,
        spot=500,
        intrad_gex=1,
        call_wall=510,
        put_wall=490,
        zero_gamma=500,
        cdf=10,
        atm_iv=0.15,
        vwap=500,
        upper_1=505,
        lower_1=495,
        upper_2=510,
        lower_2=490,
        macro_gex=1,
        gamma_flip=520,
        macro_regime="mean_reverting",
        hmm=0,
        ou_theta=0.2,
        ou_mu=500,
        ou_sigma=1,
        pin_strike=500,
        blackout=None,
        charm=False,
        entry_block=None,
        chain=pd.DataFrame(),
        macro_chain=pd.DataFrame(),
        macro_expiration=None,
        macro_dte=None,
        bars_5=pd.DataFrame(),
    )


def test_chain_parser_and_session_bars() -> None:
    from live_market import _chain_frame, session_bars

    payload = {
        "callExpDateMap": {
            "2026-10-08:0": {
                "560.0": [
                    {
                        "strikePrice": 560,
                        "gamma": 0.01,
                        "delta": 0.5,
                        "volatility": 20,
                        "openInterest": 100,
                        "totalVolume": 50,
                        "bid": 1.0,
                        "ask": 1.1,
                        "daysToExpiration": 0,
                    }
                ]
            }
        },
        "putExpDateMap": {
            "2026-10-08:0": {
                "560.0": [
                    {
                        "strikePrice": 560,
                        "gamma": 0.02,
                        "delta": -0.5,
                        "volatility": 22,
                        "openInterest": 80,
                        "totalVolume": 40,
                        "bid": 1.2,
                        "ask": 1.3,
                        "daysToExpiration": 0,
                    }
                ]
            }
        },
    }
    frame = _chain_frame(payload, lambda dte, exp: dte == 0 and exp == date(2026, 10, 8))
    assert abs(float(frame.iloc[0]["call_volatility"]) - 0.20) < 1e-9
    assert float(frame.iloc[0]["put_openInterest"]) == 80

    index = pd.date_range("2026-10-08 09:28", periods=5, freq="1min", tz=EASTERN)
    bars = pd.DataFrame(
        {"Open": 1, "High": 1, "Low": 1, "Close": 1, "Volume": 1},
        index=index,
    )
    kept = session_bars(bars, date(2026, 10, 8), time(16, 0))
    assert list(kept.index.minute) == [30, 31, 32]


def test_paper_absorption_roundtrip() -> None:
    os.environ.setdefault("SCHWAB_APP_KEY", "test-key")
    os.environ.setdefault("SCHWAB_APP_SECRET", "test-secret")
    os.environ.setdefault("SCHWAB_REFRESH_TOKEN", "test-token")
    os.environ["DISCORD_WEBHOOK_URL"] = ""
    with tempfile.TemporaryDirectory() as folder:
        os.environ["DUCKDB_PATH"] = os.path.join(folder, "forward_test.db")
        from live_forward import ForwardTester

        bot = ForwardTester()
        chain = pd.DataFrame(
            [
                {"strikePrice": 555, "put_bid": 0.40, "put_ask": 0.42, "call_bid": 1, "call_ask": 1.02},
                {"strikePrice": 556, "put_bid": 1.20, "put_ask": 1.22, "call_bid": 1, "call_ask": 1.02},
            ]
        )
        armed = _snapshot()
        armed.now = datetime(2026, 10, 8, 10, 0, tzinfo=EASTERN)
        armed.chain = chain
        armed.bars_5 = _five([("09:55", 560, 561, 559, 560)])
        assert bot._decide(armed) == []

        bot.cdf_history = [(datetime(2026, 10, 8, 9, 45, tzinfo=EASTERN), 0.0)]
        bot.iv_history = [(datetime(2026, 10, 8, 9, 45, tzinfo=EASTERN), 0.16)]
        opened_snap = _snapshot()
        opened_snap.now = datetime(2026, 10, 8, 10, 5, tzinfo=EASTERN)
        opened_snap.chain = chain
        opened_snap.spot = 558
        opened_snap.cdf = 10
        opened_snap.atm_iv = 0.15
        opened_snap.hmm = 0
        opened_snap.intrad_gex = 1
        opened_snap.macro_gex = 1
        opened_snap.put_wall = 556
        opened_snap.call_wall = 570
        opened_snap.bars_5 = _five(
            [("09:55", 560, 561, 559, 560), ("10:00", 560, 561, 555, 558)]
        )
        opened = bot._decide(opened_snap)
        assert [event["event"] for event in opened] == ["OPEN"]
        assert bot.positions["absorption"]["contracts"] >= 1
        assert bot.balances["absorption"] == 100_000

        stopped_snap = _snapshot()
        stopped_snap.now = datetime(2026, 10, 8, 10, 10, tzinfo=EASTERN)
        stopped_snap.chain = chain
        stopped_snap.spot = 550
        stopped_snap.cdf = -20
        stopped_snap.vwap = 570
        stopped_snap.zero_gamma = 575
        stopped_snap.bars_5 = _five(
            [
                ("09:55", 560, 561, 559, 560),
                ("10:00", 560, 561, 555, 558),
                ("10:05", 558, 558, 549, 550),
            ]
        )
        stopped = bot._decide(stopped_snap)
        assert [event["event"] for event in stopped] == ["STOP"]
        assert "absorption" not in bot.positions
        assert bot.balances["absorption"] < 100_000
        assert bot.balances["cumulative"] == (
            bot.balances["absorption"] + bot.balances["cascade"] + bot.balances["macro"]
        )
        bot._persist(stopped_snap, stopped)
        stored = bot.db.execute("SELECT event, strategy FROM trades").fetchall()
        assert stored == [("STOP", "absorption")]
        bot.shutdown()


def _five(rows: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    index = pd.DatetimeIndex(
        [pd.Timestamp(f"2026-10-08 {stamp}", tz=EASTERN) for stamp, *_rest in rows]
    )
    return pd.DataFrame(
        {
            "Open": [row[1] for row in rows],
            "High": [row[2] for row in rows],
            "Low": [row[3] for row in rows],
            "Close": [row[4] for row in rows],
            "Volume": [1] * len(rows),
        },
        index=index,
    )


if __name__ == "__main__":
    test_calendar()
    test_absorption_and_cascade()
    test_macro_signal_and_exits()
    test_structures_slippage_and_pnl()
    test_charm_and_five_minute_clock()
    test_ou_and_persistence()
    test_chain_parser_and_session_bars()
    test_paper_absorption_roundtrip()
    print("forward logic checks passed")
