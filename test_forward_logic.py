"""Offline checks for the SPY paper-forward rules. No network and no broker."""

from __future__ import annotations

import math
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
    touched_iv_peak,
    round_trip_fees,
    executed_fill,
    fill_price_blocks,
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
    noon = datetime(2026, 10, 8, 12, 0, tzinfo=EASTERN)
    signal = macro_swing_signal(
        now=noon,
        macro_gex=-1,
        gamma_flip=505,
        hmm_state=1,
        ou_theta=0.2,
        fifteen_closes=[500, 502],
    )
    assert signal is not None and signal.structure == "put_debit_macro"
    assert "ENTER_MACRO_PUT_DEBIT" in signal.reason
    assert macro_swing_signal(
        now=noon,
        macro_gex=-1,
        gamma_flip=505,
        hmm_state=1,
        ou_theta=0.0,
        fifteen_closes=[510, 508],
    ) is None
    assert macro_swing_signal(
        now=datetime(2026, 10, 8, 10, 30, tzinfo=EASTERN),
        macro_gex=-1,
        gamma_flip=505,
        hmm_state=1,
        ou_theta=0.2,
        fifteen_closes=[500, 500],
    ) is None
    assert macro_swing_signal(
        now=noon,
        macro_gex=1,
        gamma_flip=505,
        hmm_state=1,
        ou_theta=0.0,
        fifteen_closes=[500, 500],
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
    credit_fill = executed_fill(0.80, 0.04, 0.35)
    assert abs(credit_fill - (0.80 - 0.35 * 0.04)) < 1e-9
    assert fill_price_blocks(0.80, credit_fill, legs) is None
    assert fill_price_blocks(0.80, 0.05, legs) == "credit fill is at or below 0.05"
    debit_fill = executed_fill(-1.20, 0.08, 0.35)
    assert debit_fill < -1.20
    assert fill_price_blocks(-1.20, -2.0, debit) == "debit fill is at least the strike width"


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
        call_wall_2=None,
        put_wall_2=None,
        call_wall_1_gamma=None,
        call_wall_2_gamma=None,
        put_wall_1_gamma=None,
        put_wall_2_gamma=None,
        gex_0_1=1.0,
        total_net_gex=2.0,
        gex_volume=3.0,
        charm_hedge=0.0,
        bar_open=499,
        bar_high=501,
        bar_low=498,
        bar_close=500,
        zero_gamma=500,
        cdf=10,
        atm_iv=0.15,
        atm_call_iv=0.16,
        atm_put_iv=0.14,
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
        minute_bars=pd.DataFrame(),
        momentum_price=500.0,
        momentum_vwap=500.0,
        momentum_slope=0.0,
        iv_peak_center=None,
        iv_max_fear_upper=None,
        iv_max_fear_lower=None,
        static_open_call_wall_1=None,
        static_open_put_wall_1=None,
        static_open_pml=None,
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


def test_walls_expected_move_and_secondary_rules() -> None:
    from datetime import datetime, time

    from gex_engine import net_gex, normalized_gex, rank_gamma_walls, volume_gex
    from ta_engine import expected_move, expected_move_bands, minutes_remaining

    strikes = [100, 101, 102]
    exposure = [10, 9, 5]
    primary, primary_g, secondary, secondary_g = rank_gamma_walls(strikes, exposure)
    assert primary == 100 and primary_g == 10
    assert secondary == 101 and secondary_g == 9
    assert rank_gamma_walls(strikes, [10, 7, 1])[2] is None

    chain = pd.DataFrame({
        "call_gamma": [0.02, 0.01],
        "call_openInterest": [100, 50],
        "put_gamma": [0.01, 0.02],
        "put_openInterest": [40, 80],
    })
    # (0.02*100 - 0.01*40) + (0.01*50 - 0.02*80) = (2 - 0.4) + (0.5 - 1.6) = 0.5
    # times 100 * spot 500 = 25000
    assert net_gex(chain, 500) == 25000

    full_day = 500 * 0.16 * (1 / 252) ** 0.5
    move = expected_move(500, 0.16)
    assert abs(move - full_day) < 1e-9
    assert abs(expected_move(500, 0.16, 195) - full_day * (0.5 ** 0.5)) < 1e-9
    assert expected_move(500, 0.16, 0) == 0
    upper_1, lower_1, upper_2, lower_2 = expected_move_bands(100, 100, 0.10, 390, call_iv=0.20, put_iv=0.10)
    assert abs((upper_1 - 100) - 2.0 * (100 - lower_1)) < 1e-9
    assert abs((upper_2 - 100) - 2.0 * (upper_1 - 100)) < 1e-9
    assert abs((100 - lower_2) - 2.0 * (100 - lower_1)) < 1e-9
    assert minutes_remaining(datetime(2026, 10, 12, 15, 0), time(16, 0)) == 60
    volume_chain = chain.copy()
    volume_chain["call_volume"] = [100, 50]
    volume_chain["put_volume"] = [40, 80]
    assert volume_gex(volume_chain, 500) == 25000
    assert normalized_gex(25000, 500) == 25000 * 0.01 * 500

    bar = _bar(100, 101, 98, 99.5)
    faded = absorption_signal(
        bar,
        intrad_gex=1,
        macro_gex=1,
        hmm_state=0,
        call_wall=110,
        put_wall=100,
        upper_2=120,
        lower_2=80,
        upper_1=105,
        lower_1=99,
        call_wall_2=112,
        put_wall_2=97,
        cdf_now=5,
        cdf_prev=4,
        iv_now=0.1,
        iv_prev=0.1,
    )
    assert faded is not None and faded.stop == 97

    previous = _bar(100, 101, 99, 100)
    through_primary = _bar(100, 106, 100, 105)
    through_secondary = _bar(100, 113, 100, 112.5)
    blocked = cascade_signal(
        previous,
        through_primary,
        intrad_gex=-1,
        macro_gex=-1,
        gamma_flip=90,
        spot=105,
        call_wall=104,
        put_wall=90,
        zero_gamma=101,
        call_wall_2=112,
        vwap=100,
        cdf_now=20,
        cdf_15=10,
        cdf_30=5,
        iv_now=0.2,
        iv_prev=0.1,
    )
    assert blocked is None
    confirmed = cascade_signal(
        previous,
        through_secondary,
        intrad_gex=-1,
        macro_gex=-1,
        gamma_flip=90,
        spot=112.5,
        call_wall=104,
        put_wall=90,
        zero_gamma=101,
        call_wall_2=112,
        vwap=100,
        cdf_now=20,
        cdf_15=10,
        cdf_30=5,
        iv_now=0.2,
        iv_prev=0.1,
    )
    assert confirmed is not None and confirmed.anchor == 112


def test_charm_and_iv_zero_gamma() -> None:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from scipy.stats import norm

    from gex_engine import (
        DIVIDEND_YIELD,
        RISK_FREE,
        black_scholes_charm,
        black_scholes_gamma,
        build_intraday_snapshot,
        charm_to_close,
    )

    spot, strike, sigma, t_years = 100.0, 100.0, 0.20, 1.0 / 252.0
    sqrt_t = t_years ** 0.5
    d1 = (math.log(spot / strike) + (RISK_FREE - DIVIDEND_YIELD + 0.5 * sigma**2) * t_years) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    discount = math.exp(-DIVIDEND_YIELD * t_years)
    term1 = DIVIDEND_YIELD * discount * norm.cdf(d1)
    term2 = discount * norm.pdf(d1) * (
        (2 * (RISK_FREE - DIVIDEND_YIELD) * t_years - d2 * sigma * sqrt_t) / (2 * t_years * sqrt_t * sigma)
    )
    charm = black_scholes_charm(spot, [strike], t_years, [sigma], "call")
    assert abs(float(charm[0]) - (term1 - term2)) < 1e-9
    gamma = black_scholes_gamma(spot, [100.0, 150.0], t_years, [0.2, 0.2])
    assert gamma[0] > gamma[1] > 0

    def chain(put_iv: float) -> pd.DataFrame:
        return pd.DataFrame({
            "strikePrice": [99.0, 100.0, 101.0],
            "call_openInterest": [40, 100, 900],
            "call_gamma": [0.01, 0.02, 0.01],
            "call_delta": [0.6, 0.5, 0.4],
            "call_volatility": [0.16, 0.16, 0.16],
            "call_volume": [1, 1, 1],
            "put_openInterest": [900, 100, 40],
            "put_gamma": [0.01, 0.02, 0.01],
            "put_delta": [-0.4, -0.5, -0.6],
            "put_volatility": [put_iv, put_iv, put_iv],
            "put_volume": [1, 1, 1],
        })

    as_of = datetime(2026, 10, 13, 10, 0, tzinfo=ZoneInfo("America/New_York"))
    calm = build_intraday_snapshot(chain(0.12), 100.0, as_of)
    stressed = build_intraday_snapshot(chain(0.16), 100.0, as_of)
    assert calm.zero_gamma is not None and stressed.zero_gamma is not None
    assert stressed.zero_gamma != calm.zero_gamma
    assert calm.charm_hedge != 0
    assert charm_to_close(chain(0.2), 100.0, as_of, 0) == 0


def test_momentum_and_skew_guards() -> None:
    from gex_engine import momentum_features

    index = pd.date_range("2026-10-08 10:00", periods=15, freq="1min", tz=EASTERN)
    rising = pd.DataFrame(
        {"Open": 100, "High": 101, "Low": 99, "Close": [100 + i / 14 for i in range(15)], "Volume": 1},
        index=index,
    )
    price, _vwap, slope = momentum_features(rising)
    assert price == 101
    assert slope > 0.05
    falling = rising.copy()
    falling["Close"] = [101 - i / 14 for i in range(15)]
    assert momentum_features(falling)[2] < -0.05

    blocked = absorption_signal(
        _bar(100, 101, 98, 99.5),
        intrad_gex=1,
        macro_gex=1,
        hmm_state=0,
        call_wall=110,
        put_wall=100,
        upper_2=120,
        lower_2=80,
        upper_1=105,
        lower_1=99,
        cdf_now=5,
        cdf_prev=4,
        iv_now=0.1,
        iv_prev=0.1,
        momentum_slope=-0.06,
    )
    assert blocked is None


def test_cdf_flat_needs_three_polls() -> None:
    from live_signals import next_cdf_flat_streak

    streak = 0
    for _ in range(2):
        streak, confirmed = next_cdf_flat_streak(streak, True)
        assert not confirmed
    streak, confirmed = next_cdf_flat_streak(streak, True)
    assert streak == 3 and confirmed
    streak, confirmed = next_cdf_flat_streak(streak, False)
    assert streak == 0 and not confirmed


def test_level_spikes_are_held() -> None:
    from gex_engine import LevelState, stabilize_level, stabilize_series

    state = LevelState()
    for reading in (777, 777, 784, 777):
        state = stabilize_level(reading, state)
    assert state.accepted == 777

    state = LevelState()
    for reading in (778, 775, 778):
        state = stabilize_level(reading, state)
    assert state.accepted == 778

    state = LevelState()
    for reading in (778, 775, 775):
        state = stabilize_level(reading, state)
    assert state.accepted == 775

    state = LevelState()
    for reading in (780, 777, 777, 777):
        state = stabilize_level(reading, state)
    assert state.accepted == 777

    state = LevelState()
    for reading in (777, 778):
        state = stabilize_level(reading, state)
    assert state.accepted == 778

    cleaned = stabilize_series([777.25, 772.12, 777.25, 772.05, 777.31])
    assert cleaned == [777.25, 777.25, 777.25, 777.25, 777.31]


def test_open_levels_peak_and_fear() -> None:
    from gex_engine import iv_max_fear_levels, iv_peak_center, overnight_pml

    chain = pd.DataFrame({
        "strikePrice": [96, 98, 100, 101, 102, 104],
        "call_gamma": [0.0, 0.0, 0.01, 0.02, 0.08, 0.001],
        "put_gamma": [0.001, 0.05, 0.01, 0.0, 0.0, 0.0],
        "call_openInterest": [10, 10, 10, 10, 100, 100],
        "put_openInterest": [100, 100, 10, 10, 10, 10],
        "call_volatility": [0.10, 0.12, 0.20, 0.20, 0.11, 0.10],
        "put_volatility": [0.10, 0.12, 0.20, 0.20, 0.11, 0.10],
    })
    # Call liability peaks at 102 (0.05 × 100). Put liability peaks at 98.
    # Combined liability is largest at 102.
    assert overnight_pml(chain) == 102
    center = math.exp((math.log(100) + math.log(101)) / 2)
    assert iv_peak_center(chain) is not None
    assert abs(iv_peak_center(chain) - center) < 1e-9
    fear_upper, fear_lower = iv_max_fear_levels(chain, 100)
    assert fear_upper == 104
    assert fear_lower == 96

    bullish = absorption_signal(
        _bar(100, 101, 98, 99.5),
        intrad_gex=1,
        macro_gex=1,
        hmm_state=0,
        call_wall=110,
        put_wall=100,
        upper_2=120,
        lower_2=80,
        upper_1=105,
        lower_1=99,
        call_wall_2=112,
        put_wall_2=97,
        cdf_now=5,
        cdf_prev=4,
        iv_now=0.1,
        iv_prev=0.1,
        iv_max_fear_lower=96,
    )
    assert bullish is not None and bullish.stop == 96
    bearish = absorption_signal(
        _bar(100, 111, 99, 109),
        intrad_gex=1,
        macro_gex=1,
        hmm_state=0,
        call_wall=110,
        put_wall=90,
        upper_2=120,
        lower_2=80,
        upper_1=115,
        lower_1=85,
        cdf_now=4,
        cdf_prev=5,
        iv_now=0.1,
        iv_prev=0.1,
        iv_max_fear_upper=114,
    )
    assert bearish is not None and bearish.stop == 114
    assert touched_iv_peak("bullish", 101.2, 99, 101)
    assert touched_iv_peak("bearish", 99.5, 102, 100)
    assert not touched_iv_peak("bullish", 100, 101, 99)

    os.environ.setdefault("SCHWAB_APP_KEY", "test-key")
    os.environ.setdefault("SCHWAB_APP_SECRET", "test-secret")
    os.environ.setdefault("SCHWAB_REFRESH_TOKEN", "test-token")
    os.environ["DISCORD_WEBHOOK_URL"] = ""
    with tempfile.TemporaryDirectory() as folder:
        os.environ["DUCKDB_PATH"] = os.path.join(folder, "forward_test.db")
        from live_forward import ForwardTester

        bot = ForwardTester()
        opening = datetime(2026, 10, 14, 9, 30, tzinfo=EASTERN)
        bot._lock_open_levels(opening, 510.0, 490.0, chain)
        assert bot.static_open_call_wall == 510
        assert bot.static_open_put_wall == 490
        assert bot.static_open_pml == 102
        bot._lock_open_levels(opening.replace(hour=10), 520.0, 480.0, chain)
        assert bot.static_open_call_wall == 510
        bot._persist(_snapshot(), [])
        saved = {
            str(key): str(value)
            for key, value in bot.db.execute("SELECT key, value FROM session_state").fetchall()
        }
        assert saved["static_open_day"] == "2026-10-14"
        assert saved["static_open_call_wall_1"] == "510.0"
        assert saved["static_open_put_wall_1"] == "490.0"
        assert saved["static_open_pml"] == "102.0"
        bot.shutdown()
        reloaded = ForwardTester()
        assert reloaded.static_open_call_wall == 510
        assert reloaded.static_open_put_wall == 490
        assert reloaded.static_open_pml == 102
        reloaded.shutdown()


def test_audit_risk_fixes() -> None:
    from datetime import datetime as clock
    from gex_engine import (
        classify_aggressor,
        classified_flow_update,
        positioning_weight,
        positioning_weights,
    )
    from live_signals import Leg, exit_slippage_factor, intrabar_stop

    open_oi, open_vol = positioning_weights(clock(2026, 10, 14, 9, 30, tzinfo=EASTERN))
    assert open_oi == 1.0 and open_vol == 0.0
    before = positioning_weights(clock(2026, 10, 14, 10, 29, tzinfo=EASTERN))
    after = positioning_weights(clock(2026, 10, 14, 10, 31, tzinfo=EASTERN))
    assert abs((after[1] - before[1]) - (before[1] - positioning_weights(clock(2026, 10, 14, 10, 27, tzinfo=EASTERN))[1])) < 0.02
    blended = positioning_weight([100.0], [50.0], clock(2026, 10, 14, 10, 30, tzinfo=EASTERN))
    assert blended[0] > 40 and blended[0] < 100

    assert classify_aggressor(1.20, 1.00, 1.10, 1.00) == 1
    assert classify_aggressor(1.00, 1.00, 1.10, 1.20) == -1
    assert classify_aggressor(1.05, 1.00, 1.10, 1.05) == 0

    chain = pd.DataFrame({
        "strikePrice": [100.0],
        "call_volume": [10.0],
        "call_delta": [0.5],
        "call_last": [1.20],
        "call_bid": [1.00],
        "call_ask": [1.10],
        "put_volume": [8.0],
        "put_delta": [-0.4],
        "put_last": [0.90],
        "put_bid": [0.80],
        "put_ask": [1.00],
    })
    added, baseline = classified_flow_update(chain, None)
    assert added == 0.0
    chain.loc[0, "call_volume"] = 15.0
    added, _next = classified_flow_update(chain, baseline)
    assert abs(added - 5.0 * 0.5) < 1e-9

    legs = [Leg("put", 100, "short"), Leg("put", 98, "long")]
    assert intrabar_stop("bullish", 99.0, 100.0, 0.80, 1.20, 1, legs) == "spot through the structural stop"
    assert exit_slippage_factor("cascade", "OPEN", 0.35) == 1.0
    assert exit_slippage_factor("absorption", "STOP", 0.35) == 1.0
    assert exit_slippage_factor("absorption", "OPEN", 0.35) == 0.35


def test_price_history_redraws_the_session() -> None:
    os.environ.setdefault("SCHWAB_APP_KEY", "test-key")
    os.environ.setdefault("SCHWAB_APP_SECRET", "test-secret")
    os.environ.setdefault("SCHWAB_REFRESH_TOKEN", "test-token")
    os.environ["DISCORD_WEBHOOK_URL"] = ""
    with tempfile.TemporaryDirectory() as folder:
        os.environ["DUCKDB_PATH"] = os.path.join(folder, "forward_test.db")
        from live_forward import ForwardTester
        from live_view import build_live_view

        bot = ForwardTester()
        snap = _snapshot()
        gap = pd.date_range("2026-10-08 09:30", periods=2, freq="2min", tz=EASTERN)
        snap.minute_bars = pd.DataFrame(
            {"Open": [1.0, 3.0], "High": [2.0, 4.0], "Low": [0.5, 2.5], "Close": [1.5, 3.5]},
            index=gap,
        )
        bot._persist(snap, [])
        gappy = build_live_view(bot.db)["candles"]
        assert [row["time"] for row in gappy].count(int(pd.Timestamp("2026-10-08 09:31", tz=EASTERN).timestamp())) == 0

        full = pd.date_range("2026-10-08 09:30", periods=3, freq="1min", tz=EASTERN)
        full = full.append(pd.DatetimeIndex([pd.Timestamp("2026-10-08 10:00", tz=EASTERN)]))
        snap.minute_bars = pd.DataFrame(
            {"Open": [1.0, 2.0, 3.0, 4.0], "High": [2.0, 2.5, 4.0, 4.2], "Low": [0.5, 1.5, 2.5, 3.8], "Close": [1.5, 2.2, 3.5, 4.0]},
            index=full,
        )
        snap.spot = 501.0
        bot._persist(snap, [])
        candles = build_live_view(bot.db)["candles"]
        stamps = [datetime.fromtimestamp(row["time"], EASTERN).strftime("%H:%M") for row in candles]
        assert stamps == ["09:30", "09:31", "09:32", "10:00"]
        assert candles[-1]["close"] == 501.0
        assert candles[-1]["high"] == 501.0
        assert candles[-1]["low"] == 3.8
        bot.shutdown()


if __name__ == "__main__":
    test_calendar()
    test_absorption_and_cascade()
    test_macro_signal_and_exits()
    test_structures_slippage_and_pnl()
    test_charm_and_five_minute_clock()
    test_ou_and_persistence()
    test_chain_parser_and_session_bars()
    test_paper_absorption_roundtrip()
    test_level_spikes_are_held()
    test_walls_expected_move_and_secondary_rules()
    test_charm_and_iv_zero_gamma()
    test_cdf_flat_needs_three_polls()
    test_momentum_and_skew_guards()
    test_open_levels_peak_and_fear()
    test_price_history_redraws_the_session()
    test_audit_risk_fixes()
    print("forward logic checks passed")
