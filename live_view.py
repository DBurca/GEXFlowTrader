"""Publish the latest session for the read-only chart.

The forward tester holds the only DuckDB lock. After each commit it writes
this JSON beside the database. The dashboard reads the file and never opens
the database.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from typing import Any

from event_calendar import session_close_time
from gex_engine import normalized_gex, stabilize_optional_series, stabilize_series
from ta_engine import expected_move_bands, minutes_remaining

STRATEGIES = ("absorption", "cascade", "macro")


def live_view_path() -> str:
    explicit = os.environ.get("LIVE_VIEW_PATH")
    if explicit:
        return explicit
    database = os.environ.get("DUCKDB_PATH", "forward_test.db")
    folder = os.path.dirname(os.path.abspath(database)) or "."
    return os.path.join(folder, "live_view.json")


def publish_live_view(db, path: str | None = None) -> None:
    destination = path or live_view_path()
    payload = build_live_view(db)
    folder = os.path.dirname(destination)
    if folder:
        os.makedirs(folder, exist_ok=True)
    temporary = destination + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))
    os.replace(temporary, destination)


def build_live_view(db) -> dict[str, Any]:
    latest_row = db.execute("SELECT max(ts) FROM snapshots").fetchone()
    latest_ts = latest_row[0] if latest_row else None
    if not latest_ts:
        return {
            "updated": None,
            "session_date": None,
            "symbol": "SPY",
            "latest": None,
            "balances": _balances(db),
            "positions": _positions(db),
            "price": [],
            "flow": [],
            "pnl": [],
            "trades": [],
        }

    session_date = str(latest_ts)[:10]
    available = {row[0] for row in db.execute("DESCRIBE snapshots").fetchall()}
    extra_names = [
        name
        for name in (
            "call_wall_2_strike",
            "put_wall_2_strike",
            "gex_0_1",
            "total_net_gex",
            "gex_volume",
            "bar_open",
            "bar_high",
            "bar_low",
            "bar_close",
        )
        if name in available
    ]
    rows = db.execute(
        """
        SELECT ts, spot, call_wall, put_wall, zero_gamma, cdf, atm_iv, vwap,
               upper_1, lower_1, upper_2, lower_2,
               intrad_net_gex, macro_net_gex, macro_gamma_flip, macro_regime,
               hmm_state, entry_block_reason
        """
        + "".join(f", {name}" for name in extra_names)
        + """
        FROM snapshots
        WHERE starts_with(ts, ?)
        ORDER BY ts
        """,
        [session_date],
    ).fetchall()
    extra_index = {name: 18 + index for index, name in enumerate(extra_names)}

    def _extra(row: tuple, name: str):
        index = extra_index.get(name)
        if index is None:
            return None
        return row[index]

    call_walls = stabilize_series(row[2] for row in rows)
    put_walls = stabilize_series(row[3] for row in rows)
    zero_gammas = stabilize_series(row[4] for row in rows)
    call_walls_2 = stabilize_optional_series(_extra(row, "call_wall_2_strike") for row in rows)
    put_walls_2 = stabilize_optional_series(_extra(row, "put_wall_2_strike") for row in rows)
    price = []
    flow = []
    gex = []
    previous_time = None
    for index, row in enumerate(rows):
        moment = _unix(row[0])
        if previous_time is not None and moment - previous_time > 180:
            price.append({"time": previous_time + 1})
            flow.append({"time": previous_time + 1})
            gex.append({"time": previous_time + 1})
        spot = _num(row[1])
        vwap = _num(row[7])
        atm_iv = _num(row[6])
        stamped = datetime.fromisoformat(str(row[0]))
        upper_1, lower_1, upper_2, lower_2 = expected_move_bands(
            float("nan") if vwap is None else vwap,
            float("nan") if spot is None else spot,
            float("nan") if atm_iv is None else atm_iv,
            minutes_remaining(stamped, session_close_time(stamped.date())),
        )
        macro_gex = _num(row[13])
        near_gex = _num(_extra(row, "gex_0_1"))
        total_gex = _num(_extra(row, "total_net_gex"))
        if total_gex is None and near_gex is not None and macro_gex is not None:
            total_gex = near_gex + macro_gex
        bar_open = _num(_extra(row, "bar_open"))
        bar_high = _num(_extra(row, "bar_high"))
        bar_low = _num(_extra(row, "bar_low"))
        bar_close = _num(_extra(row, "bar_close"))
        if bar_open is None:
            bar_open = bar_high = bar_low = bar_close = spot
        price.append({
            "time": moment,
            "spot": spot,
            "open": bar_open,
            "high": bar_high,
            "low": bar_low,
            "close": bar_close,
            "call_wall": _num(call_walls[index]),
            "put_wall": _num(put_walls[index]),
            "call_wall_2": _num(call_walls_2[index]),
            "put_wall_2": _num(put_walls_2[index]),
            "zero_gamma": _num(zero_gammas[index]),
            "vwap": vwap,
            "upper_1": _num(upper_1),
            "lower_1": _num(lower_1),
            "upper_2": _num(upper_2),
            "lower_2": _num(lower_2),
        })
        flow.append({
            "time": moment,
            "cdf": _num(row[5]),
            "atm_iv": _percent(row[6]),
        })
        gex.append({
            "time": moment,
            "near": near_gex,
            "total": total_gex,
            "macro": macro_gex,
            "volume": _num(_extra(row, "gex_volume")),
        })
        previous_time = moment

    last = rows[-1]
    balances = _balances(db)
    trades = _trades(db, session_date)
    return {
        "updated": str(latest_ts),
        "session_date": session_date,
        "symbol": "SPY",
        "latest": {
            "ts": str(last[0]),
            "spot": _num(last[1]),
            "call_wall": _num(call_walls[-1]),
            "put_wall": _num(put_walls[-1]),
            "call_wall_2": _num(call_walls_2[-1]),
            "put_wall_2": _num(put_walls_2[-1]),
            "gex_0_1": gex[-1]["near"] if gex else None,
            "total_net_gex": gex[-1]["total"] if gex else None,
            "gex_volume": gex[-1].get("volume") if gex else None,
            "hedge_1pct": (
                _num(normalized_gex(gex[-1]["total"], float(last[1])))
                if gex and gex[-1]["total"] is not None and last[1] is not None
                else None
            ),
            "zero_gamma": _num(zero_gammas[-1]),
            "cdf": _num(last[5]),
            "atm_iv": _percent(last[6]),
            "vwap": _num(last[7]),
            "upper_2": price[-1]["upper_2"] if price else None,
            "lower_2": price[-1]["lower_2"] if price else None,
            "upper_1": price[-1]["upper_1"] if price else None,
            "lower_1": price[-1]["lower_1"] if price else None,
            "intrad_gex": _num(last[12]),
            "macro_gex": _num(last[13]),
            "gamma_flip": _num(last[14]),
            "macro_regime": last[15],
            "hmm": None if last[16] is None else int(last[16]),
            "entry_block": last[17] or None,
        },
        "balances": balances,
        "positions": _positions(db),
        "price": price,
        "flow": flow,
        "gex": gex,
        "pnl": _pnl_series(rows, trades, balances),
        "trades": trades,
    }


def _balances(db) -> dict[str, float]:
    found = {
        str(name): float(balance)
        for name, balance in db.execute("SELECT strategy, balance FROM balances").fetchall()
    }
    balances = {name: found.get(name, 0.0) for name in (*STRATEGIES, "cumulative")}
    if "cumulative" not in found:
        balances["cumulative"] = sum(balances[name] for name in STRATEGIES)
    return balances


def _positions(db) -> list[dict[str, Any]]:
    open_rows = db.execute("SELECT strategy, payload FROM open_positions").fetchall()
    positions = []
    for strategy, payload in open_rows:
        try:
            raw = json.loads(payload)
        except json.JSONDecodeError:
            continue
        legs = raw.get("legs") or []
        described = " / ".join(
            f"{leg.get('side')} {leg.get('strike')} {leg.get('right')}" for leg in legs
        )
        positions.append({
            "strategy": str(strategy),
            "structure": raw.get("structure"),
            "direction": raw.get("direction"),
            "contracts": raw.get("contracts"),
            "entry_net": raw.get("entry_net"),
            "entry_spot": raw.get("entry_spot"),
            "entry_time": raw.get("entry_time"),
            "expiration": raw.get("expiration") or None,
            "legs": described,
        })
    return positions


def _trades(db, session_date: str) -> list[dict[str, Any]]:
    rows = db.execute(
        """
        SELECT ts, strategy, event, structure, contracts, spot, pnl,
               strategy_balance, cumulative_balance, detail
        FROM trades
        WHERE starts_with(ts, ?)
        ORDER BY ts
        """,
        [session_date],
    ).fetchall()
    trades = []
    for row in rows:
        trades.append({
            "time": _unix(row[0]),
            "ts": str(row[0]),
            "strategy": row[1],
            "event": row[2],
            "structure": row[3],
            "contracts": None if row[4] is None else int(row[4]),
            "spot": _num(row[5]),
            "pnl": _num(row[6]),
            "strategy_balance": _num(row[7]),
            "cumulative_balance": _num(row[8]),
            "detail": row[9] or "",
        })
    return trades


def _pnl_series(
    snapshots: list[tuple],
    trades: list[dict[str, Any]],
    balances: dict[str, float],
) -> list[dict[str, Any]]:
    if not snapshots:
        return []
    realized = {name: 0.0 for name in STRATEGIES}
    for trade in trades:
        if trade["pnl"] is None or trade["strategy"] not in realized:
            continue
        realized[trade["strategy"]] += float(trade["pnl"])
    running = {
        name: balances.get(name, 0.0) - realized[name]
        for name in STRATEGIES
    }
    running["cumulative"] = sum(running[name] for name in STRATEGIES)
    first = _unix(snapshots[0][0])
    last = _unix(snapshots[-1][0])
    points = [_pnl_point(first, running)]
    for trade in trades:
        if trade["pnl"] is None or trade["strategy"] not in STRATEGIES:
            continue
        if trade["strategy_balance"] is not None:
            running[trade["strategy"]] = float(trade["strategy_balance"])
        if trade["cumulative_balance"] is not None:
            running["cumulative"] = float(trade["cumulative_balance"])
        if trade["time"] == points[-1]["time"]:
            points[-1] = _pnl_point(trade["time"], running)
        else:
            points.append(_pnl_point(trade["time"], running))
    if last > points[-1]["time"]:
        points.append(_pnl_point(last, balances))
    elif len(points) == 1:
        points.append(_pnl_point(last + 60, balances))
    return points


def _pnl_point(moment: int, balances: dict[str, float]) -> dict[str, Any]:
    return {
        "time": moment,
        "absorption": round(float(balances.get("absorption", 0.0)), 2),
        "cascade": round(float(balances.get("cascade", 0.0)), 2),
        "macro": round(float(balances.get("macro", 0.0)), 2),
        "cumulative": round(float(balances.get("cumulative", 0.0)), 2),
    }


def _unix(value: str) -> int:
    return int(datetime.fromisoformat(str(value)).timestamp())


def _num(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return round(number, 6)


def _percent(value: Any) -> float | None:
    number = _num(value)
    if number is None:
        return None
    return round(number * 100.0, 4)
