"""Read-only chart for the paper forward test.

Serves the JSON file the tester rewrites after each committed snapshot.
It does not open DuckDB and it does not call Schwab.
"""

from __future__ import annotations

import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from live_view import live_view_path

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SPY paper session</title>
<script src="https://cdn.jsdelivr.net/npm/lightweight-charts@4.2.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; background: #101418; color: #d7e0e8; font: 14px/1.4 ui-sans-serif, system-ui, sans-serif; }
  main { max-width: 1100px; margin: 0 auto; padding: 20px 16px 48px; }
  h1 { font-size: 22px; font-weight: 600; margin: 0 0 4px; }
  h2 { font-size: 15px; font-weight: 600; margin: 22px 0 6px; }
  .sub { color: #8ea0b0; margin: 0 0 14px; }
  .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 8px; margin: 12px 0 4px; }
  .stat { background: #181e25; border: 1px solid #2a3440; border-radius: 8px; padding: 8px 10px; }
  .stat b { display: block; font-size: 16px; font-weight: 600; }
  .stat span { color: #8ea0b0; font-size: 12px; }
  .chart { height: 640px; margin-top: 4px; }
  .chart.short { height: 200px; }
  .panel-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
  .panel-head h2 { margin-bottom: 0; }
  #fullscreen { background: #181e25; color: #d7e0e8; border: 1px solid #2a3440; border-radius: 6px; padding: 6px 10px; cursor: pointer; font: inherit; }
  #price-panel.fullscreen { position: fixed; inset: 0; z-index: 40; background: #101418; margin: 0; padding: 12px 16px 16px; display: flex; flex-direction: column; }
  #price-panel.fullscreen .chart { flex: 1 1 auto; height: auto; min-height: 0; }
  .legend { color: #8ea0b0; font-size: 12px; margin-top: 4px; }
  .legend span { margin-right: 10px; white-space: nowrap; }
  .legend i { display: inline-block; width: 14px; height: 3px; margin: 0 4px 2px 0; vertical-align: middle; }
  .legend i.dash, .legend i.dot { height: 0; background: transparent; }
  .legend i.dash { border-top: 2px dashed; }
  .legend i.dot { border-top: 2px dotted; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid #2a3440; }
  th { color: #8ea0b0; font-weight: 500; }
  .warn { color: #e6c07b; }
</style>
</head>
<body>
<main>
  <h1>SPY paper session</h1>
  <p class="sub" id="status">Loading the latest snapshot…</p>
  <section class="stats" id="stats"></section>
  <section id="price-panel">
  <div class="panel-head">
    <h2>Spot, VWAP, bands, and 0DTE walls</h2>
    <button type="button" id="fullscreen">Full screen</button>
  </div>
  <p class="sub">Dollars. Clock labels are Eastern. Candles are the session's one-minute history from Schwab, replaced each minute, with the latest spot written into the current bar. The dotted purple lines are the 1σ move still left in the session, spot × ATM IV × sqrt(minutes left / (390 × 252)), centered on VWAP. They move toward VWAP as the session runs out. A secondary wall is dashed in the wall's own color, and only drawn when its dollar gamma is at least 80% of the primary.</p>
  <div id="price" class="chart"></div>
  <div class="legend"><span><i style="background:#7dcea0"></i>Up</span><span><i style="background:#e06c75"></i>Down</span><span><i style="background:#8aa0b4"></i>VWAP</span><span><i class="dot" style="border-color:#c084fc"></i>1σ</span><span><i style="background:#e06c75"></i>Call wall</span><span><i class="dash" style="border-color:#e06c75"></i>Call 2</span><span><i style="background:#7dcea0"></i>Put wall</span><span><i class="dash" style="border-color:#7dcea0"></i>Put 2</span><span><i class="dot" style="border-color:#e06c75"></i>Open call</span><span><i class="dot" style="border-color:#7dcea0"></i>Open put</span><span><i class="dot" style="border-color:#d7e3ea"></i>Open PML</span><span><i style="background:#e6c07b"></i>Zero gamma</span></div>
  </section>
  <h2>Cumulative delta flow</h2>
  <p class="sub">Share-delta, in millions.</p>
  <div id="cdf" class="chart short"></div>
  <h2>At-the-money implied volatility</h2>
  <p class="sub">Percent.</p>
  <div id="iv" class="chart short"></div>
  <h2>Net GEX</h2>
  <p class="sub">Gamma × open interest × 100 × spot. Total adds the 0–1 DTE book to the 30–90 DTE book. Volume GEX uses session volume in place of open interest on the 0DTE book.</p>
  <div id="gex" class="chart short"></div>
  <div class="legend"><i style="background:#d7e3ea"></i>Total<i style="background:#6cb6ff"></i>0–1 DTE<i style="background:#c084fc"></i>Volume<i style="background:#e6c07b"></i>30–90 DTE</div>
  <h2>Paper balances</h2>
  <p class="sub">Realized profit versus each balance at the session open. The line steps when a position closes. A flat line at zero means no fills.</p>
  <div id="pnl" class="chart short"></div>
  <div class="legend"><i style="background:#6cb6ff"></i>Absorption<i style="background:#e06c75"></i>Cascade<i style="background:#e6c07b"></i>Macro<i style="background:#d7e3ea"></i>Cumulative</div>
  <h2>Open positions</h2>
  <div id="positions"></div>
  <h2>Trades this session</h2>
  <div id="trades"></div>
</main>
<script>
const ET = { timeZone: "America/New_York", hour: "2-digit", minute: "2-digit", hour12: false };
const charts = {};

function fmtTime(unix) {
  return new Intl.DateTimeFormat("en-US", ET).format(new Date(unix * 1000));
}
function money(value) {
  if (value == null || Number.isNaN(value)) return "—";
  return value.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });
}
function gexText(value) {
  if (value == null || Number.isNaN(value)) return "—";
  const abs = Math.abs(value);
  const sign = value < 0 ? "-" : "";
  if (abs >= 1e9) return sign + (abs / 1e9).toFixed(2) + "B";
  if (abs >= 1e6) return sign + (abs / 1e6).toFixed(1) + "M";
  return sign + abs.toFixed(0);
}
function num(value, digits) {
  if (value == null || Number.isNaN(value)) return "—";
  return Number(value).toLocaleString("en-US", { maximumFractionDigits: digits, minimumFractionDigits: digits });
}
function lineData(rows, key, scale) {
  return rows.map((row) => {
    if (row[key] == null) return { time: row.time };
    const value = scale ? row[key] * scale : row[key];
    return { time: row.time, value: value };
  });
}
function profitData(rows, key) {
  const first = rows.find((row) => row[key] != null);
  const start = first ? first[key] : 0;
  return rows.map((row) => {
    if (row[key] == null) return { time: row.time };
    return { time: row.time, value: row[key] - start };
  });
}
function chartOptions(formatter) {
  return {
    autoSize: true,
    layout: { background: { color: "#101418" }, textColor: "#c5d0da" },
    grid: { vertLines: { color: "#1c242d" }, horzLines: { color: "#1c242d" } },
    rightPriceScale: { borderColor: "#2a3440" },
    timeScale: {
      borderColor: "#2a3440",
      timeVisible: true,
      secondsVisible: false,
      tickMarkFormatter: (time) => fmtTime(time),
    },
    localization: {
      timeFormatter: fmtTime,
      priceFormatter: formatter || ((value) => num(value, 2)),
    },
    crosshair: { mode: 0 },
  };
}
function ensureChart(id, options) {
  if (charts[id]) return charts[id];
  const chart = LightweightCharts.createChart(document.getElementById(id), options);
  charts[id] = { chart, series: {} };
  return charts[id];
}
function series(bundle, name, options) {
  if (!bundle.series[name]) bundle.series[name] = bundle.chart.addLineSeries(options);
  return bundle.series[name];
}
function candleSeries(bundle) {
  if (!bundle.series.candle) {
    bundle.series.candle = bundle.chart.addCandlestickSeries({
      upColor: "#7dcea0",
      downColor: "#e06c75",
      borderUpColor: "#7dcea0",
      borderDownColor: "#e06c75",
      wickUpColor: "#7dcea0",
      wickDownColor: "#e06c75",
      priceLineVisible: false,
    });
  }
  return bundle.series.candle;
}
function candleData(rows) {
  return rows.map((row) => {
    const open = row.open != null ? row.open : row.spot;
    if (open == null) return { time: row.time };
    const high = row.high != null ? row.high : open;
    const low = row.low != null ? row.low : open;
    const close = row.close != null ? row.close : open;
    return { time: row.time, open: open, high: high, low: low, close: close };
  });
}
function setLine(bundle, name, options, rows, key) {
  const points = lineData(rows, key);
  if (!points.some((point) => point.value != null)) return;
  series(bundle, name, options).setData(points);
}
function render(data) {
  const latest = data.latest || {};
  const when = data.updated
    ? new Intl.DateTimeFormat("en-US", { ...ET, month: "short", day: "numeric", second: "2-digit" }).format(new Date(data.updated))
    : "no snapshot yet";
  document.getElementById("status").textContent = data.session_date
    ? "Session " + data.session_date + " · last write " + when + " ET · refreshes every minute"
    : "Waiting for the forward tester to publish a snapshot.";
  const balances = data.balances || {};
  const cells = [
    ["Spot", num(latest.spot, 2)],
    ["VWAP", num(latest.vwap, 2)],
    ["Call wall", num(latest.call_wall, 2)],
    ["Put wall", num(latest.put_wall, 2)],
    ["Call wall 2", num(latest.call_wall_2, 2)],
    ["Put wall 2", num(latest.put_wall_2, 2)],
    ["0–1 DTE GEX", gexText(latest.gex_0_1)],
    ["Total net GEX", gexText(latest.total_net_gex)],
    ["Volume GEX", gexText(latest.gex_volume)],
    ["1% hedge", gexText(latest.hedge_1pct)],
    ["Charm to close", gexText(latest.charm_hedge)],
    ["30–90 DTE GEX", gexText(latest.macro_gex)],
    ["Zero gamma", num(latest.zero_gamma, 2)],
    ["1σ high", num(latest.upper_1, 2)],
    ["1σ low", num(latest.lower_1, 2)],
    ["ATM IV", latest.atm_iv == null ? "—" : num(latest.atm_iv, 2) + "%"],
    ["HMM", latest.hmm == null ? "—" : String(latest.hmm)],
    ["Macro flip", num(latest.gamma_flip, 2)],
    ["Absorption", money(balances.absorption)],
    ["Cascade", money(balances.cascade)],
    ["Macro", money(balances.macro)],
    ["Cumulative", money(balances.cumulative)],
  ];
  document.getElementById("stats").innerHTML = cells.map(([label, value]) =>
    "<div class='stat'><b>" + value + "</b><span>" + label + "</span></div>"
  ).join("");

  const price = ensureChart("price", chartOptions());
  const candles = (data.candles && data.candles.length) ? data.candles : candleData(data.price || []);
  candleSeries(price).setData(candles);
  series(price, "vwap", { color: "#8aa0b4", lineWidth: 1, title: "VWAP", priceLineVisible: false }).setData(lineData(data.price || [], "vwap"));
  const quiet = { lineWidth: 1, priceLineVisible: false, lastValueVisible: false, title: "" };
  series(price, "u1", { ...quiet, color: "#c084fc", lineStyle: LightweightCharts.LineStyle.Dotted }).setData(lineData(data.price || [], "upper_1"));
  series(price, "l1", { ...quiet, color: "#c084fc", lineStyle: LightweightCharts.LineStyle.Dotted }).setData(lineData(data.price || [], "lower_1"));
  series(price, "call", { color: "#e06c75", lineWidth: 2, title: "Call wall", priceLineVisible: false }).setData(lineData(data.price || [], "call_wall"));
  setLine(price, "call2", { ...quiet, color: "#e06c75", lineStyle: LightweightCharts.LineStyle.Dashed }, data.price || [], "call_wall_2");
  series(price, "put", { color: "#7dcea0", lineWidth: 2, title: "Put wall", priceLineVisible: false }).setData(lineData(data.price || [], "put_wall"));
  setLine(price, "put2", { ...quiet, color: "#7dcea0", lineStyle: LightweightCharts.LineStyle.Dashed }, data.price || [], "put_wall_2");
  series(price, "zero", { color: "#e6c07b", lineWidth: 1, title: "Zero gamma", priceLineVisible: false }).setData(lineData(data.price || [], "zero_gamma"));
  setLine(price, "openCall", { ...quiet, color: "#e06c75", lineStyle: LightweightCharts.LineStyle.Dotted }, data.price || [], "static_open_call_wall_1");
  setLine(price, "openPut", { ...quiet, color: "#7dcea0", lineStyle: LightweightCharts.LineStyle.Dotted }, data.price || [], "static_open_put_wall_1");
  setLine(price, "openPml", { ...quiet, color: "#d7e3ea", lineStyle: LightweightCharts.LineStyle.Dotted }, data.price || [], "static_open_pml");
  const markers = (data.trades || []).map((trade) => ({
    time: data.candles && data.candles.length ? Math.floor(trade.time / 60) * 60 : trade.time,
    position: trade.event === "OPEN" ? "belowBar" : "aboveBar",
    color: trade.event === "STOP" ? "#e06c75" : trade.event === "OPEN" ? "#7dcea0" : "#6cb6ff",
    shape: trade.event === "OPEN" ? "arrowUp" : "arrowDown",
    text: trade.strategy + " " + trade.event,
  }));
  price.series.candle.setMarkers(markers);
  const scaleValues = [];
  for (const row of data.price || []) {
    for (const key of ["high", "low", "spot", "upper_1", "lower_1", "static_open_call_wall_1", "static_open_put_wall_1", "static_open_pml"]) {
      if (row[key] != null) scaleValues.push(row[key]);
    }
  }
  for (const row of candles) {
    if (row.high != null) scaleValues.push(row.high);
    if (row.low != null) scaleValues.push(row.low);
  }
  if (scaleValues.length) {
    const low = Math.min(...scaleValues);
    const high = Math.max(...scaleValues);
    const pad = 0.15;
    const provider = () => ({ priceRange: { minValue: low - pad, maxValue: high + pad } });
    for (const name of ["candle", "vwap", "u1", "l1", "call", "call2", "put", "put2", "zero"]) {
      if (price.series[name]) price.series[name].applyOptions({ autoscaleInfoProvider: provider });
    }
  }
  price.chart.timeScale().fitContent();

  const cdf = ensureChart("cdf", chartOptions((value) => num(value, 2) + "M"));
  series(cdf, "cdf", { color: "#7dcea0", lineWidth: 2, title: "CDF", priceLineVisible: false }).setData(lineData(data.flow || [], "cdf", 1 / 1e6));
  cdf.chart.timeScale().fitContent();

  const gex = ensureChart("gex", chartOptions(gexText));
  setLine(gex, "total", { color: "#d7e3ea", lineWidth: 2, title: "Total", priceLineVisible: false }, data.gex || [], "total");
  setLine(gex, "near", { color: "#6cb6ff", lineWidth: 2, title: "0–1 DTE", priceLineVisible: false }, data.gex || [], "near");
  setLine(gex, "volume", { color: "#c084fc", lineWidth: 2, title: "Volume", priceLineVisible: false }, data.gex || [], "volume");
  setLine(gex, "macro", { color: "#e6c07b", lineWidth: 2, title: "30–90 DTE", priceLineVisible: false }, data.gex || [], "macro");
  gex.chart.timeScale().fitContent();

  const iv = ensureChart("iv", chartOptions((value) => num(value, 2) + "%"));
  series(iv, "iv", { color: "#6cb6ff", lineWidth: 2, title: "ATM IV", priceLineVisible: false }).setData(lineData(data.flow || [], "atm_iv"));
  iv.chart.timeScale().fitContent();

  const pnl = ensureChart("pnl", chartOptions((value) => money(value)));
  const step = LightweightCharts.LineType.WithSteps;
  series(pnl, "absorption", { color: "#6cb6ff", lineWidth: 2, title: "Absorption", lineType: step, priceLineVisible: false }).setData(profitData(data.pnl || [], "absorption"));
  series(pnl, "cascade", { color: "#e06c75", lineWidth: 2, title: "Cascade", lineType: step, priceLineVisible: false }).setData(profitData(data.pnl || [], "cascade"));
  series(pnl, "macro", { color: "#e6c07b", lineWidth: 2, title: "Macro", lineType: step, priceLineVisible: false }).setData(profitData(data.pnl || [], "macro"));
  series(pnl, "cumulative", { color: "#d7e3ea", lineWidth: 2, title: "Cumulative", lineType: step, priceLineVisible: false }).setData(profitData(data.pnl || [], "cumulative"));
  pnl.chart.timeScale().fitContent();

  const positions = data.positions || [];
  document.getElementById("positions").innerHTML = positions.length
    ? "<table><tr><th>Strategy</th><th>Contracts</th><th>Structure</th><th>Entry net</th><th>Entry spot</th></tr>"
      + positions.map((row) => "<tr><td>" + row.strategy + "</td><td>" + row.contracts + "</td><td>"
        + (row.legs || row.structure || "") + "</td><td>" + num(row.entry_net, 2) + "</td><td>"
        + num(row.entry_spot, 2) + "</td></tr>").join("") + "</table>"
    : "<p class='sub'>No open paper positions.</p>";

  const trades = data.trades || [];
  document.getElementById("trades").innerHTML = trades.length
    ? "<table><tr><th>ET</th><th>Strategy</th><th>Event</th><th>PnL</th><th>Detail</th></tr>"
      + trades.map((row) => "<tr><td>" + fmtTime(row.time) + "</td><td>" + row.strategy + "</td><td>"
        + row.event + "</td><td>" + (row.pnl == null ? "—" : money(row.pnl)) + "</td><td>"
        + (row.detail || "") + "</td></tr>").join("") + "</table>"
    : "<p class='sub'>No paper trades this session.</p>";
}

async function refresh(retry) {
  try {
    const response = await fetch("/api/session", { cache: "no-store" });
    if (!response.ok) throw new Error("HTTP " + response.status);
    render(await response.json());
  } catch (error) {
    document.getElementById("status").textContent = "Session file busy. Retrying. " + error.message;
    if (!retry) setTimeout(() => refresh(true), 2000);
  }
}
const pricePanel = document.getElementById("price-panel");
const fullButton = document.getElementById("fullscreen");
function toggleFull(force) {
  const on = force == null ? !pricePanel.classList.contains("fullscreen") : force;
  pricePanel.classList.toggle("fullscreen", on);
  fullButton.textContent = on ? "Exit" : "Full screen";
  const box = document.getElementById("price");
  const chart = charts.price && charts.price.chart;
  if (chart) requestAnimationFrame(() => chart.resize(box.clientWidth, box.clientHeight));
}
fullButton.addEventListener("click", () => toggleFull());
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && pricePanel.classList.contains("fullscreen")) toggleFull(false);
});
refresh();
setInterval(refresh, 60000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            body = PAGE.encode("utf-8")
            self._send(200, "text/html; charset=utf-8", body)
            return
        if path == "/api/session":
            self._send_session()
            return
        if path == "/health":
            self._send(200, "text/plain; charset=utf-8", b"ok")
            return
        self._send(404, "text/plain; charset=utf-8", b"not found")

    def _send_session(self) -> None:
        path = live_view_path()
        empty = b'{"updated":null,"session_date":null,"symbol":"SPY","latest":null,"balances":{},"positions":[],"price":[],"flow":[],"pnl":[],"trades":[]}'
        body = b""
        for attempt in range(2):
            try:
                with open(path, "rb") as handle:
                    body = handle.read()
            except FileNotFoundError:
                body = empty
                break
            if body.strip():
                break
            if attempt == 0:
                time.sleep(0.05)
        self._send(200, "application/json", body or empty)

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        return


def main() -> None:
    host = os.environ.get("DASHBOARD_HOST", "0.0.0.0")
    port = int(os.environ.get("DASHBOARD_PORT", "8080"))
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"SPY dashboard on http://{host}:{port} reading {live_view_path()}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
