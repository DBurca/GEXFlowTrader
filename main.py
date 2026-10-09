"""
Orchestrator for the Macro GEX Prediction & Regime Tool.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import config
import chart_engine
import data_loader
import gex_engine
import signal_engine


def run(
    ticker: str | None = None,
    *,
    save_chart: str | Path | None = None,
    show_chart: bool = True,
) -> dict:
    """
    End-to-end pipeline:
      1. Pull market data from Schwab
      2. Calculate GEX profile, zones, and regime
      3. Evaluate Gamma Flip signal
      4. Render chart with future-gap GEX overlays
    """
    config.validate_config()
    ticker = ticker or config.TICKER

    print(f"Loading market data for {ticker}...")
    market_data = data_loader.load_market_data(ticker=ticker)

    print("Calculating GEX profile and zones...")
    gex_summary = gex_engine.build_gex_summary(
        market_data["option_fields"],
        market_data["spot_price"],
        resolution=config.RESOLUTION,
    )

    print("Evaluating Gamma Flip signal...")
    signal_summary = signal_engine.build_signal_summary(
        market_data["ohlcv"],
        gex_summary["profile"],
        zones=gex_summary["zones"],
        require_confirmation=config.REQUIRE_CONFIRMATION,
    )

    _print_summary(
        ticker=ticker,
        spot_price=market_data["spot_price"],
        gex_summary=gex_summary,
        signal_summary=signal_summary,
    )

    print("Rendering chart...")
    chart_result = chart_engine.render_analysis_chart(
        market_data["ohlcv"],
        gex_summary["zones"],
        gamma_flip=signal_summary["gamma_flip"],
        ticker=ticker,
        save_path=save_chart,
        show=show_chart,
    )

    return {
        "market_data": market_data,
        "gex_summary": gex_summary,
        "signal_summary": signal_summary,
        "chart": chart_result,
    }


def _print_summary(
    *,
    ticker: str,
    spot_price: float,
    gex_summary: dict,
    signal_summary: dict,
) -> None:
    signal = signal_summary["signal"]
    gamma_flip = signal_summary["gamma_flip"]

    print("\n" + "=" * 60)
    print(f"  {ticker} Macro GEX Summary")
    print("=" * 60)
    print(f"  Spot Price:      {spot_price:,.2f}")
    print(f"  Total Net GEX:   {gex_summary['total_net_gex']:,.0f}")
    print(f"  Regime:          {gex_summary['regime']}")
    print(
        f"  Gamma Flip:      {gamma_flip:,.2f}"
        if gamma_flip is not None
        else "  Gamma Flip:      N/A"
    )
    print(f"  GEX Zones:       {len(gex_summary['zones'])}")
    print(f"  Resolution:      {config.RESOLUTION}")
    print(f"  Confirmation:    {config.REQUIRE_CONFIRMATION}")
    print("-" * 60)
    print(f"  Signal:          {signal.message}")
    print("=" * 60 + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Macro GEX Prediction & Regime Tool",
    )
    parser.add_argument(
        "--ticker",
        default=None,
        help=f"Ticker symbol (default: {config.TICKER})",
    )
    parser.add_argument(
        "--save-chart",
        default=None,
        help="Optional path to save the chart image (e.g. output.png)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the chart window (useful with --save-chart)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run(
        ticker=args.ticker,
        save_chart=args.save_chart,
        show_chart=not args.no_show,
    )


if __name__ == "__main__":
    main()
