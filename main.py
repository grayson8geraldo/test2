#!/usr/bin/env python3
"""
Bear Market Crypto Trading Bot
===============================

Designed for small deposit acceleration ($200 → $1200) in bear market conditions.

Strategies:
- Trend Follow (Short): Ride the dominant downtrend
- Mean Reversion: Catch oversold bounces for quick longs
- Breakdown: Trade support breaks with volume confirmation
- Scalping: Quick in/out trades on micro-movements

Features:
- Multi-timeframe analysis (1m, 5m, 15m, 1h)
- 10+ technical indicators tuned for bear market
- Dynamic risk management with ATR-based stops
- Self-learning: auto-adjusts after every 20 trades
- Trailing stops with position scaling

Usage:
    python main.py live          # Run live trading bot
    python main.py backtest      # Run backtest on historical data
    python main.py status        # Show current status
"""

import asyncio
import sys

import config
from bot.utils.logger import log


def run_live():
    """Start live trading bot."""
    from bot.engine import TradingEngine

    engine = TradingEngine()
    asyncio.run(engine.start())


def run_backtest(days: int = 14, pairs: list[str] | None = None):
    """Run backtest on historical data."""
    from bot.backtester import Backtester

    bt = Backtester(
        pairs=pairs or config.TRADING_PAIRS[:3],
        timeframe=config.TIMEFRAMES["entry"],
        days=days,
        initial_balance=config.INITIAL_DEPOSIT,
    )
    report = bt.run()
    return report


def show_status():
    """Show current bot status and learning state."""
    import json
    import os

    log.info("=== BOT STATUS ===")

    # Trade history
    if os.path.exists(config.TRADE_LOG_FILE):
        with open(config.TRADE_LOG_FILE) as f:
            trades = json.load(f)
        total = len(trades)
        wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
        total_pnl = sum(t.get("pnl", 0) for t in trades)
        wr = wins / total if total > 0 else 0
        log.info(f"Total trades: {total}")
        log.info(f"Win rate: {wr:.1%}")
        log.info(f"Total PnL: ${total_pnl:+.2f}")
        log.info(f"Balance: ${config.INITIAL_DEPOSIT + total_pnl:.2f}")
    else:
        log.info("No trade history found")

    # Learning state
    learning_path = config.LEARNING["learning_data_path"]
    if os.path.exists(learning_path):
        with open(learning_path) as f:
            state = json.load(f)
        log.info(f"Learning cycles: {state.get('analysis_count', 0)}")
        log.info(f"Strategy weights: {state.get('strategy_weights', {})}")
        log.info(f"Indicator adjustments: {state.get('indicator_adjustments', {})}")
    else:
        log.info("No learning state found")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("Commands:")
        print("  python main.py live              Start live trading")
        print("  python main.py backtest [days]    Run backtest (default: 14 days)")
        print("  python main.py status             Show current status")
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == "live":
        run_live()
    elif command == "backtest":
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 14
        run_backtest(days=days)
    elif command == "status":
        show_status()
    else:
        print(f"Unknown command: {command}")
        print("Use: live | backtest | status")
        sys.exit(1)


if __name__ == "__main__":
    main()
