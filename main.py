#!/usr/bin/env python3
"""
Bear Market Crypto Trading Bot (Bybit)
========================================

Designed for small deposit acceleration ($200 -> $1200) in bear market conditions.
Uses Bybit exchange with paper trading (virtual balance) on real market data.

Modes:
- paper (default): Virtual $200 balance, real Bybit prices, no real money at risk
- live: Real orders on Bybit (requires API keys and PAPER_TRADING=false in .env)
- backtest: Test on historical data
- status: Show current stats

Usage:
    python main.py paper             # Paper trade with virtual $200 on real data
    python main.py live              # Real trading (PAPER_TRADING=false required)
    python main.py backtest [days]   # Backtest on historical data
    python main.py status            # Show current status
    python main.py reset             # Reset virtual balance to $200
"""

import asyncio
import sys

import config
from bot.utils.logger import log


def run_paper():
    """Start paper trading — virtual balance on real Bybit data."""
    config.PAPER_TRADING = True
    from bot.engine import TradingEngine

    engine = TradingEngine()
    asyncio.run(engine.start())


def run_live():
    """Start live trading on Bybit."""
    if config.PAPER_TRADING:
        log.warning("PAPER_TRADING=true in .env — set to false for real trading")
        log.warning("Starting in paper mode instead...")
        run_paper()
        return

    if not config.API_KEY or not config.API_SECRET:
        log.error("API_KEY and API_SECRET required for live trading")
        sys.exit(1)

    from bot.engine import TradingEngine
    engine = TradingEngine()
    asyncio.run(engine.start())


def run_backtest(days: int = 14, pairs: list[str] | None = None):
    """Run backtest on historical Bybit data."""
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
    log.info(f"Mode: {'PAPER' if config.PAPER_TRADING else 'LIVE'}")
    log.info(f"Exchange: {config.EXCHANGE}")

    # Virtual balance
    paper_state = "data/paper_state.json"
    if os.path.exists(paper_state):
        with open(paper_state) as f:
            state = json.load(f)
        log.info(f"Virtual balance: ${state.get('virtual_balance', 0):.2f}")
        log.info(f"Last updated: {state.get('last_updated', 'N/A')}")

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
    else:
        log.info("No learning state found")


def reset_balance():
    """Reset virtual balance and trade history."""
    import os

    for path in ["data/paper_state.json", config.TRADE_LOG_FILE, config.LEARNING["learning_data_path"]]:
        if os.path.exists(path):
            os.remove(path)
            log.info(f"Removed {path}")

    log.info(f"Virtual balance reset to ${config.INITIAL_DEPOSIT:.2f}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("Commands:")
        print("  python main.py paper             Paper trade (virtual $200, real data)")
        print("  python main.py live              Live trade (requires API keys)")
        print("  python main.py backtest [days]    Backtest (default: 14 days)")
        print("  python main.py status             Show stats")
        print("  python main.py reset              Reset virtual balance")
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == "paper":
        run_paper()
    elif command == "live":
        run_live()
    elif command == "backtest":
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 14
        run_backtest(days=days)
    elif command == "status":
        show_status()
    elif command == "reset":
        reset_balance()
    else:
        print(f"Unknown command: {command}")
        print("Use: paper | live | backtest | status | reset")
        sys.exit(1)


if __name__ == "__main__":
    main()
