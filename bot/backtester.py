"""
Backtesting module — test strategies on historical data before going live.

Simulates the full trading flow:
- Multi-pair analysis
- Signal generation
- Risk management
- Self-learning cycles
- PnL tracking with fees
"""

import uuid
from datetime import datetime

import ccxt
import pandas as pd
import numpy as np

import config
from bot.indicators.technical import compute_all
from bot.learning.self_learner import SelfLearner
from bot.strategies.base import Signal, SignalType, StrategyName, Trade
from bot.strategies.bear_strategies import (
    BearBreakdown,
    BearMeanReversion,
    BearScalp,
    BearTrendFollow,
)
from bot.strategies.risk_manager import RiskManager
from bot.utils.logger import log


class Backtester:

    def __init__(
        self,
        pairs: list[str] | None = None,
        timeframe: str = "5m",
        days: int = 30,
        initial_balance: float = 200.0,
        fee_pct: float = 0.04,
    ):
        self.pairs = pairs or config.TRADING_PAIRS[:3]
        self.timeframe = timeframe
        self.days = days
        self.initial_balance = initial_balance
        self.fee_pct = fee_pct / 100

        self.risk_manager = RiskManager()
        self.risk_manager.balance = initial_balance
        self.learner = SelfLearner()
        self.strategies = [
            BearTrendFollow(weight=config.STRATEGY["trend_follow_weight"]),
            BearMeanReversion(weight=config.STRATEGY["mean_reversion_weight"]),
            BearBreakdown(weight=config.STRATEGY["breakout_weight"]),
            BearScalp(weight=config.STRATEGY["scalp_weight"]),
        ]

    def run(self) -> dict:
        """Run full backtest."""
        log.info("=" * 60)
        log.info("BACKTESTING BEAR MARKET STRATEGY")
        log.info(f"Pairs: {self.pairs}")
        log.info(f"Timeframe: {self.timeframe}")
        log.info(f"Period: {self.days} days")
        log.info(f"Initial balance: ${self.initial_balance}")
        log.info("=" * 60)

        # Fetch historical data
        datasets = self._fetch_historical_data()
        if not datasets:
            log.error("No data fetched")
            return {}

        # Find common date range
        min_len = min(len(df) for df in datasets.values())
        log.info(f"Backtesting on {min_len} candles per pair")

        # Process each candle in chronological order
        equity_curve = [self.initial_balance]
        trade_count = 0

        for i in range(60, min_len):  # need 60 candles warmup
            for symbol, full_df in datasets.items():
                df_slice = full_df.iloc[:i + 1].copy()

                try:
                    df = compute_all(df_slice, self.learner.get_indicator_params())
                except Exception:
                    continue

                current_price = df.iloc[-1]["close"]

                # 1. Check open positions for this symbol
                for trade in list(self.risk_manager.open_trades):
                    if trade.symbol != symbol:
                        continue

                    # Check SL/TP
                    candle_high = df.iloc[-1]["high"]
                    candle_low = df.iloc[-1]["low"]

                    hit_sl = False
                    hit_tp = False

                    if trade.signal_type == SignalType.SHORT:
                        hit_sl = candle_high >= trade.stop_loss
                        hit_tp = candle_low <= trade.take_profit
                    else:
                        hit_sl = candle_low <= trade.stop_loss
                        hit_tp = candle_high >= trade.take_profit

                    if hit_sl:
                        self._close_backtest_trade(trade, trade.stop_loss, "sl")
                    elif hit_tp:
                        self._close_backtest_trade(trade, trade.take_profit, "tp")
                    else:
                        # Check strategy exit
                        for strat in self.strategies:
                            if strat.name == trade.strategy:
                                should_exit, reason = strat.check_exit(df)
                                if should_exit:
                                    self._close_backtest_trade(trade, current_price, f"signal:{reason}")
                                break

                        # Update trailing stop
                        self.risk_manager.update_trailing_stop(trade, current_price)

                # 2. Generate new signals
                for strategy in self.strategies:
                    try:
                        signal = strategy.evaluate(df, symbol)
                        if not signal.is_actionable:
                            continue

                        valid, reason = self.risk_manager.validate_signal(signal)
                        if not valid:
                            continue

                        # Execute in backtest
                        quantity = self.risk_manager.calculate_position_size(signal)
                        if quantity <= 0:
                            continue

                        fee = quantity * current_price * self.fee_pct

                        trade = Trade(
                            trade_id=str(uuid.uuid4())[:8],
                            symbol=symbol,
                            signal_type=signal.signal_type,
                            strategy=signal.strategy,
                            entry_price=current_price,
                            stop_loss=signal.stop_loss,
                            take_profit=signal.take_profit,
                            quantity=quantity,
                            leverage=config.RISK["leverage"],
                            entry_time=str(df.index[-1]),
                            fees=fee,
                            indicators_at_entry=signal.indicators,
                        )
                        self.risk_manager.register_open(trade)
                        trade_count += 1
                    except Exception:
                        continue

            # Track equity
            equity_curve.append(self.risk_manager.balance)

            # Self-learning check
            total_closed = len(self.risk_manager.closed_trades)
            if self.learner.should_analyze(total_closed):
                report = self.learner.analyze(self.risk_manager.closed_trades)
                # Update strategy weights
                for strat in self.strategies:
                    strat.weight = self.learner.get_strategy_weight(strat.name.value)

        # Close remaining positions at last price
        for trade in list(self.risk_manager.open_trades):
            last_price = datasets[trade.symbol].iloc[-1]["close"]
            self._close_backtest_trade(trade, last_price, "end_of_backtest")

        # Generate report
        return self._generate_report(equity_curve)

    def _close_backtest_trade(self, trade: Trade, exit_price: float, reason: str):
        """Close a trade in backtest mode."""
        if trade.signal_type == SignalType.SHORT:
            pnl = (trade.entry_price - exit_price) / trade.entry_price * trade.quantity * trade.entry_price * trade.leverage
        else:
            pnl = (exit_price - trade.entry_price) / trade.entry_price * trade.quantity * trade.entry_price * trade.leverage

        pnl -= trade.fees

        trade.exit_price = exit_price
        trade.exit_time = datetime.utcnow().isoformat()
        trade.pnl = pnl
        trade.pnl_pct = pnl / (trade.quantity * trade.entry_price / trade.leverage) * 100
        trade.status = "closed"
        trade.close_reason = reason

        self.risk_manager.register_close(trade)

    def _fetch_historical_data(self) -> dict[str, pd.DataFrame]:
        """Fetch historical OHLCV data using synchronous ccxt."""
        exchange = ccxt.bybit({
            "enableRateLimit": True,
            "options": {"defaultType": "linear"},
        })

        datasets = {}
        limit = min(self.days * 288, 1000)  # 5m = 288 per day, max 1000

        for symbol in self.pairs:
            try:
                log.info(f"Fetching {symbol} historical data...")
                ohlcv = exchange.fetch_ohlcv(symbol, self.timeframe, limit=limit)
                df = pd.DataFrame(
                    ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.set_index("timestamp", inplace=True)
                datasets[symbol] = df
                log.info(f"  {symbol}: {len(df)} candles loaded")
            except Exception as e:
                log.warning(f"Failed to fetch {symbol}: {e}")

        return datasets

    def _generate_report(self, equity_curve: list[float]) -> dict:
        """Generate backtest report."""
        stats = self.risk_manager.get_stats()

        # Calculate additional metrics
        equity = np.array(equity_curve)
        returns = np.diff(equity) / equity[:-1]
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(365 * 288) if np.std(returns) > 0 else 0

        # Strategy breakdown
        strategy_stats = {}
        for strat_name in StrategyName:
            strades = [t for t in self.risk_manager.closed_trades if t.strategy == strat_name]
            if strades:
                wins = sum(1 for t in strades if t.pnl and t.pnl > 0)
                total_pnl = sum(t.pnl for t in strades if t.pnl)
                strategy_stats[strat_name.value] = {
                    "trades": len(strades),
                    "win_rate": wins / len(strades),
                    "total_pnl": total_pnl,
                }

        report = {
            "initial_balance": self.initial_balance,
            "final_balance": stats["balance"],
            "total_return_pct": stats["total_pnl_pct"],
            "total_trades": stats["total_trades"],
            "win_rate": stats["win_rate"],
            "profit_factor": stats["profit_factor"],
            "max_drawdown_pct": stats["max_drawdown_pct"],
            "sharpe_ratio": sharpe,
            "strategy_breakdown": strategy_stats,
            "self_learning_cycles": self.learner.analysis_count,
            "final_strategy_weights": dict(self.learner.strategy_weights),
        }

        # Print report
        log.info("=" * 60)
        log.info("BACKTEST RESULTS")
        log.info("=" * 60)
        log.info(f"Initial:   ${self.initial_balance:.2f}")
        log.info(f"Final:     ${stats['balance']:.2f}")
        log.info(f"Return:    {stats['total_pnl_pct']:+.1f}%")
        log.info(f"Trades:    {stats['total_trades']}")
        log.info(f"Win Rate:  {stats['win_rate']:.1%}")
        log.info(f"PF:        {stats['profit_factor']:.2f}")
        log.info(f"Max DD:    {stats['max_drawdown_pct']:.1f}%")
        log.info(f"Sharpe:    {sharpe:.2f}")
        log.info(f"Learning:  {self.learner.analysis_count} cycles")
        log.info("-" * 40)
        for name, s in strategy_stats.items():
            log.info(f"  {name}: {s['trades']} trades, WR={s['win_rate']:.1%}, PnL={s['total_pnl']:+.2f}")
        log.info("=" * 60)

        return report
