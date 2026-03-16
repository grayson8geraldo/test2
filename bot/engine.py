"""
Main Trading Engine — orchestrates all components.

Flow:
1. Fetch market data for all pairs and timeframes
2. Compute indicators
3. Evaluate all strategies, collect signals
4. Filter signals through risk manager
5. Execute best signals
6. Monitor open positions (SL/TP/trailing)
7. Trigger self-learning after every 20 closed trades
8. Repeat
"""

import asyncio
import uuid
from datetime import datetime

import pandas as pd

import config
from bot.exchange.connector import ExchangeConnector
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


class TradingEngine:

    def __init__(self):
        self.exchange = ExchangeConnector()
        self.risk_manager = RiskManager()
        self.learner = SelfLearner()
        self.running = False
        self._last_learning_count = 0

        # Initialize strategies with learned weights
        self.strategies = self._init_strategies()

    def _init_strategies(self):
        """Initialize strategies with current learned weights."""
        return [
            BearTrendFollow(weight=self.learner.get_strategy_weight(StrategyName.TREND_FOLLOW.value)),
            BearMeanReversion(weight=self.learner.get_strategy_weight(StrategyName.MEAN_REVERSION.value)),
            BearBreakdown(weight=self.learner.get_strategy_weight(StrategyName.BREAKOUT.value)),
            BearScalp(weight=self.learner.get_strategy_weight(StrategyName.SCALP.value)),
        ]

    async def start(self):
        """Start the trading bot main loop."""
        log.info("=" * 60)
        log.info("BEAR MARKET CRYPTO TRADING BOT STARTING")
        log.info(f"Initial deposit: ${config.INITIAL_DEPOSIT}")
        log.info(f"Target: ${config.TARGET_BALANCE}")
        log.info(f"Pairs: {config.TRADING_PAIRS}")
        log.info(f"Leverage: {config.RISK['leverage']}x")
        log.info("=" * 60)

        await self.exchange.connect()

        # Set leverage for all pairs
        for pair in config.TRADING_PAIRS:
            await self.exchange.set_leverage(pair, config.RISK["leverage"])

        # Load previous state
        self.risk_manager.load_trades()
        stats = self.risk_manager.get_stats()
        log.info(f"Loaded state: Balance=${stats['balance']:.2f}, Trades={stats['total_trades']}")

        self.running = True

        try:
            while self.running:
                await self._trading_cycle()
                await self._check_target()
                await asyncio.sleep(30)  # 30s between cycles
        except KeyboardInterrupt:
            log.info("Bot stopped by user")
        except Exception as e:
            log.error(f"Bot error: {e}", exc_info=True)
        finally:
            await self._shutdown()

    async def stop(self):
        """Stop the bot gracefully."""
        self.running = False

    async def _trading_cycle(self):
        """Execute one full trading cycle."""
        try:
            # 1. Monitor open positions first
            await self._monitor_positions()

            # 2. Check if self-learning should trigger
            self._check_learning()

            # 3. Scan for new opportunities
            all_signals = []
            for pair in config.TRADING_PAIRS:
                # Check pair score from learner
                pair_score = self.learner.get_pair_score(pair)
                if pair_score < 0.3:
                    continue  # Skip underperforming pairs

                signals = await self._analyze_pair(pair)
                all_signals.extend(signals)

            # 4. Rank and filter signals
            actionable = [s for s in all_signals if s.is_actionable]
            actionable.sort(key=lambda s: s.confidence * s.risk_reward, reverse=True)

            # 5. Execute top signals
            for signal in actionable:
                valid, reason = self.risk_manager.validate_signal(signal)
                if valid:
                    await self._execute_signal(signal)

        except Exception as e:
            log.error(f"Trading cycle error: {e}", exc_info=True)

    async def _analyze_pair(self, symbol: str) -> list[Signal]:
        """Analyze a single trading pair across all strategies."""
        signals = []

        try:
            # Fetch multi-timeframe data
            data = await self.exchange.fetch_multi_timeframe(
                symbol, config.TIMEFRAMES
            )

            if "entry" not in data or len(data["entry"]) < 60:
                return signals

            # Compute indicators on entry timeframe with learned adjustments
            indicator_params = self.learner.get_indicator_params()
            df = compute_all(data["entry"], params=indicator_params)

            # Add context from higher timeframe
            if "context" in data and len(data["context"]) > 20:
                df_context = compute_all(data["context"], params=indicator_params)
                last_ctx = df_context.iloc[-1]
                # Market regime from 1h
                df["htf_bear"] = last_ctx.get("ema_bear_aligned", 0)
                df["htf_adx"] = last_ctx.get("adx", 0)

            # Evaluate each strategy
            for strategy in self.strategies:
                try:
                    signal = strategy.evaluate(df, symbol)
                    if signal.is_actionable:
                        signals.append(signal)
                except Exception as e:
                    log.warning(f"Strategy {strategy.name} error on {symbol}: {e}")

        except Exception as e:
            log.warning(f"Analysis error for {symbol}: {e}")

        return signals

    async def _execute_signal(self, signal: Signal):
        """Execute a trading signal."""
        try:
            # Calculate position size
            quantity = self.risk_manager.calculate_position_size(signal)
            if quantity <= 0:
                return

            # Determine order side
            side = "sell" if signal.signal_type == SignalType.SHORT else "buy"

            # Place market order
            order = await self.exchange.create_market_order(
                signal.symbol, side, quantity
            )

            fill_price = order.get("average", signal.entry_price)
            fees = order.get("fee", {}).get("cost", 0) or 0

            # Create trade record
            trade = Trade(
                trade_id=str(uuid.uuid4())[:8],
                symbol=signal.symbol,
                signal_type=signal.signal_type,
                strategy=signal.strategy,
                entry_price=fill_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                quantity=quantity,
                leverage=config.RISK["leverage"],
                entry_time=datetime.utcnow().isoformat(),
                fees=fees,
                indicators_at_entry=signal.indicators,
            )

            self.risk_manager.register_open(trade)

            # Place SL/TP orders
            sl_side = "buy" if signal.signal_type == SignalType.SHORT else "sell"
            try:
                await self.exchange.create_stop_loss_order(
                    signal.symbol, sl_side, quantity, signal.stop_loss
                )
                await self.exchange.create_take_profit_order(
                    signal.symbol, sl_side, quantity, signal.take_profit
                )
            except Exception as e:
                log.warning(f"Failed to place SL/TP orders: {e}")

        except Exception as e:
            log.error(f"Execution error: {e}", exc_info=True)

    async def _monitor_positions(self):
        """Monitor open positions for exits."""
        for trade in list(self.risk_manager.open_trades):
            try:
                ticker = await self.exchange.fetch_ticker(trade.symbol)
                current_price = ticker["last"]

                # Check stop loss
                if self.risk_manager.check_stop_loss(trade, current_price):
                    await self._close_trade(trade, current_price, "sl")
                    continue

                # Check take profit
                if self.risk_manager.check_take_profit(trade, current_price):
                    await self._close_trade(trade, current_price, "tp")
                    continue

                # Update trailing stop
                self.risk_manager.update_trailing_stop(trade, current_price)

                # Check strategy-based exit
                data = await self.exchange.fetch_ohlcv(trade.symbol, config.TIMEFRAMES["entry"])
                if len(data) > 30:
                    df = compute_all(data, self.learner.get_indicator_params())
                    for strategy in self.strategies:
                        if strategy.name == trade.strategy:
                            should_exit, reason = strategy.check_exit(df)
                            if should_exit:
                                await self._close_trade(trade, current_price, f"signal:{reason}")
                                break

            except Exception as e:
                log.warning(f"Error monitoring {trade.symbol}: {e}")

    async def _close_trade(self, trade: Trade, exit_price: float, reason: str):
        """Close a trade and record results."""
        try:
            # Place close order
            side = "sell" if trade.signal_type == SignalType.SHORT else "buy"
            await self.exchange.close_position(trade.symbol, side, trade.quantity)
            await self.exchange.cancel_all_orders(trade.symbol)
        except Exception as e:
            log.error(f"Error closing position for {trade.symbol}: {e}")

        # Calculate PnL
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
        self.risk_manager.save_trades()

    def _check_learning(self):
        """Check if self-learning should trigger."""
        total = len(self.risk_manager.closed_trades)
        if self.learner.should_analyze(total):
            report = self.learner.analyze(self.risk_manager.closed_trades)

            # Reinitialize strategies with updated weights
            self.strategies = self._init_strategies()

            # Apply risk adjustments
            risk_adj = self.learner.get_risk_params()
            for key, value in risk_adj.items():
                if key in config.RISK:
                    config.RISK[key] = value

            log.info("Strategies and risk params updated from self-learning")

    async def _check_target(self):
        """Check if target balance has been reached."""
        stats = self.risk_manager.get_stats()
        balance = stats["balance"]

        if balance >= config.TARGET_BALANCE:
            log.info(f"TARGET REACHED! Balance: ${balance:.2f} >= ${config.TARGET_BALANCE}")
            log.info("Switching to conservative mode...")
            # Reduce risk on target hit
            config.RISK["max_risk_per_trade_pct"] = 1.0
            config.RISK["leverage"] = 5

        # Log periodic status
        progress = (balance - config.INITIAL_DEPOSIT) / (config.TARGET_BALANCE - config.INITIAL_DEPOSIT) * 100
        log.info(
            f"Status: Balance=${balance:.2f} | "
            f"Progress={progress:.1f}% | "
            f"Trades={stats['total_trades']} | "
            f"WR={stats['win_rate']:.1%} | "
            f"DD={stats['max_drawdown_pct']:.1f}%"
        )

    async def _shutdown(self):
        """Clean shutdown."""
        log.info("Shutting down...")
        self.risk_manager.save_trades()

        # Close all open positions
        for trade in list(self.risk_manager.open_trades):
            try:
                ticker = await self.exchange.fetch_ticker(trade.symbol)
                await self._close_trade(trade, ticker["last"], "shutdown")
            except Exception as e:
                log.error(f"Error closing {trade.symbol} on shutdown: {e}")

        await self.exchange.close()
        log.info("Bot shutdown complete")
