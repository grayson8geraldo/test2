"""
Main Trading Engine — orchestrates all components.

Supports two modes:
- Paper trading (default): Virtual balance on real market data from Bybit
- Live trading: Real orders on Bybit

Flow:
1. Fetch real market data for all pairs and timeframes
2. Compute indicators
3. Evaluate all strategies, collect signals
4. Filter signals through risk manager
5. Execute best signals (paper or live)
6. Monitor open positions (SL/TP/trailing)
7. Trigger self-learning after every 20 closed trades
8. Repeat
"""

import asyncio
import uuid
from datetime import datetime

import pandas as pd

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
from bot.strategies.risk_manager import RiskManager, _fmt_price
from bot.utils.logger import log


class TradingEngine:

    def __init__(self):
        # Choose connector based on mode
        if config.PAPER_TRADING:
            from bot.exchange.paper_trader import PaperTrader
            self.exchange = PaperTrader()
            self.paper_mode = True
        else:
            from bot.exchange.connector import ExchangeConnector
            self.exchange = ExchangeConnector()
            self.paper_mode = False

        self.risk_manager = RiskManager()
        self.learner = SelfLearner()
        self.running = False

        # Cache for indicator dataframes (avoid re-fetching in same cycle)
        self._cycle_data: dict[str, pd.DataFrame] = {}

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
        mode = "PAPER TRADING" if self.paper_mode else "LIVE TRADING"
        log.info("=" * 60)
        log.info(f"BEAR MARKET CRYPTO BOT — {mode}")
        log.info(f"Exchange: {config.EXCHANGE}")
        log.info(f"Initial deposit: ${config.INITIAL_DEPOSIT}")
        log.info(f"Target: ${config.TARGET_BALANCE}")
        log.info(f"Pairs: {config.TRADING_PAIRS}")
        log.info(f"Leverage: {config.RISK['leverage']}x")
        if self.paper_mode:
            log.info(f"Paper fee: {config.PAPER_FEE_PCT}% | Slippage: {config.PAPER_SLIPPAGE_PCT}%")
        log.info("=" * 60)

        await self.exchange.connect()

        # Set leverage for all pairs
        for pair in config.TRADING_PAIRS:
            await self.exchange.set_leverage(pair, config.RISK["leverage"])

        # Load previous state
        self.risk_manager.load_trades()

        # Sync balance from paper trader if in paper mode
        if self.paper_mode:
            self.risk_manager.balance = self.exchange.get_virtual_balance()

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
            # Clear data cache for this cycle
            self._cycle_data.clear()

            # 1. Fetch data for all pairs first (used by both monitoring and signals)
            for pair in config.TRADING_PAIRS:
                try:
                    df = await self.exchange.fetch_ohlcv(pair, config.TIMEFRAMES["entry"])
                    if len(df) >= 60:
                        self._cycle_data[pair] = compute_all(df, self.learner.get_indicator_params())
                except Exception as e:
                    log.warning(f"Data fetch failed for {pair}: {e}")

            # 2. Monitor open positions
            await self._monitor_positions()

            # 3. Check if self-learning should trigger
            self._check_learning()

            # 4. Scan for new opportunities
            all_signals = []
            for pair in config.TRADING_PAIRS:
                pair_score = self.learner.get_pair_score(pair)
                if pair_score < 0.3:
                    continue

                signals = await self._analyze_pair(pair)
                all_signals.extend(signals)

            # 5. Rank and filter signals
            actionable = [s for s in all_signals if s.is_actionable]
            actionable.sort(key=lambda s: s.confidence * s.risk_reward, reverse=True)

            # 6. Execute top signals
            for signal in actionable:
                valid, reason = self.risk_manager.validate_signal(signal)
                if valid:
                    await self._execute_signal(signal)
                    log.info(f"  Signal accepted: {signal.signal_type.value} {signal.symbol} conf={signal.confidence:.0f} RR={signal.risk_reward:.1f}")
                elif reason not in ("Already in " + signal.symbol, "Signal not actionable"):
                    log.debug(f"  Signal rejected: {signal.symbol} — {reason}")

        except Exception as e:
            log.error(f"Trading cycle error: {e}", exc_info=True)

    async def _analyze_pair(self, symbol: str) -> list[Signal]:
        """Analyze a single trading pair across all strategies."""
        signals = []

        try:
            # Use cached data if available
            df = self._cycle_data.get(symbol)
            if df is None:
                data = await self.exchange.fetch_ohlcv(symbol, config.TIMEFRAMES["entry"])
                if len(data) < 60:
                    return signals
                df = compute_all(data, self.learner.get_indicator_params())
                self._cycle_data[symbol] = df

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
        """Execute a trading signal (paper or live)."""
        try:
            quantity = self.risk_manager.calculate_position_size(signal)
            if quantity <= 0:
                return

            side = "sell" if signal.signal_type == SignalType.SHORT else "buy"

            order = await self.exchange.create_market_order(
                signal.symbol, side, quantity
            )

            fill_price = order.get("average", signal.entry_price)
            fees = order.get("fee", {}).get("cost", 0) or 0

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
        if not self.risk_manager.open_trades:
            return

        for trade in list(self.risk_manager.open_trades):
            try:
                ticker = await self.exchange.fetch_ticker(trade.symbol)
                current_price = ticker["last"]

                # Calculate unrealized PnL (leverage already embedded in position size)
                if trade.signal_type == SignalType.SHORT:
                    unrealized = (trade.entry_price - current_price) * trade.quantity
                else:
                    unrealized = (current_price - trade.entry_price) * trade.quantity

                margin = trade.quantity * trade.entry_price / trade.leverage
                pnl_pct = unrealized / margin * 100

                log.info(
                    f"  POS {trade.signal_type.value} {trade.symbol} "
                    f"entry={_fmt_price(trade.entry_price)} now={_fmt_price(current_price)} "
                    f"uPnL={unrealized:+.2f} ({pnl_pct:+.1f}%) "
                    f"SL={_fmt_price(trade.stop_loss)} TP={_fmt_price(trade.take_profit)}"
                )

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

                # Check strategy-based exit using cached data
                df = self._cycle_data.get(trade.symbol)
                if df is not None and len(df) > 30:
                    for strategy in self.strategies:
                        if strategy.name == trade.strategy:
                            should_exit, reason = strategy.check_exit(df)
                            if should_exit:
                                await self._close_trade(trade, current_price, f"signal:{reason}")
                                break

            except Exception as e:
                log.error(f"Error monitoring {trade.symbol}: {e}", exc_info=True)

    async def _close_trade(self, trade: Trade, exit_price: float, reason: str):
        """Close a trade and record results."""
        try:
            side = "sell" if trade.signal_type == SignalType.SHORT else "buy"
            await self.exchange.close_position(trade.symbol, side, trade.quantity)
            await self.exchange.cancel_all_orders(trade.symbol)
        except Exception as e:
            log.error(f"Error closing position for {trade.symbol}: {e}")

        # Calculate PnL (leverage already embedded in position size via qty)
        if trade.signal_type == SignalType.SHORT:
            pnl = (trade.entry_price - exit_price) * trade.quantity
        else:
            pnl = (exit_price - trade.entry_price) * trade.quantity

        pnl -= trade.fees

        trade.exit_price = exit_price
        trade.exit_time = datetime.utcnow().isoformat()
        trade.pnl = pnl
        margin = trade.quantity * trade.entry_price / trade.leverage
        trade.pnl_pct = pnl / margin * 100
        trade.status = "closed"
        trade.close_reason = reason

        self.risk_manager.register_close(trade)
        self.risk_manager.save_trades()

        # Sync virtual balance in paper mode
        if self.paper_mode:
            self.exchange.update_balance(pnl)

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
            config.RISK["max_risk_per_trade_pct"] = 1.0
            config.RISK["leverage"] = 5

        # Log periodic status
        progress = (balance - config.INITIAL_DEPOSIT) / (config.TARGET_BALANCE - config.INITIAL_DEPOSIT) * 100
        mode_tag = "[PAPER]" if self.paper_mode else "[LIVE]"
        open_count = len(self.risk_manager.open_trades)
        open_symbols = ", ".join(t.symbol for t in self.risk_manager.open_trades) or "none"
        log.info(
            f"{mode_tag} Balance=${balance:.2f} | "
            f"Progress={progress:.1f}% | "
            f"Trades={stats['total_trades']} | "
            f"WR={stats['win_rate']:.1%} | "
            f"DD={stats['max_drawdown_pct']:.1f}% | "
            f"Open={open_count} ({open_symbols})"
        )

    async def _shutdown(self):
        """Clean shutdown."""
        log.info("Shutting down...")
        self.risk_manager.save_trades()

        for trade in list(self.risk_manager.open_trades):
            try:
                ticker = await self.exchange.fetch_ticker(trade.symbol)
                await self._close_trade(trade, ticker["last"], "shutdown")
            except Exception as e:
                log.error(f"Error closing {trade.symbol} on shutdown: {e}")

        await self.exchange.close()
        log.info("Bot shutdown complete")
