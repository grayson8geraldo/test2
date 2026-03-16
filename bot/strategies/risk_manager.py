"""
Risk Management System for small deposit acceleration.

Key principles for $200 -> $1200 in bear market:
- Aggressive but NOT reckless position sizing
- ATR-based dynamic stop losses
- Daily loss limit (circuit breaker)
- Position sizing based on Kelly-like criterion
- Correlation-aware — avoid doubling down on same direction
"""

import json
import math
import os
from datetime import datetime, timedelta

import config
from bot.strategies.base import Signal, SignalType, Trade
from bot.utils.logger import log


def _fmt_price(price: float) -> str:
    """Format price with appropriate decimal places based on magnitude."""
    if price >= 100:
        return f"{price:.2f}"
    elif price >= 1:
        return f"{price:.4f}"
    elif price >= 0.01:
        return f"{price:.6f}"
    else:
        return f"{price:.8f}"


class RiskManager:

    def __init__(self):
        self.open_trades: list[Trade] = []
        self.closed_trades: list[Trade] = []
        self.daily_pnl: float = 0.0
        self.daily_trades: int = 0
        self.day_start: str = datetime.utcnow().strftime("%Y-%m-%d")
        self.balance: float = config.INITIAL_DEPOSIT
        self.peak_balance: float = config.INITIAL_DEPOSIT

    def can_trade(self) -> tuple[bool, str]:
        """Check if we are allowed to open new positions."""
        self._reset_daily_if_needed()

        # Daily loss limit — use peak balance to avoid shrinking limit after losses
        daily_loss_limit = self.peak_balance * config.RISK["max_daily_loss_pct"] / 100
        if self.daily_pnl < -daily_loss_limit:
            return False, f"Daily loss limit hit ({self.daily_pnl:.2f} < -{daily_loss_limit:.2f})"

        # Max open positions
        if len(self.open_trades) >= config.RISK["max_open_positions"]:
            return False, f"Max positions open ({len(self.open_trades)})"

        # Max daily trades
        if self.daily_trades >= config.RISK["max_daily_trades"]:
            return False, f"Max daily trades ({self.daily_trades})"

        # Balance sanity — don't trade if <10% of initial
        if self.balance < config.INITIAL_DEPOSIT * 0.1:
            return False, "Balance critically low"

        return True, "OK"

    def calculate_position_size(self, signal: Signal) -> float:
        """Calculate position size based on risk parameters and signal confidence.

        Uses a modified Kelly criterion weighted by confidence.
        """
        risk_pct = config.RISK["max_risk_per_trade_pct"]

        # Scale risk by confidence (50-100 mapped to 0.5x-1.0x)
        confidence_mult = 0.5 + (signal.confidence - 50) / 100
        adjusted_risk_pct = risk_pct * confidence_mult

        risk_amount = self.balance * adjusted_risk_pct / 100

        # Risk amount = position_size * |entry - stop_loss| / entry
        price_risk = abs(signal.entry_price - signal.stop_loss) / signal.entry_price
        if price_risk == 0:
            return 0.0

        position_value = risk_amount / price_risk
        position_value *= config.RISK["leverage"]

        # Cap at 30% of balance * leverage
        max_position = self.balance * 0.30 * config.RISK["leverage"]
        position_value = min(position_value, max_position)

        quantity = position_value / signal.entry_price
        return quantity

    def validate_signal(self, signal: Signal) -> tuple[bool, str]:
        """Validate a signal against risk rules before execution."""
        if not signal.is_actionable:
            return False, "Signal not actionable"

        can, reason = self.can_trade()
        if not can:
            return False, reason

        # Check R:R ratio
        if signal.risk_reward < config.RISK["min_risk_reward"]:
            return False, f"R:R too low ({signal.risk_reward:.2f})"

        # Check correlation — don't stack too many same-direction trades
        same_direction = sum(
            1 for t in self.open_trades
            if t.signal_type == signal.signal_type
        )
        if same_direction >= 2:
            return False, "Too many same-direction positions"

        # Check if already in this symbol
        in_symbol = any(t.symbol == signal.symbol for t in self.open_trades)
        if in_symbol:
            return False, f"Already in {signal.symbol}"

        return True, "OK"

    def register_open(self, trade: Trade):
        """Register a new open trade."""
        self.open_trades.append(trade)
        self.daily_trades += 1
        sl_pct = abs(trade.stop_loss - trade.entry_price) / trade.entry_price * 100
        tp_pct = abs(trade.take_profit - trade.entry_price) / trade.entry_price * 100
        log.info(
            f"OPEN {trade.signal_type.value} {trade.symbol} "
            f"@ {_fmt_price(trade.entry_price)} qty={trade.quantity:.6f} "
            f"SL={_fmt_price(trade.stop_loss)} ({sl_pct:.1f}%) "
            f"TP={_fmt_price(trade.take_profit)} ({tp_pct:.1f}%)"
        )

    def register_close(self, trade: Trade):
        """Register a closed trade and update balance."""
        self.open_trades = [t for t in self.open_trades if t.trade_id != trade.trade_id]
        self.closed_trades.append(trade)

        if trade.pnl is not None:
            self.daily_pnl += trade.pnl
            self.balance += trade.pnl
            self.peak_balance = max(self.peak_balance, self.balance)

        log.info(
            f"CLOSE {trade.signal_type.value} {trade.symbol} "
            f"@ {_fmt_price(trade.exit_price)} PnL={trade.pnl:+.2f} ({trade.pnl_pct:+.2f}%) "
            f"reason={trade.close_reason} | Balance={self.balance:.2f}"
        )

    def check_stop_loss(self, trade: Trade, current_price: float) -> bool:
        """Check if stop loss has been hit."""
        if trade.signal_type == SignalType.SHORT:
            return current_price >= trade.stop_loss
        return current_price <= trade.stop_loss

    def check_take_profit(self, trade: Trade, current_price: float) -> bool:
        """Check if take profit has been hit."""
        if trade.signal_type == SignalType.SHORT:
            return current_price <= trade.take_profit
        return current_price >= trade.take_profit

    def update_trailing_stop(self, trade: Trade, current_price: float) -> Trade:
        """Update trailing stop if price has moved favorably."""
        trail_pct = config.RISK["trailing_stop_pct"] / 100

        if trade.signal_type == SignalType.SHORT:
            # For shorts, price moving down is favorable
            new_sl = current_price * (1 + trail_pct)
            if new_sl < trade.stop_loss:
                trade.stop_loss = new_sl
        else:
            # For longs, price moving up is favorable
            new_sl = current_price * (1 - trail_pct)
            if new_sl > trade.stop_loss:
                trade.stop_loss = new_sl

        return trade

    def get_stats(self) -> dict:
        """Get current risk/performance statistics."""
        total_closed = len(self.closed_trades)
        winners = [t for t in self.closed_trades if t.pnl and t.pnl > 0]
        losers = [t for t in self.closed_trades if t.pnl and t.pnl <= 0]

        win_rate = len(winners) / total_closed if total_closed > 0 else 0
        total_profit = sum(t.pnl for t in winners) if winners else 0
        total_loss = abs(sum(t.pnl for t in losers)) if losers else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

        max_drawdown = 0
        peak = config.INITIAL_DEPOSIT
        bal = config.INITIAL_DEPOSIT
        for t in self.closed_trades:
            if t.pnl:
                bal += t.pnl
                peak = max(peak, bal)
                dd = (peak - bal) / peak
                max_drawdown = max(max_drawdown, dd)

        return {
            "balance": self.balance,
            "total_trades": total_closed,
            "open_positions": len(self.open_trades),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_pnl": self.balance - config.INITIAL_DEPOSIT,
            "total_pnl_pct": (self.balance - config.INITIAL_DEPOSIT) / config.INITIAL_DEPOSIT * 100,
            "max_drawdown_pct": max_drawdown * 100,
            "daily_pnl": self.daily_pnl,
            "daily_trades": self.daily_trades,
            "avg_win": total_profit / len(winners) if winners else 0,
            "avg_loss": total_loss / len(losers) if losers else 0,
            "peak_balance": self.peak_balance,
        }

    def save_trades(self):
        """Persist trade history to disk."""
        os.makedirs(os.path.dirname(config.TRADE_LOG_FILE), exist_ok=True)
        trades = [self._trade_to_dict(t) for t in self.closed_trades]
        with open(config.TRADE_LOG_FILE, "w") as f:
            json.dump(trades, f, indent=2)

    def load_trades(self):
        """Load trade history from disk."""
        if os.path.exists(config.TRADE_LOG_FILE):
            with open(config.TRADE_LOG_FILE) as f:
                data = json.load(f)
            self.closed_trades = [self._dict_to_trade(d) for d in data]
            # Recalculate balance
            self.balance = config.INITIAL_DEPOSIT + sum(
                t.pnl for t in self.closed_trades if t.pnl
            )
            self.peak_balance = max(self.peak_balance, self.balance)

    def _reset_daily_if_needed(self):
        today = datetime.utcnow().strftime("%Y-%m-%d")
        if today != self.day_start:
            self.day_start = today
            self.daily_pnl = 0.0
            self.daily_trades = 0

    @staticmethod
    def _trade_to_dict(t: Trade) -> dict:
        return {
            "trade_id": t.trade_id,
            "symbol": t.symbol,
            "signal_type": t.signal_type.value,
            "strategy": t.strategy.value,
            "entry_price": t.entry_price,
            "stop_loss": t.stop_loss,
            "take_profit": t.take_profit,
            "quantity": t.quantity,
            "leverage": t.leverage,
            "entry_time": t.entry_time,
            "exit_price": t.exit_price,
            "exit_time": t.exit_time,
            "pnl": t.pnl,
            "pnl_pct": t.pnl_pct,
            "fees": t.fees,
            "status": t.status,
            "indicators_at_entry": t.indicators_at_entry,
            "indicators_at_exit": t.indicators_at_exit,
            "close_reason": t.close_reason,
        }

    @staticmethod
    def _dict_to_trade(d: dict) -> Trade:
        return Trade(
            trade_id=d["trade_id"],
            symbol=d["symbol"],
            signal_type=SignalType(d["signal_type"]),
            strategy=StrategyName(d["strategy"]),
            entry_price=d["entry_price"],
            stop_loss=d["stop_loss"],
            take_profit=d["take_profit"],
            quantity=d["quantity"],
            leverage=d["leverage"],
            entry_time=d["entry_time"],
            exit_price=d.get("exit_price"),
            exit_time=d.get("exit_time"),
            pnl=d.get("pnl"),
            pnl_pct=d.get("pnl_pct"),
            fees=d.get("fees", 0),
            status=d.get("status", "closed"),
            indicators_at_entry=d.get("indicators_at_entry", {}),
            indicators_at_exit=d.get("indicators_at_exit", {}),
            close_reason=d.get("close_reason", ""),
        )
