"""Base strategy and signal model."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SignalType(Enum):
    LONG = "long"
    SHORT = "short"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"
    NO_SIGNAL = "no_signal"


class StrategyName(Enum):
    TREND_FOLLOW = "trend_follow"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    SCALP = "scalp"


@dataclass
class Signal:
    signal_type: SignalType
    strategy: StrategyName
    symbol: str
    confidence: float  # 0-100
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    reasoning: str
    indicators: dict = field(default_factory=dict)
    timestamp: Optional[str] = None

    @property
    def is_actionable(self) -> bool:
        return self.signal_type not in (SignalType.NO_SIGNAL,) and self.confidence >= 50


@dataclass
class Trade:
    trade_id: str
    symbol: str
    signal_type: SignalType
    strategy: StrategyName
    entry_price: float
    stop_loss: float
    take_profit: float
    quantity: float
    leverage: int
    entry_time: str
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    fees: float = 0.0
    status: str = "open"  # open, closed, cancelled
    indicators_at_entry: dict = field(default_factory=dict)
    indicators_at_exit: dict = field(default_factory=dict)
    close_reason: str = ""  # tp, sl, trailing, signal, manual
