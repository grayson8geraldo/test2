"""Tests for risk management system."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.strategies.base import Signal, SignalType, StrategyName, Trade
from bot.strategies.risk_manager import RiskManager
import config


def _make_signal(
    signal_type=SignalType.SHORT,
    confidence=70,
    entry=50000,
    sl=51000,
    tp=47000,
) -> Signal:
    rr = abs(entry - tp) / abs(sl - entry) if abs(sl - entry) > 0 else 0
    return Signal(
        signal_type=signal_type,
        strategy=StrategyName.TREND_FOLLOW,
        symbol="BTC/USDT",
        confidence=confidence,
        entry_price=entry,
        stop_loss=sl,
        take_profit=tp,
        risk_reward=rr,
        reasoning="test",
    )


def test_can_trade_initial():
    rm = RiskManager()
    can, reason = rm.can_trade()
    assert can, f"Should be able to trade initially: {reason}"


def test_position_sizing():
    rm = RiskManager()
    rm.balance = 200
    signal = _make_signal(confidence=70, entry=50000, sl=51000, tp=47000)
    qty = rm.calculate_position_size(signal)
    assert qty > 0, "Position size should be positive"
    # Check max position cap
    position_value = qty * signal.entry_price
    max_allowed = rm.balance * 0.30 * config.RISK["leverage"]
    assert position_value <= max_allowed * 1.01, "Position exceeds max cap"


def test_validate_signal_low_confidence():
    rm = RiskManager()
    signal = _make_signal(confidence=30)
    valid, reason = rm.validate_signal(signal)
    assert not valid, "Low confidence signal should be rejected"


def test_max_positions_limit():
    rm = RiskManager()
    # Fill up positions
    for i in range(config.RISK["max_open_positions"]):
        trade = Trade(
            trade_id=str(i), symbol=f"PAIR{i}/USDT",
            signal_type=SignalType.SHORT, strategy=StrategyName.TREND_FOLLOW,
            entry_price=100, stop_loss=110, take_profit=90,
            quantity=1, leverage=10, entry_time="2024-01-01",
        )
        rm.open_trades.append(trade)

    can, reason = rm.can_trade()
    assert not can, "Should not trade when max positions reached"


def test_daily_loss_limit():
    rm = RiskManager()
    rm.balance = 200
    rm.daily_pnl = -20  # 10% loss > 8% limit
    can, reason = rm.can_trade()
    assert not can, "Should stop trading after daily loss limit"


def test_trailing_stop_short():
    rm = RiskManager()
    trade = Trade(
        trade_id="t1", symbol="BTC/USDT",
        signal_type=SignalType.SHORT, strategy=StrategyName.TREND_FOLLOW,
        entry_price=50000, stop_loss=51000, take_profit=47000,
        quantity=0.01, leverage=10, entry_time="2024-01-01",
    )
    # Price moves favorably (down for short)
    updated = rm.update_trailing_stop(trade, 48000)
    assert updated.stop_loss < 51000, "Trailing stop should move down for profitable short"


def test_trailing_stop_long():
    rm = RiskManager()
    trade = Trade(
        trade_id="t1", symbol="BTC/USDT",
        signal_type=SignalType.LONG, strategy=StrategyName.MEAN_REVERSION,
        entry_price=50000, stop_loss=49000, take_profit=53000,
        quantity=0.01, leverage=10, entry_time="2024-01-01",
    )
    # Price moves favorably (up for long)
    updated = rm.update_trailing_stop(trade, 52000)
    assert updated.stop_loss > 49000, "Trailing stop should move up for profitable long"


def test_stats_calculation():
    rm = RiskManager()
    rm.balance = 220

    # Add some closed trades
    for i in range(5):
        t = Trade(
            trade_id=str(i), symbol="BTC/USDT",
            signal_type=SignalType.SHORT, strategy=StrategyName.TREND_FOLLOW,
            entry_price=50000, stop_loss=51000, take_profit=47000,
            quantity=0.01, leverage=10, entry_time="2024-01-01",
            exit_price=49000, pnl=4.0 if i < 3 else -2.0,
            pnl_pct=2.0 if i < 3 else -1.0, status="closed",
        )
        rm.closed_trades.append(t)

    stats = rm.get_stats()
    assert stats["total_trades"] == 5
    assert stats["win_rate"] == 0.6


if __name__ == "__main__":
    test_can_trade_initial()
    test_position_sizing()
    test_validate_signal_low_confidence()
    test_max_positions_limit()
    test_daily_loss_limit()
    test_trailing_stop_short()
    test_trailing_stop_long()
    test_stats_calculation()
    print("All risk manager tests passed!")
