"""Tests for self-learning module."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.learning.self_learner import SelfLearner
from bot.strategies.base import Trade, SignalType, StrategyName


def _make_trades(n: int = 25, win_rate: float = 0.6) -> list[Trade]:
    """Generate synthetic trade history."""
    trades = []
    for i in range(n):
        is_win = i % 10 < int(win_rate * 10)
        pnl = 5.0 if is_win else -3.0
        strategy = [StrategyName.TREND_FOLLOW, StrategyName.MEAN_REVERSION,
                     StrategyName.BREAKOUT, StrategyName.SCALP][i % 4]

        trades.append(Trade(
            trade_id=str(i),
            symbol=["BTC/USDT", "ETH/USDT", "SOL/USDT"][i % 3],
            signal_type=SignalType.SHORT if i % 2 == 0 else SignalType.LONG,
            strategy=strategy,
            entry_price=50000,
            stop_loss=51000,
            take_profit=47000,
            quantity=0.01,
            leverage=10,
            entry_time="2024-01-01",
            exit_price=49000 if is_win else 50500,
            pnl=pnl,
            pnl_pct=2.5 if is_win else -1.5,
            status="closed",
            close_reason="tp" if is_win else "sl",
            indicators_at_entry={"rsi": 45 + i, "atr": 500, "bb_pctb": 0.3},
        ))
    return trades


def test_should_analyze():
    learner = SelfLearner()
    assert not learner.should_analyze(10), "Should not analyze before 20 trades"
    assert learner.should_analyze(20), "Should analyze at 20 trades"
    assert learner.should_analyze(25), "Should analyze after 20 trades"


def test_analyze_produces_report():
    learner = SelfLearner()
    trades = _make_trades(25)
    report = learner.analyze(trades)

    assert "strategy_analysis" in report
    assert "weight_changes" in report
    assert "indicator_analysis" in report
    assert "risk_analysis" in report
    assert "pair_analysis" in report
    assert "summary" in report


def test_weights_sum_to_one():
    learner = SelfLearner()
    trades = _make_trades(25)
    learner.analyze(trades)

    total = sum(learner.strategy_weights.values())
    assert abs(total - 1.0) < 0.01, f"Weights should sum to 1.0, got {total}"


def test_pair_scores():
    learner = SelfLearner()
    trades = _make_trades(25)
    learner.analyze(trades)

    for symbol in ["BTC/USDT", "ETH/USDT", "SOL/USDT"]:
        score = learner.get_pair_score(symbol)
        assert 0 <= score <= 1.0, f"Pair score out of range: {score}"


def test_multiple_cycles():
    learner = SelfLearner()

    # First cycle
    trades = _make_trades(25)
    learner.analyze(trades)
    w1 = dict(learner.strategy_weights)

    # Second cycle with different performance
    more_trades = _make_trades(45, win_rate=0.8)
    learner.analysis_count = 1  # reset to allow next analysis
    learner.analyze(more_trades)
    w2 = dict(learner.strategy_weights)

    assert learner.analysis_count == 2, "Should have done 2 analysis cycles"
    # Weights may have changed
    assert sum(w2.values()) - 1.0 < 0.01


if __name__ == "__main__":
    test_should_analyze()
    test_analyze_produces_report()
    test_weights_sum_to_one()
    test_pair_scores()
    test_multiple_cycles()
    print("All self-learner tests passed!")
