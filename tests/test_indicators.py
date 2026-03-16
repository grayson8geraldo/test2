"""Tests for technical indicators module."""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.indicators.technical import compute_all


def _make_ohlcv(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data simulating a bear market."""
    rng = np.random.RandomState(seed)
    # Downtrend with noise
    base = 100 - np.linspace(0, 30, n) + rng.randn(n) * 2
    opens = base + rng.randn(n) * 0.5
    closes = base + rng.randn(n) * 0.5
    highs = np.maximum(opens, closes) + abs(rng.randn(n)) * 1.0
    lows = np.minimum(opens, closes) - abs(rng.randn(n)) * 1.0
    volume = rng.uniform(1000, 5000, n)

    df = pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volume,
    })
    return df


def test_compute_all_columns():
    """Test that all expected indicator columns are produced."""
    df = _make_ohlcv(100)
    result = compute_all(df)

    expected_cols = [
        "ema_fast", "ema_medium", "ema_slow",
        "rsi", "rsi_oversold", "rsi_overbought",
        "bb_upper", "bb_middle", "bb_lower", "bb_pctb",
        "macd_line", "macd_signal", "macd_hist",
        "atr", "atr_pct",
        "stoch_k", "stoch_d",
        "adx", "plus_di", "minus_di",
        "vol_ma", "vol_ratio",
        "obv",
        "short_score", "long_score",
    ]

    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"


def test_rsi_range():
    """RSI should be between 0 and 100."""
    df = _make_ohlcv(150)
    result = compute_all(df)
    rsi = result["rsi"].dropna()
    assert rsi.min() >= 0, f"RSI below 0: {rsi.min()}"
    assert rsi.max() <= 100, f"RSI above 100: {rsi.max()}"


def test_bb_ordering():
    """Bollinger Bands: upper > middle > lower."""
    df = _make_ohlcv(100)
    result = compute_all(df).dropna()
    assert (result["bb_upper"] >= result["bb_middle"]).all()
    assert (result["bb_middle"] >= result["bb_lower"]).all()


def test_bear_aligned_detection():
    """In a downtrend, ema_bear_aligned should be frequently True."""
    df = _make_ohlcv(200, seed=10)
    result = compute_all(df)
    bear_pct = result["ema_bear_aligned"].iloc[60:].mean()
    # In our synthetic bear data, should detect bear alignment often
    assert bear_pct > 0.3, f"Bear alignment too low: {bear_pct:.1%}"


def test_scores_range():
    """Short/long scores should be between 0 and 100."""
    df = _make_ohlcv(150)
    result = compute_all(df).dropna()
    assert result["short_score"].min() >= 0
    assert result["short_score"].max() <= 100
    assert result["long_score"].min() >= 0
    assert result["long_score"].max() <= 100


def test_custom_params():
    """Test that custom parameters override defaults."""
    df = _make_ohlcv(100)
    result = compute_all(df, params={"rsi_period": 7, "ema_fast": 5})
    # Should still have all columns
    assert "rsi" in result.columns
    assert "ema_fast" in result.columns


if __name__ == "__main__":
    test_compute_all_columns()
    test_rsi_range()
    test_bb_ordering()
    test_bear_aligned_detection()
    test_scores_range()
    test_custom_params()
    print("All indicator tests passed!")
