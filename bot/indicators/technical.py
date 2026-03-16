"""
Technical indicators optimized for bear market crypto trading.

Indicators chosen for bear market:
- RSI: Identify oversold bounces and overbought short entries
- Bollinger Bands: Volatility-based mean reversion signals
- MACD: Trend confirmation and momentum shifts
- EMA ribbon (9/21/55): Trend direction and dynamic S/R
- ATR: Volatility-based stop-loss and position sizing
- Stochastic: Momentum extremes in ranging/bear conditions
- ADX: Trend strength to filter noise
- Volume Profile: Confirm breakdowns and reversals
- OBV: Detect smart money divergences
"""

import numpy as np
import pandas as pd

import config


def compute_all(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """Compute all technical indicators on OHLCV dataframe.

    Args:
        df: DataFrame with columns [open, high, low, close, volume]
        params: Override indicator parameters (used by self-learning module)

    Returns:
        DataFrame with all indicator columns added.
    """
    p = {**config.INDICATORS, **(params or {})}
    df = df.copy()

    df = _ema_ribbon(df, p)
    df = _rsi(df, p)
    df = _bollinger_bands(df, p)
    df = _macd(df, p)
    df = _atr(df, p)
    df = _stochastic(df, p)
    df = _adx(df, p)
    df = _volume_analysis(df, p)
    df = _obv(df)
    df = _bear_market_signals(df, p)

    return df


def _ema_ribbon(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    """EMA ribbon: fast/medium/slow for trend context."""
    df["ema_fast"] = df["close"].ewm(span=p["ema_fast"], adjust=False).mean()
    df["ema_medium"] = df["close"].ewm(span=p["ema_medium"], adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=p["ema_slow"], adjust=False).mean()

    # Bear alignment: fast < medium < slow
    df["ema_bear_aligned"] = (
        (df["ema_fast"] < df["ema_medium"]) & (df["ema_medium"] < df["ema_slow"])
    ).astype(int)

    # Bull alignment for bounce detection
    df["ema_bull_aligned"] = (
        (df["ema_fast"] > df["ema_medium"]) & (df["ema_medium"] > df["ema_slow"])
    ).astype(int)

    # EMA slope (rate of change)
    df["ema_fast_slope"] = df["ema_fast"].pct_change(3)
    df["ema_medium_slope"] = df["ema_medium"].pct_change(3)

    return df


def _rsi(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    """RSI with divergence detection."""
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / p["rsi_period"], min_periods=p["rsi_period"]).mean()
    avg_loss = loss.ewm(alpha=1 / p["rsi_period"], min_periods=p["rsi_period"]).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    df["rsi_oversold"] = (df["rsi"] < p["rsi_oversold"]).astype(int)
    df["rsi_overbought"] = (df["rsi"] > p["rsi_overbought"]).astype(int)

    # RSI divergence: price makes lower low but RSI makes higher low (bullish)
    df["rsi_bull_div"] = _detect_bullish_divergence(df["close"], df["rsi"], lookback=14)
    # Price makes higher high but RSI makes lower high (bearish)
    df["rsi_bear_div"] = _detect_bearish_divergence(df["close"], df["rsi"], lookback=14)

    return df


def _bollinger_bands(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Bollinger Bands for volatility and mean reversion."""
    sma = df["close"].rolling(window=p["bb_period"]).mean()
    std = df["close"].rolling(window=p["bb_period"]).std()

    df["bb_upper"] = sma + p["bb_std"] * std
    df["bb_middle"] = sma
    df["bb_lower"] = sma - p["bb_std"] * std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]

    # BB %B — position within bands
    df["bb_pctb"] = (df["close"] - df["bb_lower"]) / (
        df["bb_upper"] - df["bb_lower"]
    ).replace(0, np.nan)

    # Squeeze detection (low volatility before breakout)
    bb_width_avg = df["bb_width"].rolling(window=50).mean()
    df["bb_squeeze"] = (df["bb_width"] < bb_width_avg * 0.75).astype(int)

    return df


def _macd(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    """MACD for momentum and trend confirmation."""
    ema_fast = df["close"].ewm(span=p["macd_fast"], adjust=False).mean()
    ema_slow = df["close"].ewm(span=p["macd_slow"], adjust=False).mean()

    df["macd_line"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd_line"].ewm(span=p["macd_signal"], adjust=False).mean()
    df["macd_hist"] = df["macd_line"] - df["macd_signal"]

    # MACD histogram momentum
    df["macd_hist_rising"] = (df["macd_hist"] > df["macd_hist"].shift(1)).astype(int)
    df["macd_cross_up"] = (
        (df["macd_line"] > df["macd_signal"])
        & (df["macd_line"].shift(1) <= df["macd_signal"].shift(1))
    ).astype(int)
    df["macd_cross_down"] = (
        (df["macd_line"] < df["macd_signal"])
        & (df["macd_line"].shift(1) >= df["macd_signal"].shift(1))
    ).astype(int)

    return df


def _atr(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    """ATR for volatility-based stops and position sizing."""
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = true_range.ewm(span=p["atr_period"], adjust=False).mean()
    df["atr_pct"] = df["atr"] / df["close"] * 100

    return df


def _stochastic(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Stochastic oscillator for momentum extremes."""
    low_min = df["low"].rolling(window=p["stoch_k"]).min()
    high_max = df["high"].rolling(window=p["stoch_k"]).max()

    denom = (high_max - low_min).replace(0, np.nan)
    df["stoch_k"] = ((df["close"] - low_min) / denom) * 100
    df["stoch_d"] = df["stoch_k"].rolling(window=p["stoch_d"]).mean()

    df["stoch_oversold"] = (df["stoch_k"] < 20).astype(int)
    df["stoch_overbought"] = (df["stoch_k"] > 80).astype(int)

    return df


def _adx(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    """ADX for trend strength measurement."""
    period = p["adx_period"]

    up_move = df["high"] - df["high"].shift(1)
    down_move = df["low"].shift(1) - df["low"]

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    plus_dm_s = pd.Series(plus_dm, index=df.index).ewm(span=period, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm, index=df.index).ewm(span=period, adjust=False).mean()

    atr = df["atr"] if "atr" in df.columns else _atr(df.copy(), p)["atr"]

    plus_di = (plus_dm_s / atr.replace(0, np.nan)) * 100
    minus_di = (minus_dm_s / atr.replace(0, np.nan)) * 100

    df["plus_di"] = plus_di
    df["minus_di"] = minus_di

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    df["adx"] = dx.ewm(span=period, adjust=False).mean()

    df["strong_trend"] = (df["adx"] > p["adx_strong_trend"]).astype(int)
    df["bear_trend"] = ((df["minus_di"] > df["plus_di"]) & df["strong_trend"].astype(bool)).astype(int)

    return df


def _volume_analysis(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Volume analysis for confirmation."""
    df["vol_ma"] = df["volume"].rolling(window=p["volume_ma_period"]).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma"].replace(0, np.nan)

    # High volume spike
    df["vol_spike"] = (df["vol_ratio"] > 2.0).astype(int)

    # Volume trend
    df["vol_increasing"] = (
        df["vol_ma"] > df["vol_ma"].shift(5)
    ).astype(int)

    # Sell volume dominance (bear market key signal)
    df["sell_volume"] = np.where(df["close"] < df["open"], df["volume"], 0)
    df["buy_volume"] = np.where(df["close"] >= df["open"], df["volume"], 0)
    df["sell_vol_ratio"] = (
        pd.Series(df["sell_volume"]).rolling(window=10).sum()
        / df["volume"].rolling(window=10).sum().replace(0, np.nan)
    )

    return df


def _obv(df: pd.DataFrame) -> pd.DataFrame:
    """On-Balance Volume for smart money detection."""
    obv = np.where(
        df["close"] > df["close"].shift(1),
        df["volume"],
        np.where(df["close"] < df["close"].shift(1), -df["volume"], 0),
    )
    df["obv"] = pd.Series(obv, index=df.index).cumsum()
    df["obv_ema"] = df["obv"].ewm(span=20, adjust=False).mean()
    df["obv_divergence"] = np.sign(df["obv"].diff(5)) != np.sign(df["close"].diff(5))

    return df


def _bear_market_signals(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Composite bear market signals combining multiple indicators."""

    # SHORT signal strength (0-100)
    short_score = pd.Series(0.0, index=df.index)
    short_score += df["ema_bear_aligned"] * 20
    short_score += df["rsi_overbought"] * 15
    short_score += df["bear_trend"] * 20
    short_score += df["macd_cross_down"] * 15
    short_score += (df["bb_pctb"] > 0.8).astype(float) * 10
    short_score += df["stoch_overbought"] * 10
    short_score += (df["sell_vol_ratio"] > 0.6).astype(float) * 10
    df["short_score"] = short_score.clip(0, 100)

    # LONG signal strength (0-100) — for bear market bounces
    long_score = pd.Series(0.0, index=df.index)
    long_score += df["rsi_oversold"] * 20
    long_score += df["rsi_bull_div"].astype(float) * 20
    long_score += df["macd_cross_up"] * 15
    long_score += (df["bb_pctb"] < 0.1).astype(float) * 15
    long_score += df["stoch_oversold"] * 10
    long_score += df["vol_spike"] * 10
    long_score += df["macd_hist_rising"] * 10
    df["long_score"] = long_score.clip(0, 100)

    return df


def _detect_bullish_divergence(
    price: pd.Series, indicator: pd.Series, lookback: int = 14
) -> pd.Series:
    """Detect bullish divergence: price lower low, indicator higher low."""
    result = pd.Series(False, index=price.index)
    for i in range(lookback, len(price)):
        window_price = price.iloc[i - lookback : i + 1]
        window_ind = indicator.iloc[i - lookback : i + 1]

        price_lows = _find_local_minima(window_price)
        if len(price_lows) >= 2:
            last_two = price_lows[-2:]
            p1, p2 = window_price.iloc[last_two[0]], window_price.iloc[last_two[1]]
            i1, i2 = window_ind.iloc[last_two[0]], window_ind.iloc[last_two[1]]

            if p2 < p1 and i2 > i1:
                result.iloc[i] = True
    return result


def _detect_bearish_divergence(
    price: pd.Series, indicator: pd.Series, lookback: int = 14
) -> pd.Series:
    """Detect bearish divergence: price higher high, indicator lower high."""
    result = pd.Series(False, index=price.index)
    for i in range(lookback, len(price)):
        window_price = price.iloc[i - lookback : i + 1]
        window_ind = indicator.iloc[i - lookback : i + 1]

        price_highs = _find_local_maxima(window_price)
        if len(price_highs) >= 2:
            last_two = price_highs[-2:]
            p1, p2 = window_price.iloc[last_two[0]], window_price.iloc[last_two[1]]
            i1, i2 = window_ind.iloc[last_two[0]], window_ind.iloc[last_two[1]]

            if p2 > p1 and i2 < i1:
                result.iloc[i] = True
    return result


def _find_local_minima(series: pd.Series, order: int = 3) -> list[int]:
    """Find local minima indices in a series."""
    minima = []
    vals = series.values
    for i in range(order, len(vals) - order):
        if all(vals[i] <= vals[i - j] for j in range(1, order + 1)) and all(
            vals[i] <= vals[i + j] for j in range(1, order + 1)
        ):
            minima.append(i)
    return minima


def _find_local_maxima(series: pd.Series, order: int = 3) -> list[int]:
    """Find local maxima indices in a series."""
    maxima = []
    vals = series.values
    for i in range(order, len(vals) - order):
        if all(vals[i] >= vals[i - j] for j in range(1, order + 1)) and all(
            vals[i] >= vals[i + j] for j in range(1, order + 1)
        ):
            maxima.append(i)
    return maxima
