"""
Bear market trading strategies for crypto.

Four complementary strategies:
1. Trend Follow (Short) — ride the downtrend
2. Mean Reversion — catch oversold bounces
3. Breakout/Breakdown — trade support/resistance breaks
4. Scalp — quick trades on micro-movements
"""

import pandas as pd
import numpy as np

import config
from bot.strategies.base import Signal, SignalType, StrategyName


class BearTrendFollow:
    """
    Strategy: Follow the dominant bear trend with short positions.

    Entry: EMA bear-aligned + RSI turning down from mid-zone + MACD bearish +
           strong ADX downtrend + volume confirmation.
    Exit: RSI oversold + MACD histogram reversing + trailing stop.
    """

    name = StrategyName.TREND_FOLLOW

    def __init__(self, weight: float = 0.35, params: dict | None = None):
        self.weight = weight
        self.params = params or {}

    def evaluate(self, df: pd.DataFrame, symbol: str) -> Signal:
        last = df.iloc[-1]
        prev = df.iloc[-2]
        price = last["close"]

        confidence = 0.0
        reasons = []

        # Core: EMA bear alignment
        if last["ema_bear_aligned"]:
            confidence += 25
            reasons.append("EMA bear-aligned")

        # RSI in bearish zone (40-65) and declining
        if 40 < last["rsi"] < 65 and last["rsi"] < prev["rsi"]:
            confidence += 20
            reasons.append(f"RSI declining ({last['rsi']:.0f})")

        # MACD bearish
        if last["macd_line"] < last["macd_signal"] and last["macd_hist"] < 0:
            confidence += 15
            reasons.append("MACD bearish")

        # Strong downtrend via ADX
        if last["bear_trend"]:
            confidence += 20
            reasons.append(f"Strong bear trend (ADX={last['adx']:.0f})")

        # Volume confirms selling pressure
        if last["sell_vol_ratio"] > 0.55:
            confidence += 10
            reasons.append("Sell volume dominant")

        # Bearish divergence bonus
        if last["rsi_bear_div"]:
            confidence += 10
            reasons.append("RSI bearish divergence")

        confidence = min(confidence * self.weight / 0.35, 100)

        # Calculate stops based on ATR
        atr = last["atr"]
        stop_loss = price + atr * 2.0
        take_profit = price - atr * 3.5
        rr = abs(price - take_profit) / abs(stop_loss - price) if abs(stop_loss - price) > 0 else 0

        if confidence < 40 or rr < config.RISK["min_risk_reward"]:
            return Signal(
                signal_type=SignalType.NO_SIGNAL, strategy=self.name,
                symbol=symbol, confidence=0, entry_price=price,
                stop_loss=stop_loss, take_profit=take_profit,
                risk_reward=rr, reasoning="Insufficient confidence",
            )

        return Signal(
            signal_type=SignalType.SHORT,
            strategy=self.name,
            symbol=symbol,
            confidence=confidence,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward=rr,
            reasoning=" | ".join(reasons),
            indicators={
                "rsi": last["rsi"],
                "macd_hist": last["macd_hist"],
                "adx": last["adx"],
                "atr": atr,
                "ema_bear_aligned": last["ema_bear_aligned"],
            },
        )

    def check_exit(self, df: pd.DataFrame) -> tuple[bool, str]:
        """Check if an open short should be closed — only on strong reversal."""
        last = df.iloc[-1]

        # RSI extreme + confirmed momentum shift
        if last["rsi"] < 20 and last["macd_hist_rising"]:
            return True, "RSI extreme oversold + momentum shift"
        # MACD bullish cross confirmed by RSI turning up from oversold
        if last["macd_cross_up"] and last["rsi"] < 35 and last["macd_hist_rising"]:
            return True, "MACD bullish crossover from oversold"
        # Strong bullish divergence
        if last["rsi_bull_div"] and last["vol_spike"]:
            return True, "Bullish divergence + volume spike"
        return False, ""


class BearMeanReversion:
    """
    Strategy: Catch oversold bounces in the bear market for quick longs.

    Entry: RSI deeply oversold + price at/below BB lower + stochastic oversold +
           volume spike (capitulation) + bullish divergence preferred.
    Exit: Price hits BB middle or EMA21 + RSI exits oversold.
    """

    name = StrategyName.MEAN_REVERSION

    def __init__(self, weight: float = 0.30, params: dict | None = None):
        self.weight = weight
        self.params = params or {}

    def evaluate(self, df: pd.DataFrame, symbol: str) -> Signal:
        last = df.iloc[-1]
        prev = df.iloc[-2]
        price = last["close"]

        confidence = 0.0
        reasons = []

        # RSI deeply oversold
        if last["rsi"] < 25:
            confidence += 25
            reasons.append(f"RSI deeply oversold ({last['rsi']:.0f})")
        elif last["rsi"] < 30:
            confidence += 15
            reasons.append(f"RSI oversold ({last['rsi']:.0f})")

        # Price at or below BB lower band
        if last["bb_pctb"] < 0.05:
            confidence += 20
            reasons.append("Price below BB lower")
        elif last["bb_pctb"] < 0.15:
            confidence += 10
            reasons.append("Price near BB lower")

        # Stochastic oversold
        if last["stoch_oversold"]:
            confidence += 15
            reasons.append("Stochastic oversold")

        # Volume spike — capitulation signal
        if last["vol_spike"]:
            confidence += 15
            reasons.append("Volume spike (capitulation)")

        # Bullish divergence — strongest signal
        if last["rsi_bull_div"]:
            confidence += 20
            reasons.append("RSI bullish divergence")

        # MACD histogram turning up
        if last["macd_hist_rising"] and last["macd_hist"] < 0:
            confidence += 5
            reasons.append("MACD hist reversing up")

        confidence = min(confidence * self.weight / 0.30, 100)

        atr = last["atr"]
        stop_loss = price - atr * 1.5
        # Conservative TP for bear bounce: target BB middle or EMA21
        take_profit = min(last["bb_middle"], last["ema_medium"])
        rr = abs(take_profit - price) / abs(price - stop_loss) if abs(price - stop_loss) > 0 else 0

        if confidence < 40 or rr < config.RISK["min_risk_reward"]:
            return Signal(
                signal_type=SignalType.NO_SIGNAL, strategy=self.name,
                symbol=symbol, confidence=0, entry_price=price,
                stop_loss=stop_loss, take_profit=take_profit,
                risk_reward=rr, reasoning="Insufficient confidence",
            )

        return Signal(
            signal_type=SignalType.LONG,
            strategy=self.name,
            symbol=symbol,
            confidence=confidence,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward=rr,
            reasoning=" | ".join(reasons),
            indicators={
                "rsi": last["rsi"],
                "bb_pctb": last["bb_pctb"],
                "stoch_k": last["stoch_k"],
                "vol_ratio": last["vol_ratio"],
                "atr": atr,
            },
        )

    def check_exit(self, df: pd.DataFrame) -> tuple[bool, str]:
        """Exit long bounce — let TP/SL handle most exits."""
        last = df.iloc[-1]

        # Only signal exit when RSI shows the bounce is exhausted
        if last["rsi"] > 60 and last["close"] >= last["bb_middle"]:
            return True, "Bounce target reached (RSI>60 + BB middle)"
        if last["rsi"] > 65:
            return True, "RSI overbought for bounce trade"
        return False, ""


class BearBreakdown:
    """
    Strategy: Trade breakdowns through key support levels.

    Entry: Price breaks below BB lower with volume + bearish MACD + ADX strong trend.
    Exit: Volume exhaustion + RSI extreme + price deviation from BB.
    """

    name = StrategyName.BREAKOUT

    def __init__(self, weight: float = 0.20, params: dict | None = None):
        self.weight = weight
        self.params = params or {}

    def evaluate(self, df: pd.DataFrame, symbol: str) -> Signal:
        last = df.iloc[-1]
        prev = df.iloc[-2]
        price = last["close"]

        confidence = 0.0
        reasons = []

        # Price broke below BB lower (breakdown)
        broke_bb = price < last["bb_lower"] and prev["close"] >= prev["bb_lower"]
        if broke_bb:
            confidence += 25
            reasons.append("BB lower breakdown")

        # Was in squeeze — breakdown from low volatility is powerful
        if last["bb_squeeze"] or prev["bb_squeeze"]:
            confidence += 15
            reasons.append("Post-squeeze breakdown")

        # Volume confirms the breakdown
        if last["vol_ratio"] > 1.5:
            confidence += 20
            reasons.append(f"Volume confirms ({last['vol_ratio']:.1f}x)")

        # MACD bearish momentum
        if last["macd_hist"] < prev["macd_hist"] and last["macd_hist"] < 0:
            confidence += 15
            reasons.append("MACD accelerating bearish")

        # EMA bear context
        if last["ema_bear_aligned"]:
            confidence += 10
            reasons.append("Bear trend context")

        # ADX trending
        if last["strong_trend"]:
            confidence += 10
            reasons.append("Strong trend")

        # Recent lower lows (momentum)
        lookback = df.tail(10)
        new_low = price <= lookback["low"].min()
        if new_low:
            confidence += 5
            reasons.append("New local low")

        confidence = min(confidence * self.weight / 0.20, 100)

        atr = last["atr"]
        stop_loss = price + atr * 1.8
        take_profit = price - atr * 4.0
        rr = abs(price - take_profit) / abs(stop_loss - price) if abs(stop_loss - price) > 0 else 0

        if confidence < 40 or rr < config.RISK["min_risk_reward"]:
            return Signal(
                signal_type=SignalType.NO_SIGNAL, strategy=self.name,
                symbol=symbol, confidence=0, entry_price=price,
                stop_loss=stop_loss, take_profit=take_profit,
                risk_reward=rr, reasoning="Insufficient confidence",
            )

        return Signal(
            signal_type=SignalType.SHORT,
            strategy=self.name,
            symbol=symbol,
            confidence=confidence,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward=rr,
            reasoning=" | ".join(reasons),
            indicators={
                "rsi": last["rsi"],
                "bb_pctb": last["bb_pctb"],
                "bb_squeeze": last["bb_squeeze"],
                "vol_ratio": last["vol_ratio"],
                "macd_hist": last["macd_hist"],
                "atr": atr,
            },
        )

    def check_exit(self, df: pd.DataFrame) -> tuple[bool, str]:
        last = df.iloc[-1]

        # Only exit on strong reversal signals, not transient conditions
        if last["rsi"] < 15 and last["macd_hist_rising"]:
            return True, "Extreme RSI with momentum reversal"
        if last["rsi_bull_div"]:
            return True, "Bullish divergence detected"
        if last["macd_cross_up"] and last["rsi"] < 30:
            return True, "MACD cross up from oversold"
        return False, ""


class BearScalp:
    """
    Strategy: Quick scalping on micro-movements using 1m/5m timeframes.

    Entry: Stochastic crosses in oversold/overbought zones + tight BB +
           volume micro-spike.
    Exit: Quick TP at 0.5-1% or opposing stochastic signal.
    """

    name = StrategyName.SCALP

    def __init__(self, weight: float = 0.15, params: dict | None = None):
        self.weight = weight
        self.params = params or {}

    def evaluate(self, df: pd.DataFrame, symbol: str) -> Signal:
        last = df.iloc[-1]
        prev = df.iloc[-2]
        price = last["close"]
        atr = last["atr"]

        # Determine scalp direction based on micro-signals
        short_score = 0.0
        long_score = 0.0
        short_reasons = []
        long_reasons = []

        # Stochastic overbought -> short scalp
        if last["stoch_overbought"] and last["stoch_k"] < prev["stoch_k"]:
            short_score += 30
            short_reasons.append("Stoch turning from overbought")

        # Stochastic oversold -> long scalp
        if last["stoch_oversold"] and last["stoch_k"] > prev["stoch_k"]:
            long_score += 30
            long_reasons.append("Stoch turning from oversold")

        # RSI micro-signals
        if 60 < last["rsi"] < 70 and last["rsi"] < prev["rsi"]:
            short_score += 20
            short_reasons.append("RSI turning down")
        if 30 < last["rsi"] < 40 and last["rsi"] > prev["rsi"]:
            long_score += 20
            long_reasons.append("RSI turning up")

        # Price at BB edge
        if last["bb_pctb"] > 0.9:
            short_score += 20
            short_reasons.append("At BB upper")
        if last["bb_pctb"] < 0.1:
            long_score += 20
            long_reasons.append("At BB lower")

        # Volume micro-spike
        if 1.3 < last["vol_ratio"] < 3.0:
            short_score += 10
            long_score += 10
            short_reasons.append("Vol micro-spike")
            long_reasons.append("Vol micro-spike")

        # Bear market bias: prefer shorts
        short_score *= 1.2 if last["ema_bear_aligned"] else 0.8
        long_score *= 0.8 if last["ema_bear_aligned"] else 1.0

        # Pick direction
        if short_score > long_score and short_score >= 40:
            confidence = min(short_score * self.weight / 0.15, 100)
            sl = price + atr * 1.0
            tp = price - atr * 1.5
            rr = abs(price - tp) / abs(sl - price) if abs(sl - price) > 0 else 0

            if confidence >= 40 and rr >= 1.2:
                return Signal(
                    signal_type=SignalType.SHORT, strategy=self.name,
                    symbol=symbol, confidence=confidence, entry_price=price,
                    stop_loss=sl, take_profit=tp, risk_reward=rr,
                    reasoning=" | ".join(short_reasons),
                    indicators={"rsi": last["rsi"], "stoch_k": last["stoch_k"], "atr": atr},
                )

        elif long_score > short_score and long_score >= 40:
            confidence = min(long_score * self.weight / 0.15, 100)
            sl = price - atr * 1.0
            tp = price + atr * 1.5
            rr = abs(tp - price) / abs(price - sl) if abs(price - sl) > 0 else 0

            if confidence >= 40 and rr >= 1.2:
                return Signal(
                    signal_type=SignalType.LONG, strategy=self.name,
                    symbol=symbol, confidence=confidence, entry_price=price,
                    stop_loss=sl, take_profit=tp, risk_reward=rr,
                    reasoning=" | ".join(long_reasons),
                    indicators={"rsi": last["rsi"], "stoch_k": last["stoch_k"], "atr": atr},
                )

        return Signal(
            signal_type=SignalType.NO_SIGNAL, strategy=self.name,
            symbol=symbol, confidence=0, entry_price=price,
            stop_loss=price, take_profit=price, risk_reward=0,
            reasoning="No scalp setup",
        )

    def check_exit(self, df: pd.DataFrame) -> tuple[bool, str]:
        """Scalp exits are primarily via TP/SL. Strategy exit only on clear reversal."""
        # Scalps should rely on TP/SL, not signal-based exits
        # Only exit on extreme counter-move
        return False, ""
