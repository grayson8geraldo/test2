"""
Crypto Trading Bot Configuration
Bear market strategy for small deposit acceleration ($200 -> $1200)
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Exchange settings
EXCHANGE = os.getenv("EXCHANGE", "binance")
API_KEY = os.getenv("API_KEY", "")
API_SECRET = os.getenv("API_SECRET", "")
USE_TESTNET = os.getenv("USE_TESTNET", "true").lower() == "true"

# Trading pairs optimized for bear market volatility
TRADING_PAIRS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "DOGE/USDT",
]

# Deposit and target
INITIAL_DEPOSIT = float(os.getenv("INITIAL_DEPOSIT", "200"))
TARGET_BALANCE = float(os.getenv("TARGET_BALANCE", "1200"))

# Timeframes for multi-timeframe analysis
TIMEFRAMES = {
    "scalp": "1m",
    "entry": "5m",
    "trend": "15m",
    "context": "1h",
}

# Risk management — aggressive but controlled for small deposit
RISK = {
    "max_risk_per_trade_pct": 3.0,       # 3% of balance per trade
    "max_open_positions": 3,              # max simultaneous positions
    "max_daily_loss_pct": 8.0,            # stop trading if daily loss > 8%
    "max_daily_trades": 30,               # prevent overtrading
    "leverage": 10,                       # futures leverage
    "default_sl_pct": 1.5,               # stop-loss percentage
    "default_tp_pct": 3.0,               # take-profit percentage
    "trailing_stop_pct": 1.0,            # trailing stop activation
    "min_risk_reward": 1.8,              # minimum R:R ratio
    "scale_in_enabled": True,            # allow scaling into positions
    "scale_in_max_adds": 2,              # max scale-in additions
}

# Bear market indicator settings
INDICATORS = {
    "rsi_period": 14,
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "ema_fast": 9,
    "ema_medium": 21,
    "ema_slow": 55,
    "bb_period": 20,
    "bb_std": 2.0,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "atr_period": 14,
    "volume_ma_period": 20,
    "stoch_k": 14,
    "stoch_d": 3,
    "stoch_smooth": 3,
    "adx_period": 14,
    "adx_strong_trend": 25,
}

# Strategy weights (bear market tuned)
STRATEGY = {
    "trend_follow_weight": 0.35,     # follow the bear trend (shorts)
    "mean_reversion_weight": 0.30,   # catch oversold bounces (longs)
    "breakout_weight": 0.20,         # breakdown trades (shorts)
    "scalp_weight": 0.15,            # quick in/out on volatility
}

# Self-learning settings
LEARNING = {
    "analysis_interval": 20,          # analyze after every N closed trades
    "min_trades_for_adjustment": 20,
    "performance_window": 100,         # rolling window for stats
    "weight_adjustment_step": 0.05,   # max weight change per cycle
    "indicator_sensitivity_step": 5,  # % change in indicator params
    "win_rate_target": 0.55,          # target win rate
    "profit_factor_target": 1.5,      # target profit factor
    "save_learning_data": True,
    "learning_data_path": "data/learning_state.json",
}

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = "logs/trading_bot.log"
TRADE_LOG_FILE = "data/trades.json"
