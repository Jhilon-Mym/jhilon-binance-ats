import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    USE_TESTNET = os.getenv("USE_TESTNET", "true").lower() == "true"
    if USE_TESTNET:
        API_KEY = os.getenv("TESTNET_API_KEY")
        API_SECRET = os.getenv("TESTNET_API_SECRET")
    else:
        API_KEY = os.getenv("MAINNET_API_KEY")
        API_SECRET = os.getenv("MAINNET_API_SECRET")
    SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
    BUY_USDT_PER_TRADE = float(os.getenv("BUY_USDT_PER_TRADE", 20))
    MIN_USDT_BAL = float(os.getenv("MIN_USDT_BAL", 10))
    FAST_SMA = int(os.getenv("FAST_SMA", 9))
    SLOW_SMA = int(os.getenv("SLOW_SMA", 21))
    ATR_LEN = int(os.getenv("ATR_LEN", 14))
    EMA_HTF = int(os.getenv("EMA_HTF", 200))
    # Tight SL/TP for quick profit
    ATR_SL_MULT = float(os.getenv("ATR_SL_MULT", 1.0))
    ATR_TP_MULT = float(os.getenv("ATR_TP_MULT", 1.25))

    # Trailing/profit-close config
    TRAIL_ACTIVATE_ATR = float(os.getenv("TRAIL_ACTIVATE_ATR", 0.6))  # Activate trailing after +0.6 ATR move
    TRAIL_OFFSET_ATR = float(os.getenv("TRAIL_OFFSET_ATR", 0.5))      # Trailing stop distance
    MIN_PROFIT_TO_CLOSE = float(os.getenv("MIN_PROFIT_TO_CLOSE", 0.0005))  # ~5bps, covers fees
    POLL_SECONDS = int(os.getenv("POLL_SECONDS", 5))
    # If AI predicts with probability >= this, allow overriding strict EMA location checks
    AI_OVERRIDE_PROB = float(os.getenv("AI_OVERRIDE_PROB", 0.7))
    # Default minimal AI weight used when auto-tune suggests a lower weight. Raise to
    # increase AI contribution in weighted consensus (0..1)
    DEFAULT_AI_WEIGHT = float(os.getenv("DEFAULT_AI_WEIGHT", 0.4))
    # If AI is confident above this (but below the hard override), allow AI-only signals
    AI_MIN_CONFIDENCE_OVERRIDE = float(os.getenv("AI_MIN_CONFIDENCE_OVERRIDE", 0.75))
    # Number of indicator confirmations required (sma,rsi,macd)
    INDICATOR_CONFIRM_COUNT = int(os.getenv("INDICATOR_CONFIRM_COUNT", 2))