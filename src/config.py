import os
from dotenv import load_dotenv

load_dotenv()

class _EnvClassProperty:
    def __init__(self, key, cast=str, default=None):
        self.key = key
        self.cast = cast
        self.default = default

    def __get__(self, obj, owner):
        raw = os.getenv(self.key, None)
        if raw is None:
            return self.default
        try:
            return self.cast(raw)
        except Exception:
            try:
                return self.cast(str(self.default))
            except Exception:
                return self.default


class Config:
    # Access environment-configured values dynamically so updates to os.environ
    # (or via the /api/config endpoint which updates .env and os.environ) are
    # reflected immediately without forcing a module reload.
    SIGNAL_THRESHOLD = _EnvClassProperty('SIGNAL_THRESHOLD', cast=float, default=0.8)
    USE_TESTNET = _EnvClassProperty('USE_TESTNET', cast=lambda v: str(v).lower() == 'true', default=True)

    @classmethod
    def API_KEY(cls):
        return os.getenv('TESTNET_API_KEY' if cls.USE_TESTNET else 'MAINNET_API_KEY')

    @classmethod
    def API_SECRET(cls):
        return os.getenv('TESTNET_API_SECRET' if cls.USE_TESTNET else 'MAINNET_API_SECRET')

    SYMBOL = _EnvClassProperty('SYMBOL', cast=str, default='BTCUSDT')
    HISTORY_PRELOAD = _EnvClassProperty('HISTORY_PRELOAD', cast=int, default=600)
    BUY_USDT_PER_TRADE = _EnvClassProperty('BUY_USDT_PER_TRADE', cast=float, default=20.0)
    MIN_USDT_BAL = _EnvClassProperty('MIN_USDT_BAL', cast=float, default=10.0)
    FAST_SMA = _EnvClassProperty('FAST_SMA', cast=int, default=9)
    SLOW_SMA = _EnvClassProperty('SLOW_SMA', cast=int, default=21)
    ATR_LEN = _EnvClassProperty('ATR_LEN', cast=int, default=14)
    EMA_HTF = _EnvClassProperty('EMA_HTF', cast=int, default=200)
    ATR_SL_MULT = _EnvClassProperty('ATR_SL_MULT', cast=float, default=1.0)
    ATR_TP_MULT = _EnvClassProperty('ATR_TP_MULT', cast=float, default=1.25)

    TRAIL_ACTIVATE_ATR = _EnvClassProperty('TRAIL_ACTIVATE_ATR', cast=float, default=0.6)
    TRAIL_OFFSET_ATR = _EnvClassProperty('TRAIL_OFFSET_ATR', cast=float, default=0.5)
    MIN_PROFIT_TO_CLOSE = _EnvClassProperty('MIN_PROFIT_TO_CLOSE', cast=float, default=0.0005)
    POLL_SECONDS = _EnvClassProperty('POLL_SECONDS', cast=int, default=5)
    AI_OVERRIDE_PROB = _EnvClassProperty('AI_OVERRIDE_PROB', cast=float, default=0.7)
    DEFAULT_AI_WEIGHT = _EnvClassProperty('DEFAULT_AI_WEIGHT', cast=float, default=0.4)
    AI_MIN_CONFIDENCE_OVERRIDE = _EnvClassProperty('AI_MIN_CONFIDENCE_OVERRIDE', cast=float, default=0.75)
    INDICATOR_CONFIRM_COUNT = _EnvClassProperty('INDICATOR_CONFIRM_COUNT', cast=int, default=2)