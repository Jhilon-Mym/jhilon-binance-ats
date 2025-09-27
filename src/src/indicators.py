"""
Indicators Library
==================
- Simple moving average (SMA)
- Exponential moving average (EMA)
- Average True Range (ATR)
- RSI
- MACD
"""

import pandas as pd
import ta
from src.config import Config

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add SMA, EMA, RSI, MACD, ATR, and advanced features to dataframe (ATR window from config)"""
    df["sma_fast"] = ta.trend.SMAIndicator(df["close"], window=Config.FAST_SMA).sma_indicator()
    df["sma_slow"] = ta.trend.SMAIndicator(df["close"], window=Config.SLOW_SMA).sma_indicator()
    df["ema_200"] = ta.trend.EMAIndicator(df["close"], window=Config.EMA_HTF).ema_indicator()

    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=7).rsi()

    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    atr = ta.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=Config.ATR_LEN
    )
    df["atr"] = atr.average_true_range()

    # Advanced live feature engineering
    df["volatility"] = df["close"].rolling(10).std()
    df["momentum"] = df["close"].diff(3)
    df["returns"] = df["close"].pct_change(1)
    df["volume_sma"] = df["volume"].rolling(5).mean()
    df["high_low_spread"] = df["high"] - df["low"]

    return df