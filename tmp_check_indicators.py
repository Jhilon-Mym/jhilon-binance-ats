import os, sys
sys.path.insert(0, os.path.abspath('.'))
from src.utils import fetch_klines_rest
from src.strategy import check_indicators
import pandas as pd

# load data
df = fetch_klines_rest(symbol='BTCUSDT', interval='5m', limit=600)
# add features same as debug_signal
from tools.debug_signal import make_features

df = make_features(df)
history = df.to_dict(orient='records')
print('rows', len(history))
last = history[-1]
print('last rsi, sma_fast, sma_slow, ema_200, macd, macd_signal')
print(last.get('rsi'), last.get('sma_fast'), last.get('sma_slow'), last.get('ema_200'), last.get('macd'), last.get('macd_signal'))
print('check_indicators for BUY:', check_indicators(history, 'BUY'))
print('check_indicators for SELL:', check_indicators(history, 'SELL'))
