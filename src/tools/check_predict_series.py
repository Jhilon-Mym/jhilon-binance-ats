import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from src.utils import fetch_klines_rest
from src.strategy import predict_signal, load_model

print('Loading 300 klines...')
df = fetch_klines_rest(symbol='BTCUSDT', interval='5m', limit=300)
print('rows=', len(df))
model = load_model()
print('model loaded:', bool(model))

start = max(50, len(df)-50)
for k in range(start, len(df)):
    sub = df.iloc[:k+1].reset_index(drop=True)
    sig = predict_signal(sub)
    print(f'idx={k} time={sub.iloc[-1]["timestamp"]} close={sub.iloc[-1]["close"]} -> side={sig.get("side")} win_prob={sig.get("win_prob")} reason={sig.get("reason")}')
