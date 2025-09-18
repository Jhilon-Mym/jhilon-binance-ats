import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from src.utils import fetch_klines_rest
from src.strategy import load_model, predict_signal

print('Loading klines...')
df = fetch_klines_rest(symbol='BTCUSDT', interval='5m', limit=10)
print(f'Rows: {len(df)}')
model_obj = load_model()
print('Model loaded:', bool(model_obj))

# show last 6 rows features and predict
for i in range(-6, 0):
    sub = df.copy()
    # keep only up to that row to simulate different last_row states
    sub = sub.iloc[:i+1].reset_index(drop=True) if i != -1 else sub
    print('\n--- row idx:', sub.index[-1])
    last = sub.iloc[-1]
    print(last[['timestamp','close']].to_dict())
    try:
        sig = predict_signal(sub)
        print('predict_signal ->', sig)
    except Exception as e:
        print('predict_signal exception:', e)

print('\nDone')
