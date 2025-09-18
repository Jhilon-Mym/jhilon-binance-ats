import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
from src.strategy import load_model
from src.utils import fetch_klines_rest

print('Loading klines...')
df = fetch_klines_rest(symbol='BTCUSDT', interval='5m', limit=10)
print('rows=', len(df))
model_obj = load_model()
if not model_obj:
    print('No model loaded; abort')
    sys.exit(1)

# support dict model with scaler & features
if isinstance(model_obj, dict):
    model = model_obj.get('model')
    scaler = model_obj.get('scaler')
    features = list(model_obj.get('features') or [])
else:
    model = model_obj
    scaler = None
    features = []

if not features:
    features = [
        "returns",
        "sma_fast",
        "sma_slow",
        "ema_200",
        "volatility",
        "momentum",
        "rsi",
        "macd",
        "macd_signal",
        "atr",
    ]

print('features len:', len(features))

for i in range(-6, 0):
    sub = df.copy()
    # keep up to that row
    sub = sub.iloc[:i+1].reset_index(drop=True) if i != -1 else sub
    last_idx = sub.index[-1]
    print('\n--- row idx:', last_idx, 'timestamp=', sub.iloc[-1]['timestamp'])
    # ensure rsi_lag etc like in predict_signal
    if 'rsi' in sub.columns:
        if 'rsi_lag1' not in sub.columns:
            sub['rsi_lag1'] = sub['rsi'].shift(1).bfill()
        if 'rsi_lag2' not in sub.columns:
            sub['rsi_lag2'] = sub['rsi'].shift(2).bfill()
        if 'rsi_diff1' not in sub.columns:
            sub['rsi_diff1'] = (sub['rsi'] - sub['rsi_lag1']).fillna(0)
    for f in features:
        if f not in sub.columns:
            sub[f] = float('nan')
    X = sub[features].iloc[-1:]
    try:
        X = X.ffill().bfill().fillna(0).values
    except Exception:
        X = X.fillna(0).values
    print('raw X:', X.tolist())
    if scaler is not None:
        try:
            Xs = scaler.transform(X)
            print('scaled X:', Xs.tolist())
        except Exception as e:
            print('scaler.transform error:', e)
            Xs = None
    else:
        Xs = X
    if hasattr(model, 'predict_proba'):
        try:
            proba = model.predict_proba(Xs if Xs is not None else X)[0]
            print('classes:', list(getattr(model, 'classes_', [])))
            print('proba:', proba.tolist())
        except Exception as e:
            print('predict_proba error:', e)
    elif hasattr(model, 'decision_function'):
        try:
            d = float(model.decision_function(Xs if Xs is not None else X)[0])
            prob = 1/(1+np.exp(-d))
            print('decision d=', d, 'prob=', prob)
        except Exception as e:
            print('decision_function error:', e)

print('\ndone')
