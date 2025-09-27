import os, sys, traceback
import pandas as pd

# Ensure repo root is on sys.path so `from src...` works when run from tools/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

print('PWD:', os.getcwd())
print('Repo root:', ROOT)
print('Model exists:', os.path.exists(os.path.join(ROOT,'model_xgb.pkl')))
print('Scaler exists:', os.path.exists(os.path.join(ROOT,'scaler.pkl')))

try:
    from src.hybrid_ai import predict_signal
    from src.indicators import add_indicators

    df = pd.read_csv('klines_BTCUSDT_5m.csv')
    df = df[['timestamp','open','high','low','close','volume']].copy()
    for c in ['open','high','low','close','volume']:
        df[c] = df[c].astype(float)
    df = add_indicators(df).dropna().reset_index(drop=True)
    print('Indicator rows:', len(df))
    try:
        sig = predict_signal(df)
        print('predict_signal ->', sig)
    except Exception:
        print('predict_signal raised:')
        traceback.print_exc()
except Exception:
    print('Import or setup failed:')
    traceback.print_exc()
