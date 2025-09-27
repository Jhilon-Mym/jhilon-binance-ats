
# training/train_hybrid.py — fetch REST klines -> build features -> train simple model -> save models/model.pkl
import os, argparse, pickle
import pandas as pd
from binance.client import Client

def fetch_klines(client, symbol, interval, limit):
    kl = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    cols = ["open_time","open","high","low","close","volume","close_time","qav","num_trades","tb_base","tb_quote","ignore"]
    df = pd.DataFrame(kl, columns=cols)
    df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
    df['ts'] = pd.to_datetime(df['open_time'], unit='ms')
    return df

def sma(x, n): return x.rolling(n).mean()
def ema(x, n): return x.ewm(span=n, adjust=False).mean()

def build_features(df):
    df['close'] = df['close'].astype(float)
    df['ema20'] = ema(df['close'], 20)
    df['ema50'] = ema(df['close'], 50)
    df['ema200'] = ema(df['close'], 200)
    df['sma20'] = sma(df['close'], 20)
    # label: 1 if next close is higher than current close by > 0.2%
    df['y'] = (df['close'].shift(-1) >= df['close'] * 1.002).astype(int)
    feats = df[['close','ema20','ema50','ema200','sma20','y']].dropna().copy()
    return feats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbol', default=os.getenv('SYMBOL','BTCUSDT'))
    ap.add_argument('--interval', default=os.getenv('INTERVAL','5m'))
    ap.add_argument('--limit', type=int, default=2000)
    ap.add_argument('--out', default='models/model.pkl')
    args = ap.parse_args()

    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    use_testnet = os.getenv('USE_TESTNET','true').lower()=='true'
    client = Client(api_key, api_secret, testnet=use_testnet)

    df = fetch_klines(client, args.symbol, args.interval, args.limit)
    feats = build_features(df)
    X = feats[['close','ema20','ema50','ema200','sma20']].values
    y = feats['y'].values

    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'wb') as f:
        pickle.dump(model, f)
    print(f'Saved model -> {args.out} ({len(X)} samples)')

if __name__ == '__main__':
    main()
