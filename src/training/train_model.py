import os
import pickle
import numpy as np
import pandas as pd
from binance.client import Client
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from dotenv import load_dotenv
import sys
import subprocess
import traceback

# --------------------------
# Load API keys
# --------------------------
load_dotenv()
api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")
use_testnet = os.getenv("USE_TESTNET", "true").lower() == "true"

client = Client(api_key, api_secret, testnet=use_testnet)


# --------------------------
# Step 1: Fetch historical data
# --------------------------
def fetch_klines(symbol="BTCUSDT", interval="5m", limit=1000, lookback=50000):
    """Fetch historical klines (multiple requests if needed)."""
    all_data = []
    last_end_time = None

    while len(all_data) < lookback:
        klines = client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=limit,
            endTime=last_end_time
        )
        if not klines:
            break
        all_data.extend(klines)
        last_end_time = klines[0][0] - 1  # move backwards
        print(f"Fetched {len(all_data)} rows...", end="\r")

    df = pd.DataFrame(all_data, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","qav","trades","tb_base","tb_quote","ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
    df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# --------------------------
# Step 2: Feature Engineering
# --------------------------
def make_features(df):
    df["returns"] = df["close"].pct_change()
    df["sma_fast"] = df["close"].rolling(9).mean()
    df["sma_slow"] = df["close"].rolling(21).mean()
    df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()
    df["volatility"] = df["returns"].rolling(20).std()
    df["momentum"] = df["close"].diff(4)
    # নতুন: RSI, MACD, MACD_SIGNAL, ATR যোগ করুন (যদি না থাকে)
    if "rsi" not in df.columns:
        # simple rolling-percentile RSI-like proxy (add small eps to avoid div0)
        df["rsi"] = (df["close"].rolling(7).mean() - df["close"].rolling(7).min()) / (
            df["close"].rolling(7).max() - df["close"].rolling(7).min() + 1e-9) * 100
    if "macd" not in df.columns:
        df["macd"] = df["close"].ewm(span=12, adjust=False).mean() - df["close"].ewm(span=26, adjust=False).mean()
    if "macd_signal" not in df.columns:
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    if "atr" not in df.columns:
        df["atr"] = (df["high"] - df["low"]).rolling(14).mean()
    df = df.fillna(0).reset_index(drop=True)
    return df


# --------------------------
# Step 3: Labeling
# --------------------------
def label_data(df, horizon=5, threshold=0.002):
    """
    BUY if future return > +threshold
    SELL if future return < -threshold
    else 0 (ignore)
    """
    df["future_return"] = df["close"].shift(-horizon) / df["close"] - 1
    df["label"] = 0
    df.loc[df["future_return"] > threshold, "label"] = 1   # BUY
    df.loc[df["future_return"] < -threshold, "label"] = -1 # SELL
    df = df.dropna().reset_index(drop=True)
    return df


# --------------------------
# Step 4: Balance Dataset
# --------------------------
def balance_dataset(df):
    buy = df[df["label"] == 1]
    sell = df[df["label"] == -1]

    if len(buy) == 0 or len(sell) == 0:
        raise ValueError("Not enough BUY/SELL samples to balance")

    min_len = min(len(buy), len(sell))
    buy_bal = resample(buy, replace=True, n_samples=min_len, random_state=42)
    sell_bal = resample(sell, replace=True, n_samples=min_len, random_state=42)

    df_bal = pd.concat([buy_bal, sell_bal]).sample(frac=1, random_state=42).reset_index(drop=True)
    return df_bal


# --------------------------
# Step 5: Train Model
# --------------------------
def train_model(df):
    # Add RSI-lag and simple RSI-derived features so model learns RSI patterns
    df["rsi_lag1"] = df["rsi"].shift(1).bfill()
    df["rsi_lag2"] = df["rsi"].shift(2).bfill()
    df["rsi_diff1"] = (df["rsi"] - df["rsi_lag1"]).fillna(0)

    features = [
        "returns","sma_fast","sma_slow","ema_200","volatility","momentum",
        "rsi","rsi_lag1","rsi_lag2","rsi_diff1","macd","macd_signal","atr"
    ]

    X = df[features].values
    y = df["label"].astype(int).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    model.fit(X_scaled, y)

    # Try to train an XGBoost model if available
    xgb_model = None
    try:
        from xgboost import XGBClassifier
    except Exception:
        # attempt to install xgboost in the current python env
        try:
            print("xgboost not found, attempting to install xgboost via pip...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'xgboost'])
            from xgboost import XGBClassifier
        except Exception as e:
            print(f"Could not install xgboost: {e}")
            XGBClassifier = None

    if 'XGBClassifier' in locals() and XGBClassifier is not None:
        try:
            xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=200, max_depth=6, random_state=42)
            # XGBoost expects labels starting at 0; map -1 -> 0, 1 -> 1
            try:
                y_xgb = (y == 1).astype(int)
            except Exception:
                # fallback: map using list comprehension
                y_xgb = np.array([1 if vv == 1 else 0 for vv in y])
            xgb_model.fit(X_scaled, y_xgb)
        except Exception:
            print("XGBoost training failed:")
            traceback.print_exc()
            xgb_model = None
    else:
        xgb_model = None

    return {"model": model, "xgb_model": xgb_model, "scaler": scaler, "features": features}


# --------------------------
# Step 6: Save Model
# --------------------------

def save_model_and_scaler(model, scaler, model_path="models/model.pkl", scaler_path="models/scaler.pkl"):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Model saved to {model_path}")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"✅ Scaler saved to {scaler_path}")


def save_models(obj, rf_path="models/model_rf.pkl", xgb_path="models/model_xgb.pkl", scaler_path="models/scaler.pkl"):
    os.makedirs(os.path.dirname(rf_path), exist_ok=True)
    with open(rf_path, "wb") as f:
        pickle.dump({"model": obj["model"], "features": obj["features"]}, f)
    print(f"✅ RandomForest saved to {rf_path}")
    # Save XGBoost if trained; otherwise write a placeholder file so consumers can detect absence
    if obj.get("xgb_model") is not None:
        with open(xgb_path, "wb") as f:
            pickle.dump({"model": obj["xgb_model"], "features": obj["features"]}, f)
        print(f"✅ XGBoost saved to {xgb_path}")
    else:
        try:
            with open(xgb_path, "wb") as f:
                pickle.dump({"model": None, "features": obj["features"], "error": "xgb_not_trained"}, f)
            print(f"ℹ️ XGBoost not trained; placeholder written to {xgb_path}")
        except Exception as e:
            print(f"Warning: could not write XGBoost placeholder: {e}")
    # always save scaler
    with open(scaler_path, "wb") as f:
        pickle.dump(obj["scaler"], f)
    print(f"✅ Scaler saved to {scaler_path}")


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    # CSV paths
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'klines_BTCUSDT_5m.csv'))
    balanced_csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'klines_BTCUSDT_5m_balanced.csv'))

    # Try to fetch fresh data via REST if API keys are available; otherwise fall back to local CSV
    if api_key and api_secret:
        try:
            print("Fetching data from Binance via REST...")
            df = fetch_klines(lookback=30000)
            print(f"Fetched {len(df)} rows from REST")
            # save/update CSV for reproducibility
            try:
                df.to_csv(csv_path, index=False)
                print(f"Updated CSV at {csv_path}")
            except Exception as e:
                print(f"Warning: could not write CSV: {e}")
        except Exception as e:
            print(f"REST fetch failed: {e}")
            if os.path.exists(csv_path):
                print(f"Falling back to local CSV {csv_path}")
                df = pd.read_csv(csv_path)
                if 'timestamp' in df.columns:
                    try:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                    except Exception:
                        pass
                for c in ('open','high','low','close','volume'):
                    if c in df.columns:
                        df[c] = df[c].astype(float)
                df = df.sort_values('timestamp').reset_index(drop=True)
            else:
                raise
    else:
        # no API keys, use local CSV if present
        if os.path.exists(csv_path):
            print(f"Loading local CSV data from {csv_path}")
            df = pd.read_csv(csv_path)
            if 'timestamp' in df.columns:
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                except Exception:
                    pass
            for c in ('open','high','low','close','volume'):
                if c in df.columns:
                    df[c] = df[c].astype(float)
            df = df.sort_values('timestamp').reset_index(drop=True)
            print(f"Loaded {len(df)} rows from CSV")
        else:
            raise RuntimeError("No API keys and no local CSV available for training")

    df = make_features(df)
    df = label_data(df)

    print(f"Total labeled rows: {len(df)}")
    # Save full feature CSV (with labels) for reproducibility
    try:
        df.to_csv(csv_path, index=False)
        print(f"Saved features+labels CSV to {csv_path}")
    except Exception as e:
        print(f"Warning: could not save feature CSV: {e}")

    # Balance dataset and save balanced CSV used for training
    df_bal = balance_dataset(df)
    try:
        df_bal.to_csv(balanced_csv_path, index=False)
        print(f"Saved balanced CSV to {balanced_csv_path}")
    except Exception as e:
        print(f"Warning: could not save balanced CSV: {e}")
    print(f"Balanced dataset size: {len(df_bal)}")

    obj = train_model(df_bal)
    # save primary model as dict {model, scaler, features} so runtime can load scaler/features
    os.makedirs(os.path.dirname("models/model.pkl"), exist_ok=True)
    with open("models/model.pkl", "wb") as f:
        pickle.dump({"model": obj["model"], "scaler": obj["scaler"], "features": obj["features"]}, f)
    print(f"✅ Primary model (dict) saved to models/model.pkl")
    # save RF and XGB models separately for ensemble (save_models will also write the scaler)
    save_models(obj, rf_path="models/model_rf.pkl", xgb_path="models/model_xgb.pkl", scaler_path="models/scaler.pkl")

    # Validate that the scaler inside models/model.pkl matches models/scaler.pkl
    try:
        with open("models/model.pkl", "rb") as f:
            primary = pickle.load(f)
        with open("models/scaler.pkl", "rb") as f:
            scaler_file = pickle.load(f)
        primary_scaler = primary.get("scaler") if isinstance(primary, dict) else None
        if primary_scaler is None:
            print("⚠️ Primary model dict does not contain a scaler to validate.")
        else:
            # compare common ndarray attributes if present
            attrs = ["mean_", "scale_", "var_"]
            comparable = True
            for a in attrs:
                if not (hasattr(primary_scaler, a) and hasattr(scaler_file, a)):
                    comparable = False
                    break
            if comparable:
                all_ok = True
                for a in attrs:
                    v1 = getattr(primary_scaler, a)
                    v2 = getattr(scaler_file, a)
                    if not np.allclose(v1, v2, equal_nan=True):
                        all_ok = False
                        break
                if all_ok:
                    print("✅ Scaler in models/scaler.pkl matches scaler inside models/model.pkl")
                else:
                    print("⚠️ MISMATCH: scaler in separate file differs from scaler inside model.pkl")
            else:
                # fallback: compare pickles by bytes length as a weak check
                try:
                    import io
                    buf1 = io.BytesIO()
                    buf2 = io.BytesIO()
                    pickle.dump(primary_scaler, buf1)
                    pickle.dump(scaler_file, buf2)
                    if buf1.getbuffer().nbytes == buf2.getbuffer().nbytes:
                        print("ℹ️ Scaler files look similar by size (weak check).")
                    else:
                        print("⚠️ Scaler files differ by size (weak check); consider manual inspection.")
                except Exception:
                    print("ℹ️ Could not perform scaler binary comparison; please validate manually if needed.")
    except Exception as e:
        print(f"Warning: scaler validation failed: {e}")
    print("🎯 Training complete.")
