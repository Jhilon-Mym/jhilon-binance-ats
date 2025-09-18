import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
from dotenv import load_dotenv
load_dotenv()
from src.config import Config
from src.utils import fetch_klines_rest
from src.strategy import apply_strategy, predict_signal

def make_features(df):
    df["returns"] = df["close"].pct_change()
    df["sma_fast"] = df["close"].rolling(9).mean()
    df["sma_slow"] = df["close"].rolling(21).mean()
    df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()
    df["volatility"] = df["returns"].rolling(20).std()
    df["momentum"] = df["close"].diff(4)
    if "rsi" not in df.columns:
        df["rsi"] = (df["close"].rolling(7).mean() - df["close"].rolling(7).min()) / (df["close"].rolling(7).max() - df["close"].rolling(7).min()) * 100
    if "macd" not in df.columns:
        df["macd"] = df["close"].ewm(span=12, adjust=False).mean() - df["close"].ewm(span=26, adjust=False).mean()
    if "macd_signal" not in df.columns:
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    if "atr" not in df.columns:
        df["atr"] = (df["high"] - df["low"]).rolling(14).mean()
    df = df.fillna(0).reset_index(drop=True)
    return df

if __name__ == "__main__":
    print("debug_signal: project root =", os.getcwd())

    df = fetch_klines_rest(symbol="BTCUSDT", interval="5m", limit=600)
    print(f"Loaded rows: {len(df)} (from REST)")

    df = make_features(df)

    print("\nLast row (indicator snapshot):")
    print(df.iloc[-1])

    try:
        ai_signal = predict_signal(df)
    except Exception as e:
        print(f"AI predict_signal exception: {e}")
        ai_signal = {"side": None, "win_prob": 0.0, "reason": f"predict_exception:{e}"}

    # যদি ai_signal None বা error হয়, fallback deterministic signal দিন
    if ai_signal.get("side") is None:
        print("AI signal not found, applying fallback deterministic rule...")
        last = df.iloc[-1]
        sma_fast = float(last.get("sma_fast", 0.0))
        sma_slow = float(last.get("sma_slow", 0.0))
        macd = float(last.get("macd", 0.0))
        macd_signal = float(last.get("macd_signal", 0.0))
        if sma_fast > sma_slow:
            side = "BUY"
            win_prob = 0.55 if macd > macd_signal else 0.45
        else:
            side = "SELL"
            win_prob = 0.55 if macd < macd_signal else 0.45
        ai_signal = {"side": side, "win_prob": win_prob, "reason": "fallback_deterministic"}

    print(f"\nAI predict_signal -> {ai_signal['side']} | win_prob={ai_signal['win_prob']}")
    print(f"Reason: {ai_signal.get('reason','')}")

    history = df.to_dict(orient="records")
    output = apply_strategy(history, ai_signal=ai_signal, threshold=0.7)
    print(f"\napply_strategy -> {output}\n")
    # Write structured debug entry (JSONL) so we can inspect model inputs/outputs later
    try:
        import json
        debug_entry = {
            "timestamp": str(pd.Timestamp.now()),
            "ai_signal": ai_signal,
            "apply_strategy": output,
            "last_row": df.iloc[-1].to_dict()
        }
        jsonl_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model_debug.jsonl'))
        with open(jsonl_path, 'a', encoding='utf-8') as jf:
            jf.write(json.dumps(debug_entry, default=str) + "\n")
        print(f"Wrote debug entry to {jsonl_path}")
        print(json.dumps(debug_entry, indent=2, default=str))
    except Exception as e:
        print(f"Warning: could not write model_debug.jsonl: {e}")
    print("Done.")
