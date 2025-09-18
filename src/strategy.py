import os
import pickle
import math
import numpy as np
import pandas as pd
import json
from datetime import datetime

_MODEL = None


def load_model():
    """Load cached model object from ../models/model.pkl which should be a dict with
    keys: {"model": sklearn_model, "scaler": scaler, "features": [feature_names]}
    Returns None if file missing.
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "model.pkl")
    model_path = os.path.abspath(model_path)

    logger = __import__('logging').getLogger(__name__)
    logger.info(f"Trying to load model from: {model_path}")

    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            _MODEL = pickle.load(f)
        logger.info("Model loaded successfully")
        return _MODEL
    else:
        logger.info("Model file not found at %s", model_path)

    return None


def fallback_signal(df, reason="fallback_deterministic"):
    """Deterministic simple fallback using SMA and MACD sign to produce a weak signal."""
    try:
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
        return {"side": side, "win_prob": round(win_prob, 3), "reason": reason}
    except Exception as e:
        return {"side": None, "win_prob": 0.0, "reason": f"fallback_exception:{e}"}


def _map_proba_to_side(classes, proba):
    """Robustly map model.classes_ + predict_proba row to buy/sell probabilities.
    Expected logical labels are 1 for BUY and -1 for SELL but classes may be str/int/float.
    Returns (side_str, win_prob_float)
    """
    buy_prob = 0.0
    sell_prob = 0.0
    for c, p in zip(classes, proba):
        try:
            # compare by int value when possible
            if int(float(c)) == 1:
                buy_prob = float(p)
            elif int(float(c)) == -1:
                sell_prob = float(p)
        except Exception:
            # fallback to string checks
            try:
                if str(c).lower() in ("1", "buy", "b"):
                    buy_prob = float(p)
                elif str(c).lower() in ("-1", "sell", "s"):
                    sell_prob = float(p)
            except Exception:
                continue
    if buy_prob >= sell_prob:
        return "BUY", buy_prob
    return "SELL", sell_prob


def predict_signal(df):
    """Predict using trained ML model (models/model.pkl). Returns dict {side, win_prob, reason}.

    On any error the function falls back to deterministic `fallback_signal`.
    """
    model_obj = load_model()
    if model_obj is None:
        # try ensemble prototype from root-level pickles (model_rf.pkl / model_xgb.pkl)
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        ensemble_files = [os.path.join(root, 'model_rf.pkl'), os.path.join(root, 'model_xgb.pkl')]
        members = []
        for p in ensemble_files:
            if os.path.exists(p):
                try:
                    with open(p, 'rb') as f:
                        members.append(pickle.load(f))
                except Exception:
                    continue
        if members:
            # create a lightweight ensemble descriptor
            model_obj = {'ensemble': True, 'members': members, 'features': None}
        else:
            return fallback_signal(df, reason="no_model")
    try:
        # support single model dict or raw estimator
        if isinstance(model_obj, dict) and model_obj.get('ensemble'):
            ensemble_members = model_obj.get('members', [])
            features = list(model_obj.get('features') or [])
        else:
            ensemble_members = None
            if isinstance(model_obj, dict):
                model = model_obj.get('model')
                scaler = model_obj.get('scaler')
                features = list(model_obj.get('features', []))
            else:
                model = model_obj
                scaler = None
                features = []

        # default feature set if features not provided
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

        # Ensure basic technical features exist; compute them if missing.
        try:
            if 'close' in df.columns:
                # returns, SMA/EMA, volatility, momentum
                if 'returns' not in df.columns:
                    df['returns'] = df['close'].pct_change()
                if 'sma_fast' not in df.columns:
                    df['sma_fast'] = df['close'].rolling(9).mean()
                if 'sma_slow' not in df.columns:
                    df['sma_slow'] = df['close'].rolling(21).mean()
                if 'ema_200' not in df.columns:
                    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
                if 'volatility' not in df.columns:
                    df['volatility'] = df['returns'].rolling(20).std()
                if 'momentum' not in df.columns:
                    df['momentum'] = df['close'].diff(4)
                # RSI (simple Wilder-like average)
                if 'rsi' not in df.columns:
                    delta = df['close'].diff()
                    gain = delta.clip(lower=0).rolling(14).mean()
                    loss = (-delta.clip(upper=0)).rolling(14).mean()
                    rs = gain / loss.replace({0: np.nan})
                    df['rsi'] = 100 - 100 / (1 + rs)
                # MACD
                if 'macd' not in df.columns:
                    ema12 = df['close'].ewm(span=12, adjust=False).mean()
                    ema26 = df['close'].ewm(span=26, adjust=False).mean()
                    df['macd'] = ema12 - ema26
                if 'macd_signal' not in df.columns:
                    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
                # ATR (approx)
                if 'atr' not in df.columns:
                    if 'high' in df.columns and 'low' in df.columns:
                        df['atr'] = (df['high'] - df['low']).rolling(14).mean()
                    else:
                        df['atr'] = (df['close'].diff().abs()).rolling(14).mean()
                # RSI lags
                if 'rsi' in df.columns:
                    df['rsi_lag1'] = df['rsi'].shift(1)
                    df['rsi_lag2'] = df['rsi'].shift(2)
                    df['rsi_diff1'] = (df['rsi'] - df['rsi_lag1']).fillna(0)
        except Exception:
            # best-effort, proceed even if computation fails
            pass

        # Ensure all expected features exist as columns; use column-wise forward/backfill
        for f in features:
            if f not in df.columns:
                df[f] = np.nan

        # Fill along the column so each row keeps its historical values (avoid making all rows identical)
        try:
            df[features] = df[features].ffill().bfill().fillna(0)
        except Exception:
            # fallback to per-row fill if something unexpected happens
            df[features] = df[features].fillna(0)

        # Build X from the last row (1 x n_features)
        X = df[features].iloc[[-1]].values
        X_scaled = scaler.transform(X) if ('scaler' in locals() and scaler is not None) else X
        X_scaled = scaler.transform(X) if ('scaler' in locals() and scaler is not None) else X

        # if ensemble_members present, compute ensemble average of probs
        if ensemble_members:
            buy_probs = []
            sell_probs = []
            member_logs = []
            for mobj in ensemble_members:
                try:
                    mem_model = mobj['model'] if isinstance(mobj, dict) and 'model' in mobj else mobj
                    mem_scaler = mobj.get('scaler') if isinstance(mobj, dict) else None
                    Xm = X
                    Xm_scaled = mem_scaler.transform(Xm) if (mem_scaler is not None) else Xm
                    if hasattr(mem_model, 'predict_proba'):
                        classes_m = list(getattr(mem_model, 'classes_', []))
                        proba_m = mem_model.predict_proba(Xm_scaled)[0]
                        side_m, win_prob_m = _map_proba_to_side(classes_m, proba_m)
                        member_logs.append({'type': type(mem_model).__name__, 'classes': classes_m, 'proba': list(map(float, proba_m)), 'side': side_m, 'win_prob': win_prob_m})
                        buy_probs.append(win_prob_m if side_m == 'BUY' else 0.0)
                        sell_probs.append(win_prob_m if side_m == 'SELL' else 0.0)
                    elif hasattr(mem_model, 'decision_function'):
                        d = float(mem_model.decision_function(Xm_scaled)[0])
                        prob_m = 1 / (1 + math.exp(-d))
                        side_m = 'BUY' if prob_m >= 0.5 else 'SELL'
                        win_prob_m = prob_m if side_m == 'BUY' else 1 - prob_m
                        member_logs.append({'type': type(mem_model).__name__, 'classes': None, 'proba': None, 'side': side_m, 'win_prob': win_prob_m})
                        buy_probs.append(win_prob_m if side_m == 'BUY' else 0.0)
                        sell_probs.append(win_prob_m if side_m == 'SELL' else 0.0)
                except Exception as e:
                    member_logs.append({'error': str(e)})
                    continue

            avg_buy = float(np.mean(buy_probs)) if buy_probs else 0.0
            avg_sell = float(np.mean(sell_probs)) if sell_probs else 0.0
            side = 'BUY' if avg_buy >= avg_sell else 'SELL'
            win_prob = avg_buy if side == 'BUY' else avg_sell
            # write structured JSONL log
            try:
                logrec = {
                    'timestamp': datetime.utcnow().isoformat() + 'Z',
                    'ensemble': True,
                    'members': member_logs,
                    'features': features,
                    'X_shape': getattr(X, 'shape', None),
                    'X_scaled_shape': getattr(X_scaled, 'shape', None),
                    'side': side,
                    'win_prob': win_prob,
                    'reason': 'ensemble'
                }
                with open('model_debug.jsonl', 'a', encoding='utf-8') as jf:
                    jf.write(json.dumps(logrec) + '\n')
            except Exception:
                pass

            return {'side': side, 'win_prob': round(float(win_prob), 3), 'reason': 'ensemble'}

        if hasattr(model, "predict_proba"):
            classes = list(getattr(model, "classes_", []))
            proba = model.predict_proba(X_scaled)[0]
            side, win_prob = _map_proba_to_side(classes, proba)
            # detailed debug logging for future troubleshooting
            try:
                with open("model_debug.txt", "a", encoding="utf-8") as dbg:
                    dbg.write("--- model_debug ---\n")
                    dbg.write(f"model_type: {type(model)!r}\n")
                    dbg.write(f"classes: {classes}\n")
                    dbg.write(f"proba: {list(map(float, proba))}\n")
                    dbg.write(f"features: {features}\n")
                    if 'scaler' in locals() and scaler is not None:
                        # attempt to log scaler attributes safely
                        try:
                            dbg.write(f"scaler_type: {type(scaler)!r}\n")
                            # log expected input dim if available
                            if hasattr(scaler, 'n_features_in_'):
                                dbg.write(f"scaler.n_features_in_: {scaler.n_features_in_}\n")
                        except Exception:
                            dbg.write("scaler: <unavailable details>\n")
                    dbg.write(f"X.shape: {getattr(X, 'shape', None)}\n")
                    try:
                        dbg.write(f"X_scaled.shape: {getattr(X_scaled, 'shape', None)}\n")
                    except Exception:
                        dbg.write("X_scaled: <error reading shape>\n")
            except Exception:
                pass
            # structured jsonl log
            try:
                # include small, safe serialization of X and X_scaled for debugging
                X_ser = None
                Xs_ser = None
                try:
                    X_ser = [float(x) for x in X.flatten().tolist()]
                except Exception:
                    X_ser = None
                try:
                    Xs_ser = [float(x) for x in X_scaled.flatten().tolist()] if X_scaled is not None else None
                except Exception:
                    Xs_ser = None

                logrec = {
                    'timestamp': datetime.utcnow().isoformat() + 'Z',
                    'ensemble': False,
                    'model_type': type(model).__name__,
                    'classes': classes,
                    'proba': list(map(float, proba)),
                    'features': features,
                    'X_shape': getattr(X, 'shape', None),
                    'X_scaled_shape': getattr(X_scaled, 'shape', None),
                    'X': X_ser,
                    'X_scaled': Xs_ser,
                    'side': side,
                    'win_prob': win_prob,
                    'reason': 'model_proba'
                }
                with open('model_debug.jsonl', 'a', encoding='utf-8') as jf:
                    jf.write(json.dumps(logrec) + '\n')
            except Exception:
                pass
            return {"side": side, "win_prob": round(float(win_prob), 3), "reason": "model_proba"}
        elif hasattr(model, "decision_function"):
            d = float(model.decision_function(X_scaled)[0])
            prob = 1 / (1 + math.exp(-d))
            side = "BUY" if prob >= 0.5 else "SELL"
            win_prob = prob if side == "BUY" else 1 - prob
            return {"side": side, "win_prob": round(float(win_prob), 3), "reason": "model_decision"}
        else:
            return fallback_signal(df, reason="model_no_predict")

    except Exception as e:
        return fallback_signal(df, reason=f"predict_error:{e}")


def check_indicators(history, side):
    """Simple indicator checks using SMA and EMA200 to confirm AI signal direction.

    history: list-of-dicts with key 'close' at minimum.
    side: "BUY" or "SELL"
    """
    closes = pd.Series([h.get("close", 0.0) for h in history])
    # protect against too-short history
    if len(closes) < 21:
        return False

    fast = closes.rolling(9).mean().iloc[-1]
    slow = closes.rolling(21).mean().iloc[-1]
    ema_htf = closes.ewm(span=200, adjust=False).mean().iloc[-1]

    # require RSI match AND at least one other confirmation (SMA crossover / EMA200 / MACD)
    last = history[-1] if hasattr(history, '__len__') else {}
    try:
        rsi = float(last.get('rsi', float('nan')))
    except Exception:
        rsi = float('nan')

    # Build confirmations from available indicators (RSI optional)
    confirms = 0

    # RSI confirmation if available: oversold for BUY (<=30), overbought for SELL (>=70)
    try:
        if not math.isnan(rsi):
            if side == 'BUY' and rsi <= 30:
                confirms += 1
            if side == 'SELL' and rsi >= 70:
                confirms += 1
    except Exception:
        pass

    # SMA crossover
    try:
        if side == 'BUY' and fast > slow:
            confirms += 1
        if side == 'SELL' and fast < slow:
            confirms += 1
    except Exception:
        pass

    # EMA 200 trend confirmation
    try:
        if side == 'BUY' and closes.iloc[-1] > ema_htf:
            confirms += 1
        if side == 'SELL' and closes.iloc[-1] < ema_htf:
            confirms += 1
    except Exception:
        pass

    # MACD confirmation if present in last
    try:
        macd = float(last.get('macd', float('nan')))
        macd_signal = float(last.get('macd_signal', float('nan')))
        if not math.isnan(macd) and not math.isnan(macd_signal):
            if side == 'BUY' and macd > macd_signal:
                confirms += 1
            if side == 'SELL' and macd < macd_signal:
                confirms += 1
    except Exception:
        pass

    # Require at least two confirmations among the available indicators
    return bool(confirms >= 2)


def indicator_majority(history):
    """Return indicator-majority side: 'BUY', 'SELL', or None if no clear majority.

    We count confirmations for BUY and SELL separately (using same rules as check_indicators)
    and return the side with >=2 confirms. If neither side has >=2, return None.
    """
    closes = pd.Series([h.get("close", 0.0) for h in history])
    if len(closes) < 21:
        return None
    fast = closes.rolling(9).mean().iloc[-1]
    slow = closes.rolling(21).mean().iloc[-1]
    ema_htf = closes.ewm(span=200, adjust=False).mean().iloc[-1]

    last = history[-1] if hasattr(history, '__len__') else {}
    try:
        rsi = float(last.get('rsi', float('nan')))
    except Exception:
        rsi = float('nan')

    buy_conf = 0
    sell_conf = 0
    # RSI
    try:
        if not math.isnan(rsi):
            if rsi <= 30:
                buy_conf += 1
            if rsi >= 70:
                sell_conf += 1
    except Exception:
        pass
    # SMA crossover
    try:
        if fast > slow:
            buy_conf += 1
        if fast < slow:
            sell_conf += 1
    except Exception:
        pass
    # EMA200
    try:
        if closes.iloc[-1] > ema_htf:
            buy_conf += 1
        if closes.iloc[-1] < ema_htf:
            sell_conf += 1
    except Exception:
        pass
    # MACD
    try:
        macd = float(last.get('macd', float('nan')))
        macd_signal = float(last.get('macd_signal', float('nan')))
        if not math.isnan(macd) and not math.isnan(macd_signal):
            if macd > macd_signal:
                buy_conf += 1
            if macd < macd_signal:
                sell_conf += 1
    except Exception:
        pass

    if buy_conf >= 2 and buy_conf > sell_conf:
        return 'BUY'
    if sell_conf >= 2 and sell_conf > buy_conf:
        return 'SELL'
    return None


def _indicator_score_from_row(last_row):
    """Compute a normalized indicator score from a single last_row dict.

    Returns (ind_score_norm, ind_major) where ind_score_norm in [-1,1]
    positive => BUY bias, negative => SELL bias. ind_major is 'BUY'/'SELL'/None
    """
    try:
        # read fields (they may be strings)
        close = float(last_row.get('close', 0.0))
        sma_fast = float(last_row.get('sma_fast', float('nan')))
        sma_slow = float(last_row.get('sma_slow', float('nan')))
        ema_200 = float(last_row.get('ema_200', float('nan')))
        try:
            rsi = float(last_row.get('rsi', float('nan')))
        except Exception:
            rsi = float('nan')
        try:
            macd = float(last_row.get('macd', float('nan')))
            macd_signal = float(last_row.get('macd_signal', float('nan')))
        except Exception:
            macd = macd_signal = float('nan')
    except Exception:
        return 0.0, None

    buy_conf = 0
    sell_conf = 0
    # RSI
    try:
        if not math.isnan(rsi):
            if rsi <= 30:
                buy_conf += 1
            if rsi >= 70:
                sell_conf += 1
    except Exception:
        pass
    # SMA
    try:
        if not math.isnan(sma_fast) and not math.isnan(sma_slow):
            if sma_fast > sma_slow:
                buy_conf += 1
            if sma_fast < sma_slow:
                sell_conf += 1
    except Exception:
        pass
    # EMA200
    try:
        if not math.isnan(ema_200):
            if close > ema_200:
                buy_conf += 1
            if close < ema_200:
                sell_conf += 1
    except Exception:
        pass
    # MACD
    try:
        if not math.isnan(macd) and not math.isnan(macd_signal):
            if macd > macd_signal:
                buy_conf += 1
            if macd < macd_signal:
                sell_conf += 1
    except Exception:
        pass

    max_conf = 4.0
    raw = float(buy_conf - sell_conf) / max_conf
    ind_major = 'BUY' if buy_conf >= 2 and buy_conf > sell_conf else ('SELL' if sell_conf >= 2 and sell_conf > buy_conf else None)
    return max(-1.0, min(1.0, raw)), ind_major


def _auto_tune_weights(history_path='model_debug.jsonl', max_lines=500):
    """Lightweight auto-tune of AI vs indicator weight based on recent debug logs.

    Strategy: for a small grid of ai_weight values, compute how often the combined
    soft-vote matches indicator majority (on entries where an indicator majority exists).
    Return best ai_weight (0..1). If insufficient data, return 0.5.
    """
    if not os.path.exists(history_path):
        return 0.5

    samples = []
    try:
        with open(history_path, 'r', encoding='utf-8') as fh:
            for i, line in enumerate(reversed(fh.readlines())):
                if i >= max_lines:
                    break
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                # expect 'ai_signal' and possibly 'last_row'
                ai = rec.get('ai_signal') or rec.get('ai')
                last = rec.get('last_row') or rec.get('features')
                if not ai or not last:
                    continue
                win_prob = float(ai.get('win_prob', 0.0))
                side = ai.get('side')
                ind_score, ind_major = _indicator_score_from_row(last)
                if ind_major is None:
                    continue
                samples.append({'win_prob': win_prob, 'ai_side': side, 'ind_major': ind_major, 'ind_score': ind_score})
    except Exception:
        return 0.5

    if not samples:
        return 0.5

    best_w = 0.5
    best_match = -1
    # grid search for ai weight
    for w in [i/10.0 for i in range(1,10)]:
        matches = 0
        for s in samples:
            ai_score = (s['win_prob'] - 0.5) * 2.0
            combined = w * ai_score + (1.0 - w) * s['ind_score']
            comb_side = 'BUY' if combined >= 0 else 'SELL'
            if comb_side == s['ind_major']:
                matches += 1
        if matches > best_match:
            best_match = matches
            best_w = w

    try:
        from src.config import Config
        default_w = float(getattr(Config, 'DEFAULT_AI_WEIGHT', 0.5))
    except Exception:
        default_w = 0.5
    return max(float(best_w), default_w)


def _auto_tune_threshold(history_path='model_debug.jsonl', max_lines=500):
    """Auto-tune minimal absolute combined_score threshold from recent logs.

    Strategy: compute historical combined scores using auto-tuned w_ai and pick a
    conservative percentile (25th) as the minimal confidence required. Falls back
    to 0.15 if insufficient data.
    """
    if not os.path.exists(history_path):
        return 0.15
    try:
        w_ai = _auto_tune_weights(history_path=history_path, max_lines=max_lines)
        scores = []
        with open(history_path, 'r', encoding='utf-8') as fh:
            for i, line in enumerate(reversed(fh.readlines())):
                if i >= max_lines:
                    break
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                ai = rec.get('ai_signal') or rec.get('ai')
                last = rec.get('last_row') or rec.get('features')
                if not ai or not last:
                    continue
                win_prob = float(ai.get('win_prob', 0.0))
                ind_score, _ = _indicator_score_from_row(last)
                ai_score = (win_prob - 0.5) * 2.0
                combined = w_ai * ai_score + (1.0 - w_ai) * ind_score
                scores.append(abs(combined))
        if not scores:
            return 0.15
        import numpy as _np
        thresh = float(_np.percentile(_np.array(scores), 25))
        return max(0.05, min(0.5, thresh))
    except Exception:
        return 0.15


def weighted_consensus(history, ai_signal, tune=True):
    """Return combined decision ('BUY'|'SELL') and combined_score using weighted soft-vote.

    If tune=True, attempt to auto-tune ai weight from recent model_debug.jsonl.
    """
    last = history[-1] if hasattr(history, '__len__') else {}
    ind_score, ind_major = _indicator_score_from_row(last)
    win_prob = float(ai_signal.get('win_prob', 0.0))
    ai_score = (win_prob - 0.5) * 2.0

    if tune:
        tuned = _auto_tune_weights()
        try:
            from src.config import Config
            default_w = float(getattr(Config, 'DEFAULT_AI_WEIGHT', 0.5))
        except Exception:
            default_w = 0.5
        w_ai = max(tuned, default_w)
    else:
        w_ai = 0.5

    combined = w_ai * ai_score + (1.0 - w_ai) * ind_score
    comb_side = 'BUY' if combined >= 0 else 'SELL'
    return {'side': comb_side, 'combined_score': combined, 'w_ai': w_ai, 'ind_major': ind_major}


def apply_strategy(history, ai_signal=None, threshold=0.8):
    """Combine AI signal + indicator filters into concrete order parameters.

    Returns dict: {side, sl, tp, win_prob, reason}
    """
    if len(history) < 200:
        return {"side": None, "sl": None, "tp": None, "win_prob": 0.0, "reason": "not_enough_data"}

    if ai_signal is None or ai_signal.get("side") is None:
        return {"side": None, "sl": None, "tp": None, "win_prob": 0.0, "reason": "no_ai_signal"}

    side = ai_signal["side"]
    win_prob = float(ai_signal.get("win_prob", 0.0))

    # read override threshold from config (default conservative)
    try:
        from src.config import Config
        override_prob = float(getattr(Config, 'AI_OVERRIDE_PROB', 0.9))
    except Exception:
        override_prob = 0.9

    # Use weighted consensus (soft-vote) between AI and indicators
    wc = weighted_consensus(history, ai_signal, tune=True)
    comb_side = wc.get('side')
    combined_score = float(wc.get('combined_score', 0.0))
    w_ai = float(wc.get('w_ai', 0.5))
    ind_major = wc.get('ind_major')

    # tuned minimal absolute combined score required to accept a soft-vote
    tuned_thresh = _auto_tune_threshold()

    # If indicators have a clear majority that contradicts AI, and both AI and combined
    # confidence are below their thresholds, reject; otherwise allow the soft-vote to decide.
    if ind_major is not None and ind_major != side and win_prob < override_prob and abs(combined_score) < tuned_thresh:
        return {"side": None, "sl": None, "tp": None, "win_prob": win_prob, "reason": "indicators_reject"}

    # Accept when either AI has sufficient prob OR combined soft-vote is confident
    if not (win_prob >= threshold or abs(combined_score) >= tuned_thresh):
        return {"side": None, "sl": None, "tp": None, "win_prob": win_prob, "reason": f"low_conf(win_prob<{threshold}, combined<{tuned_thresh:.2f})"}

    final_side = comb_side
    last_row = history[-1]
    last_close = last_row["close"]
    atr = last_row.get("atr")
    try:
        from src.config import Config
        sl_mult = float(getattr(Config, 'ATR_SL_MULT', 1.0))
        tp_mult = float(getattr(Config, 'ATR_TP_MULT', 1.25))
    except Exception:
        sl_mult = 1.0
        tp_mult = 1.25

    if atr is not None and not pd.isna(atr):
        if final_side == "BUY":
            sl = last_close - atr * sl_mult
            tp = last_close + atr * tp_mult
        else:
            sl = last_close + atr * sl_mult
            tp = last_close - atr * tp_mult
    else:
        # fallback to fixed 1% if ATR missing
        sl = last_close * (0.99 if final_side == "BUY" else 1.01)
        tp = last_close * (1.01 if final_side == "BUY" else 0.99)

    # log tuning info
    try:
        logrec = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'event': 'weighted_decision',
            'ai_signal': ai_signal,
            'weighted': {'w_ai': w_ai, 'combined_score': combined_score, 'tuned_thresh': tuned_thresh},
            'ind_major': ind_major,
            'final_side': final_side,
            'history_len': len(history)
        }
        with open('model_debug.jsonl', 'a', encoding='utf-8') as jf:
            jf.write(json.dumps(logrec) + '\n')
    except Exception:
        pass

    # include auxiliary tuning info so the caller can apply live-safety checks
    return {"side": final_side, "sl": sl, "tp": tp, "win_prob": win_prob, "reason": 'weighted_consensus',
            "combined_score": combined_score, "w_ai": w_ai, "ind_major": ind_major, "tuned_thresh": tuned_thresh}

