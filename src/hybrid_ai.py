import os
import numpy as np
import pandas as pd
import logging
try:
	import joblib
except Exception:
	joblib = None

# xgboost is optional; detect availability without breaking joblib import
try:
	import xgboost  # type: ignore
	xgb_available = True
except Exception:
	xgb_available = False

MODEL_PATH = "model_xgb.pkl"
SCALER_PATH = "scaler.pkl"

# Weights to blend model probability with indicator agreement score.
# Higher model_weight favors the ML model; indicator_weight gives boost when indicators agree.
MODEL_WEIGHT = float(os.getenv('HYBRID_MODEL_WEIGHT', 0.85))
INDICATOR_WEIGHT = float(os.getenv('HYBRID_INDICATOR_WEIGHT', 0.15))


def predict_signal(df):
	# Use last row features for prediction
	features = [
		"sma_fast", "sma_slow", "ema_200", "rsi", "macd", "macd_signal", "atr",
		"volatility", "momentum", "returns", "volume_sma", "high_low_spread"
	]
	# Ensure features present
	if not all(f in df.columns for f in features):
		logging.debug('predict_signal: missing features, falling back to indicator rule')
		# deterministic indicator fallback (SMA+MACD)
		last = df.iloc[-1]
		sma_fast = float(last.get('sma_fast', 0.0))
		sma_slow = float(last.get('sma_slow', 0.0))
		macd = float(last.get('macd', 0.0))
		macd_signal = float(last.get('macd_signal', 0.0))
		if sma_fast > sma_slow:
			conf = 0.6 if macd > macd_signal else 0.45
			return 'BUY', float(conf)
		else:
			conf = 0.6 if macd < macd_signal else 0.45
			return 'SELL', float(conf)

	# If joblib and artifacts exist, use models when possible.
	if joblib is not None and os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
		try:
			# load scaler and calibrated/base models if available
			scaler = joblib.load(SCALER_PATH)
			calibrated = None
			base = None
			try:
				calibrated = joblib.load(MODEL_PATH)
			except Exception:
				calibrated = None
			# attempt to also load a separate XGB artifact if present (backwards compat)
			xgb_path = 'model_xgb.pkl'
			if os.path.exists(xgb_path):
				try:
					base = joblib.load(xgb_path)
				except Exception:
					base = None

				X = pd.DataFrame([df[features].iloc[-1]], columns=features)
				Xs = scaler.transform(X)

				# If we have a calibrated model, prefer it. If we also have base XGB, ensemble lightly.
				if calibrated is not None and hasattr(calibrated, 'predict_proba'):
					cal_proba = calibrated.predict_proba(Xs)[0]
					# Try to get xgb probs to ensemble; if not available just use calibrated
					if base is not None and hasattr(base, 'predict_proba'):
						try:
							base_proba = base.predict_proba(Xs)[0]
							# weighted ensemble: prefer calibrated (80%) and base (20%)
							ensemble = 0.8 * cal_proba + 0.2 * base_proba
						except Exception:
							ensemble = cal_proba
					else:
						ensemble = cal_proba

					# classes assumed [hold,buy,sell]
					buy_conf = float(ensemble[1]) if len(ensemble) > 1 else 0.0
					sell_conf = float(ensemble[2]) if len(ensemble) > 2 else 0.0

					# If both buy and sell confidences are extremely low (models uncertain),
					# fall back to deterministic indicator rule with a stronger confidence
					if max(buy_conf, sell_conf) < 0.1:
						logging.debug('Model confidences low (%.4f); falling back to indicator rule', max(buy_conf, sell_conf))
						last = df.iloc[-1]
						sma_fast = float(last.get('sma_fast', 0.0))
						sma_slow = float(last.get('sma_slow', 0.0))
						macd = float(last.get('macd', 0.0))
						macd_signal = float(last.get('macd_signal', 0.0))
						if sma_fast > sma_slow:
							conf = 0.6 if macd > macd_signal else 0.45
							return 'BUY', float(conf)
						else:
							conf = 0.6 if macd < macd_signal else 0.45
							return 'SELL', float(conf)

					# Determine model-best side and confidence
					if buy_conf >= sell_conf:
						model_side = 'BUY'
						model_conf = buy_conf
					else:
						model_side = 'SELL'
						model_conf = sell_conf

					# Compute a simple indicator agreement score (0..1) from available indicator columns
					last = df.iloc[-1]
					ind_checks = []
					try:
						sma_fast = float(last.get('sma_fast', 0.0))
						sma_slow = float(last.get('sma_slow', 0.0))
						ind_checks.append(1.0 if (sma_fast > sma_slow and model_side == 'BUY') or (sma_fast < sma_slow and model_side == 'SELL') else 0.0)
					except Exception:
						ind_checks.append(0.0)
					try:
						macd = float(last.get('macd', 0.0))
						macd_signal = float(last.get('macd_signal', 0.0))
						ind_checks.append(1.0 if (macd > macd_signal and model_side == 'BUY') or (macd < macd_signal and model_side == 'SELL') else 0.0)
					except Exception:
						ind_checks.append(0.0)
					try:
						rsi = float(last.get('rsi', 0.0))
						ind_checks.append(1.0 if (rsi > 50 and model_side == 'BUY') or (rsi < 50 and model_side == 'SELL') else 0.0)
					except Exception:
						ind_checks.append(0.0)

					indicator_score = sum(ind_checks) / max(len(ind_checks), 1)

					# Blend model confidence with indicator agreement to produce final win_prob
					win_prob = float(min(0.99, MODEL_WEIGHT * float(model_conf) + INDICATOR_WEIGHT * float(indicator_score)))

					return model_side, win_prob

			# If calibrated not available, but a base model exists, try base directly
			if base is not None and hasattr(base, 'predict_proba'):
				proba = base.predict_proba(Xs)[0]
				buy_conf = float(proba[1]) if len(proba) > 1 else 0.0
				sell_conf = float(proba[2]) if len(proba) > 2 else 0.0
				if buy_conf >= sell_conf:
					return 'BUY', buy_conf
				else:
					return 'SELL', sell_conf

		except Exception:
			logging.exception('Model predict failed; falling back to indicators')

	# Default deterministic indicator fallback (should not be random)
	last = df.iloc[-1]
	sma_fast = float(last.get('sma_fast', 0.0))
	sma_slow = float(last.get('sma_slow', 0.0))
	macd = float(last.get('macd', 0.0))
	macd_signal = float(last.get('macd_signal', 0.0))
	if sma_fast > sma_slow:
		conf = 0.55 if macd > macd_signal else 0.45
		return 'BUY', float(conf)
	else:
		conf = 0.55 if macd < macd_signal else 0.45
		return 'SELL', float(conf)
