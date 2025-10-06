"""
Hybrid prediction module.

Provides `predict_hybrid(features)` which returns a single probability (float 0..1)
combining LSTM (0.4), LightGBM (0.3), and XGBoost (0.3).

Usage (CLI):
 python -m src.predict_hybrid --model-dir models/hybrid --csv sample.csv

Or import and call:
 from src.predict_hybrid import HybridPredictor
 pred = HybridPredictor('models/hybrid')
 prob = pred.predict_single(sample_features_array)
"""
import os
import numpy as np
import joblib
import json
import argparse
import pandas as pd
import time
import logging

# Heavy ML imports are performed lazily inside the class to allow simple imports
_tf = None
_xgb = None


class HybridPredictor:
    def __init__(self, model_dir='models/hybrid'):
        self.model_dir = model_dir
        manifest_path = os.path.join(model_dir, 'manifest.json')
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f'Manifest not found in {model_dir}. Have you trained the hybrid models?')
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)

        # Load scaler
        self.scaler = joblib.load(os.path.join(model_dir, self.manifest['models']['scaler']))
        # Lazy-load heavy models when instantiating predictor so module import is cheap
        try:
            # LightGBM model (joblib)
            self.lgb = joblib.load(os.path.join(model_dir, self.manifest['models']['lgb']))
        except Exception as e:
            raise RuntimeError('Failed to load LightGBM model: ' + str(e))

        try:
            global _xgb
            import xgboost as xgb
            _xgb = xgb
            xgb_path = os.path.join(model_dir, self.manifest['models']['xgb'])
            self.xgb = xgb.Booster()
            self.xgb.load_model(xgb_path)
        except Exception as e:
            raise RuntimeError('Failed to load XGBoost model: ' + str(e))

        try:
            global _tf
            from tensorflow import keras
            _tf = True
            self.lstm = keras.models.load_model(os.path.join(model_dir, self.manifest['models']['lstm']))
        except Exception as e:
            raise RuntimeError('Failed to load LSTM model (TensorFlow required): ' + str(e))

        # Weights
        self.weights = self.manifest.get('weights', {'lstm': 0.4, 'lgb': 0.3, 'xgb': 0.3})

    def _prep(self, x):
        # x: 1D array-like of numeric features (same order as training numeric columns)
        arr = np.asarray(x, dtype=float).reshape(1, -1)
        scaled = self.scaler.transform(arr)
        return arr, scaled

    def predict_single(self, x):
        """Return final probability for a single sample feature vector."""
        arr, scaled = self._prep(x)

        # Measure per-model inference time for diagnosis
        logger = logging.getLogger(__name__)
        t0 = time.time()
        prob_lgb = 0.0
        prob_xgb = 0.0
        prob_lstm = 0.0
        try:
            lgb_start = time.time()
            prob_lgb = self.lgb.predict_proba(scaled)[0][1]
            lgb_dt = time.time() - lgb_start
        except Exception as e:
            lgb_dt = None
            logger.debug('lgb predict failed: %s', e)
        try:
            xgb_start = time.time()
            dmat = _xgb.DMatrix(scaled)
            prob_xgb = float(self.xgb.predict(dmat).ravel()[0])
            xgb_dt = time.time() - xgb_start
        except Exception as e:
            xgb_dt = None
            logger.debug('xgb predict failed: %s', e)
        try:
            lstm_start = time.time()
            lstm_in = scaled.reshape((scaled.shape[0], 1, scaled.shape[1]))
            prob_lstm = float(self.lstm.predict(lstm_in, verbose=0).ravel()[0])
            lstm_dt = time.time() - lstm_start
        except Exception as e:
            lstm_dt = None
            logger.debug('lstm predict failed: %s', e)

        final = self.weights['lstm'] * prob_lstm + self.weights['lgb'] * prob_lgb + self.weights['xgb'] * prob_xgb
        # Clamp to [0,1]
        final = max(0.0, min(1.0, float(final)))
        total_dt = time.time() - t0
        try:
            logger.info('[TIMING][HybridPredictor] total=%.4fs lgb=%s xgb=%s lstm=%s', total_dt, ('%.4fs' % lgb_dt) if lgb_dt is not None else 'err', ('%.4fs' % xgb_dt) if xgb_dt is not None else 'err', ('%.4fs' % lstm_dt) if lstm_dt is not None else 'err')
        except Exception:
            pass
        return final

    def predict_csv(self, csv_path, output_col='win_prob'):
        df = pd.read_csv(csv_path)
        # Use numeric columns only (same as training used)
        X = df.select_dtypes(include=[np.number])
        probs = []
        for i in range(len(X)):
            p = self.predict_single(X.iloc[i].values)
            probs.append(p)
        df[output_col] = probs
        return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', default='models/hybrid')
    parser.add_argument('--csv', help='CSV file to predict (will print first 5 rows with probabilities)')
    parser.add_argument('--row', help='Comma-separated numeric features for a single prediction')
    args = parser.parse_args()

    pred = HybridPredictor(args.model_dir)
    if args.csv:
        out = pred.predict_csv(args.csv)
        print(out.head())
    elif args.row:
        parts = [float(x) for x in args.row.split(',')]
        p = pred.predict_single(parts)
        print('probability:', p)
    else:
        print('No input provided. Use --csv or --row')


if __name__ == '__main__':
    main()
