"""
Hybrid training pipeline (LightGBM + XGBoost + LSTM)

Saves models to models/hybrid/:
 - lgb_model.pkl (joblib)
 - xgb_model.json (XGBoost native
 - lstm_model.h5 (Keras)
 - scaler.pkl (joblib)

Usage:
 python training/train_hybrid.py --input data/train_ready.csv --out models/hybrid --epochs 5

This script is intentionally conservative with defaults so it can be run on small machines.
"""
import os
import argparse
import joblib
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score


from tensorflow import keras

# Heavy ML libraries are imported lazily inside main() to keep module import cheap


def default_feature_engineering(df):
    # Minimal, conservative feature engineering. Expand in your pipeline as needed.
    df = df.copy()
    # Fill missing
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)
    # Example engineered features (if columns exist)
    if 'close' in df.columns and 'volume' in df.columns:
        df['close_x_vol'] = df['close'] * (df['volume'] + 1e-9)
    return df


def build_lstm(input_shape):
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(64, input_shape=input_shape))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='CSV with features and target column named "target"')
    parser.add_argument('--out', default='models/hybrid', help='Output directory for models')
    parser.add_argument('--epochs', type=int, default=5, help='Epochs for LSTM (kept small by default)')
    parser.add_argument('--test-size', type=float, default=0.2)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(os.path.join(r"D:\binance_ats_clone\obaidur-binance-ats-main", os.path.basename(args.input)))
    # Support existing labeling conventions: 'target', 'label', or 'y'
    label_col = None
    for c in ('target', 'label', 'y'):
        if c in df.columns:
            label_col = c
            break
    if label_col is None:
        raise SystemExit('Input CSV must contain a label column named one of: target, label, y')

    df = default_feature_engineering(df)

    # Map multi-class label conventions (e.g., -1/0/1) to binary 0/1: treat 1 as positive, others as negative
    y = (df[label_col] == 1).astype(int)

    # Use all numeric columns except the label column as features
    X = df.select_dtypes(include=[np.number]).drop(columns=[label_col])

    # Train/test split (no shuffle, time-series friendly)
    split_idx = int(len(X) * (1 - args.test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Import heavy ML libs here (lazy import)
    try:
        import lightgbm as lgb
        import xgboost as xgb
        from tensorflow import keras
    except Exception as e:
        raise RuntimeError('Missing ML dependencies. Please install lightgbm, xgboost and tensorflow.') from e

    # LightGBM
    lgb_model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=6)
    lgb_model.fit(X_train_s, y_train)
    lgb_pred = lgb_model.predict_proba(X_test_s)[:, 1]

    # XGBoost
    xgb_model = xgboost = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train_s, y_train)
    xgb_pred = xgb_model.predict_proba(X_test_s)[:, 1]

    # LSTM expects 3D: (samples, timesteps, features)
    X_train_lstm = X_train_s.reshape((X_train_s.shape[0], 1, X_train_s.shape[1]))
    X_test_lstm = X_test_s.reshape((X_test_s.shape[0], 1, X_test_s.shape[1]))

    # Build a small LSTM using the helper
    lstm_model = keras.models.Sequential()
    lstm_model.add(keras.layers.LSTM(64, input_shape=(1, X_train_s.shape[1])))
    lstm_model.add(keras.layers.Dense(1, activation='sigmoid'))
    lstm_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    lstm_model.fit(X_train_lstm, y_train.values, epochs=args.epochs, batch_size=32, verbose=1)
    lstm_pred = lstm_model.predict(X_test_lstm, verbose=0).reshape(-1)

    # Ensemble weighted average
    final_prob = 0.4 * lstm_pred + 0.3 * lgb_pred + 0.3 * xgb_pred
    final_label = (final_prob >= 0.5).astype(int)

    # Metrics
    acc = accuracy_score(y_test, final_label)
    try:
        auc = roc_auc_score(y_test, final_prob)
    except Exception:
        auc = None

    print(f'Hybrid ensemble accuracy on test: {acc:.4f}, AUC: {auc}')

    # Save models and scaler
    joblib.dump(scaler, os.path.join(args.out, 'scaler.pkl'))
    joblib.dump(lgb_model, os.path.join(args.out, 'lgb_model.pkl'))
    # XGBoost save to json
    xgb_model.get_booster().save_model(os.path.join(args.out, 'xgb_model.json'))
    lstm_model.save(os.path.join(args.out, 'lstm_model.h5'))

    # Save a tiny manifest
    manifest = {
        'models': {
            'lgb': 'lgb_model.pkl',
            'xgb': 'xgb_model.json',
            'lstm': 'lstm_model.h5',
            'scaler': 'scaler.pkl'
        },
        'weights': {'lstm': 0.4, 'lgb': 0.3, 'xgb': 0.3}
    }
    with open(os.path.join(args.out, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    print('Models saved to', args.out)


if __name__ == '__main__':
    main()
