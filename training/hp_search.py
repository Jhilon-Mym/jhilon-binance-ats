"""
hp_search.py: Hyperparameter search for Hybrid Model (LightGBM, XGBoost, LSTM)

- Loads CSV (auto-detect label column)
- Runs randomized grid search for each model
- Logs best params and scores
- Saves results to training/hp_search_results.json
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
import lightgbm as lgb
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras

DATA_PATH = "klines_BTCUSDT_5m_balanced.csv"
RESULT_PATH = "training/hp_search_results.json"

# Utility: Auto-detect label column
def detect_label_col(df):
    for col in ["label", "target", "y"]:
        if col in df.columns:
            return col
    raise ValueError("No label column found!")

def load_data():
    df = pd.read_csv(DATA_PATH)
    label_col = detect_label_col(df)
    X = df.drop(columns=[label_col])
    y = df[label_col]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# LightGBM search
def search_lgb(X_train, y_train):
    param_dist = {
        'num_leaves': [15, 31, 63],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [50, 100, 200],
    }
    model = lgb.LGBMClassifier()
    search = RandomizedSearchCV(model, param_dist, n_iter=10, scoring='roc_auc', cv=3, random_state=42)
    search.fit(X_train, y_train)
    return search.best_params_, search.best_score_

# XGBoost search
def search_xgb(X_train, y_train):
    param_dist = {
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [50, 100, 200],
        'subsample': [0.7, 0.8, 1.0],
    }
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    search = RandomizedSearchCV(model, param_dist, n_iter=10, scoring='roc_auc', cv=3, random_state=42)
    search.fit(X_train, y_train)
    return search.best_params_, search.best_score_

# LSTM search (simple grid)
def search_lstm(X_train, y_train):
    # For LSTM, use only a few param combos due to speed
    input_dim = X_train.shape[1]
    param_grid = [
        {'units': 32, 'epochs': 5, 'lr': 0.001},
        {'units': 64, 'epochs': 5, 'lr': 0.001},
        {'units': 32, 'epochs': 10, 'lr': 0.0005},
    ]
    best_score = 0
    best_params = None
    for params in param_grid:
        model = keras.Sequential([
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(params['units'], activation='relu'),
            keras.layers.Dense(1, activation='sigmoid'),
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=params['lr']), loss='binary_crossentropy', metrics=['AUC'])
        history = model.fit(X_train, y_train, epochs=params['epochs'], batch_size=32, verbose=0)
        preds = model.predict(X_train).flatten()
        auc = roc_auc_score(y_train, preds)
        if auc > best_score:
            best_score = auc
            best_params = params
    return best_params, best_score

def main():
    X_train, X_test, y_train, y_test = load_data()
    results = {}
    print("Searching LightGBM...")
    lgb_params, lgb_score = search_lgb(X_train, y_train)
    results['lightgbm'] = {'params': lgb_params, 'score': lgb_score}
    print("Searching XGBoost...")
    xgb_params, xgb_score = search_xgb(X_train, y_train)
    results['xgboost'] = {'params': xgb_params, 'score': xgb_score}
    print("Searching LSTM...")
    lstm_params, lstm_score = search_lstm(X_train, y_train)
    results['lstm'] = {'params': lstm_params, 'score': lstm_score}
    with open(RESULT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print("Done. Results saved to", RESULT_PATH)

if __name__ == "__main__":
    main()
