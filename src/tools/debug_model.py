import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Paths
MODEL_PATH = 'model_xgb.pkl'
SCALER_PATH = 'scaler.pkl'
DATA_PATH = 'klines_BTCUSDT_5m.csv'

# Features (must match hybrid_ai.py)
features = [
    "sma_fast", "sma_slow", "ema_200", "rsi", "macd", "macd_signal", "atr",
    "volatility", "momentum", "returns", "volume_sma", "high_low_spread"
]

def add_indicators(df):
    from src.indicators import add_indicators
    return add_indicators(df)

def main():
    # Load data
    df = pd.read_csv(DATA_PATH)
    df = add_indicators(df)
    df = df.dropna().reset_index(drop=True)
    # Target: 1=BUY win, 2=SELL win, 0=HOLD/LOSS (simple logic)
    future_close = df['close'].shift(-1)
    df['target'] = 0
    df.loc[df['close'] < future_close, 'target'] = 1
    df.loc[df['close'] > future_close, 'target'] = 2
    X = df[features]
    y = df['target']
    # Load model & scaler
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    X_scaled = scaler.transform(X)
    # Predict
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)
    # Confusion matrix
    print('Confusion Matrix:')
    print(confusion_matrix(y, y_pred))
    print('\nClassification Report:')
    print(classification_report(y, y_pred, digits=3))
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        print('\nFeature Importances:')
        for f, imp in zip(features, model.feature_importances_):
            print(f'{f:16s}: {imp:.4f}')
    elif hasattr(model, 'get_booster'):
        print('\nFeature Importances (XGBoost):')
        booster = model.get_booster()
        scores = booster.get_score(importance_type='weight')
        for f in features:
            print(f'{f:16s}: {scores.get(f, 0):.4f}')
    # Prediction distribution
    buy_probs = y_proba[:,1]
    sell_probs = y_proba[:,2]
    print(f'\nMean BUY win_prob: {np.mean(buy_probs):.3f}, Mean SELL win_prob: {np.mean(sell_probs):.3f}')
    print(f'BUY win_prob > 0.7: {(buy_probs>0.7).sum()} / {len(buy_probs)}')
    print(f'SELL win_prob > 0.7: {(sell_probs>0.7).sum()} / {len(sell_probs)}')
    # Optional: plot histogram
    try:
        plt.hist(buy_probs, bins=30, alpha=0.5, label='BUY win_prob')
        plt.hist(sell_probs, bins=30, alpha=0.5, label='SELL win_prob')
        plt.legend()
        plt.title('Prediction Probability Distribution')
        plt.show()
    except Exception:
        pass

if __name__ == '__main__':
    main()
