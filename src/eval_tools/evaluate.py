"""
evaluate.py: Hybrid Model Evaluation & Threshold Tuning

- Loads trained hybrid models (LightGBM, XGBoost, LSTM, scaler)
- Loads validation CSV, runs predictions
- Computes metrics: accuracy, AUC, win_prob distribution
- Sweeps decision threshold, reports best threshold for 75%+ win_prob
- Saves evaluation report to src/eval_tools/eval_report.json
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import importlib
predict_hybrid = importlib.import_module('predict_hybrid')
HybridPredictor = predict_hybrid.HybridPredictor

DATA_PATH = "../../klines_BTCUSDT_5m_balanced.csv"
REPORT_PATH = "eval_report.json"

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
    return X, y

def evaluate(predictor, X, y):
    preds = []
    for i in range(len(X)):
        row = X.iloc[i].to_dict()
        win_prob = predictor.predict_single(row)
        preds.append(win_prob)
    auc = roc_auc_score(y, preds)
    acc = accuracy_score(y, [int(p > 0.5) for p in preds])
    # Threshold sweep
    best_acc = 0
    best_thresh = 0.5
    for thresh in np.arange(0.5, 0.95, 0.01):
        acc_t = accuracy_score(y, [int(p > thresh) for p in preds])
        if acc_t > best_acc:
            best_acc = acc_t
            best_thresh = thresh
    return {
        'auc': auc,
        'accuracy': acc,
        'best_acc': best_acc,
        'best_thresh': best_thresh,
        'win_prob_mean': float(np.mean(preds)),
        'win_prob_std': float(np.std(preds)),
        'confusion_matrix': confusion_matrix(y, [int(p > best_thresh) for p in preds]).tolist(),
    }

def main():
    X, y = load_data()
    predictor = HybridPredictor()
    print("Evaluating hybrid model...")
    report = evaluate(predictor, X, y)
    with open(REPORT_PATH, 'w') as f:
        json.dump(report, f, indent=2)
    print("Done. Report saved to", REPORT_PATH)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
