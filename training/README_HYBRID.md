Hybrid Ensemble Model (LightGBM + XGBoost + LSTM)
===============================================

Quick notes:

- Training script: `python training/train_hybrid.py --input data/train_ready.csv --out models/hybrid --epochs 10`
- Prediction module: `python -m src.predict_hybrid --model-dir models/hybrid --csv sample.csv`

Files produced in `models/hybrid/`:

- `scaler.pkl` - StandardScaler
- `lgb_model.pkl` - LightGBM model (joblib)
- `xgb_model.json` - XGBoost model (json)
- `lstm_model.h5` - Keras LSTM model
- `manifest.json` - small manifest listing files and weights

Notes and recommendations:

- Train on a machine with TensorFlow available (GPU optional). LSTM training is lightweight by default in the script but you can increase epochs.
- Ensure the training CSV has a `target` column (0/1) and that numeric features are present consistently between training and prediction.
- The ensemble weights default to LSTM 0.4, LightGBM 0.3, XGBoost 0.3 and are stored in `manifest.json` if you want to adjust them later.
