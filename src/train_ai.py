import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False

# Load historical data (update path as needed)
df = pd.read_csv('klines_BTCUSDT_5m.csv')

# Import indicators and add features
def add_indicators(df):
    from src.indicators import add_indicators
    return add_indicators(df)

df = add_indicators(df)

# Drop rows with NaN (from indicators)
df = df.dropna().reset_index(drop=True)

# Feature columns (must match hybrid_ai.py)
features = [
    "sma_fast", "sma_slow", "ema_200", "rsi", "macd", "macd_signal", "atr",
    "volatility", "momentum", "returns", "volume_sma", "high_low_spread"
]

ATR_MULT = 1.5  # ATR-based move threshold
future_close = df['close'].shift(-1)
df['target'] = 0
df.loc[future_close > df['close'] + ATR_MULT * df['atr'], 'target'] = 1  # BUY win
df.loc[future_close < df['close'] - ATR_MULT * df['atr'], 'target'] = 2  # SELL win

X = df[features]
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train ensemble model (XGBoost if available, else RandomForest)
if xgb_available:
    model = XGBClassifier(n_estimators=200, max_depth=8, use_label_encoder=False, eval_metric='mlogloss')
else:
    model = RandomForestClassifier(n_estimators=200, max_depth=8)

model.fit(X_scaled, y)

joblib.dump(model, 'model_xgb.pkl')
joblib.dump(scaler, 'scaler.pkl')

print('Model and scaler saved!')
