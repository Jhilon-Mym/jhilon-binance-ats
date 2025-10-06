import os, pickle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
model_p = os.path.join(ROOT, 'models', 'model.pkl')
scaler_p = os.path.join(ROOT, 'models', 'scaler.pkl')

if not os.path.exists(model_p):
    print('model.pkl not found')
    raise SystemExit(1)

with open(model_p, 'rb') as f:
    primary = pickle.load(f)

if not isinstance(primary, dict) or 'scaler' not in primary:
    print('model.pkl does not contain scaler')
    raise SystemExit(1)

scaler = primary['scaler']
with open(scaler_p, 'wb') as f:
    pickle.dump(scaler, f)

print('synced scaler to', scaler_p)
