#!/usr/bin/env python3
import glob, json, os

# Look for trade_history files in project root
files = glob.glob(os.path.join(os.path.dirname(__file__), '..', 'trade_history*.json'))
# fallback to current dir pattern if none found
if not files:
    files = glob.glob('trade_history*.json')

if not files:
    print('No trade_history*.json files found in project root')
    raise SystemExit(0)

for f in files:
    try:
        with open(f, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
    except Exception as e:
        print('Failed to load', f, e)
        continue
    # Try common shapes
    if isinstance(data, list):
        trades = data
    elif isinstance(data, dict):
        trades = data.get('trades') or data.get('history') or data.get('history', []) or []
        if trades is None:
            trades = []
    else:
        trades = []
    wins = sum(1 for t in trades if str(t.get('status')).lower() == 'win')
    losses = sum(1 for t in trades if str(t.get('status')).lower() == 'loss')
    realized = sum(float(t.get('realized_pnl_usdt') or 0) for t in trades)
    print(f"{os.path.basename(f)}: len={len(trades)} wins={wins} losses={losses} realized={realized}")
