#!/usr/bin/env python3
"""Check backend persisted history and pending files and print summary counts.

Run with the project's Python, e.g.:
  & .venv/.venv/Scripts/Activate.ps1
  python tools/check_backend_history.py
"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FILES = [
    'trade_history.json',
    'trade_history_testnet.json',
    'trade_history_mainnet.json',
]
PENDING_FILES = [
    'pending_trades.json',
    'pending_trades_testnet.json',
    'pending_trades_mainnet.json',
]

def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print('  Failed to load', path, '=>', e)
        return None

def summarize_trades(trades):
    if not isinstance(trades, list):
        # try common wrappers
        if isinstance(trades, dict):
            for k in ('trades','history','data'):
                if k in trades and isinstance(trades[k], list):
                    trades = trades[k]
                    break
        if not isinstance(trades, list):
            return (0,0,0.0,0)
    wins = sum(1 for t in trades if str(t.get('status','')).lower()=='win')
    losses = sum(1 for t in trades if str(t.get('status','')).lower()=='loss')
    realized = 0.0
    for t in trades:
        try:
            realized += float(t.get('realized_pnl_usdt') or 0)
        except Exception:
            pass
    return (len(trades), wins, losses, realized)

def main():
    print('Repository root:', ROOT)
    print('\nHistory files:')
    for fn in FILES:
        path = os.path.join(ROOT, fn)
        exists = os.path.exists(path)
        print(f'- {fn}: exists={exists}')
        if exists:
            data = load_json(path)
            if data is None:
                continue
            total, wins, losses, realized = summarize_trades(data)
            print(f'  -> total={total} wins={wins} losses={losses} realized_pnl={realized}')

    print('\nPending files:')
    for fn in PENDING_FILES:
        path = os.path.join(ROOT, fn)
        exists = os.path.exists(path)
        print(f'- {fn}: exists={exists}')
        if exists:
            data = load_json(path)
            if data is None:
                continue
            # pending may be list or dict
            if isinstance(data, dict) and 'open_trades' in data and isinstance(data['open_trades'], list):
                pend = data['open_trades']
            elif isinstance(data, list):
                pend = data
            else:
                pend = []
            pending_open = len([t for t in pend if t.get('status')=='open'])
            pending_amount = sum(float(t.get('amount_usdt') or 0) for t in pend if t.get('status')=='open')
            print(f'  -> open={pending_open} pending_amount={pending_amount}')

    # Also try the backend helper path used in webui (two levels down)
    repo_level = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    print('\nOther checks:')
    # print a short sample of last 3 trades from the testnet history if present
    testnet_path = os.path.join(ROOT, 'trade_history_testnet.json')
    if os.path.exists(testnet_path):
        d = load_json(testnet_path) or []
        if isinstance(d, list) and d:
            print('Last 3 testnet trades timestamps:', [t.get('timestamp') for t in d[-3:]])

if __name__ == '__main__':
    main()
from src.utils import get_history_file
import json, os
hf = get_history_file()
print('get_history_file ->', hf)
if os.path.exists(hf):
    with open(hf,'r',encoding='utf-8') as f:
        data=json.load(f)
    if isinstance(data,list): trades=data
    elif isinstance(data,dict): trades=data.get('trades') or data.get('history') or data.get('trade_history') or []
    else: trades=[]
    wins=sum(1 for t in trades if str(t.get('status')).lower()=='win')
    losses=sum(1 for t in trades if str(t.get('status')).lower()=='loss')
    print('len=',len(trades),'wins=',wins,'losses=',losses,'realized=',sum(float(t.get('realized_pnl_usdt') or 0) for t in trades))
else:
    print('history file missing')
