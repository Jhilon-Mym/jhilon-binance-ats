#!/usr/bin/env python3
"""Small helper: print repo-level trade history stats.

Run with the project's Python interpreter so it reads the same .env/paths.
"""
import os, json

def find_repo_history():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # prefer testnet file then mainnet
    t = os.path.join(root, 'trade_history_testnet.json')
    m = os.path.join(root, 'trade_history_mainnet.json')
    if os.path.exists(t):
        return t
    if os.path.exists(m):
        return m
    return None


def summarize(path):
    if not path or not os.path.exists(path):
        print('No repo-level trade history found at expected locations.')
        return 1
    try:
        with open(path, 'r', encoding='utf-8') as f:
            trades = json.load(f)
    except Exception as e:
        print('Failed to read history file:', e)
        return 2
    if not isinstance(trades, list):
        print('Unexpected format: history file is not a list')
        return 3
    total = len(trades)
    wins = sum(1 for t in trades if str(t.get('status')).lower() == 'win')
    losses = sum(1 for t in trades if str(t.get('status')).lower() == 'loss')
    realized = 0.0
    for t in trades:
        try:
            realized += float(t.get('realized_pnl_usdt') or 0)
        except Exception:
            pass
    print('repo_history file:', path)
    print('total=', total, 'wins=', wins, 'losses=', losses, 'realized_pnl=', realized)
    print('\nLast 5 trades:')
    for t in trades[-5:]:
        print(' -', t.get('timestamp') or t.get('closed_at') or '-', t.get('status'), t.get('realized_pnl_usdt'))
    return 0


if __name__ == '__main__':
    p = find_repo_history()
    exit(summarize(p))
