#!/usr/bin/env python3
"""Merge trade_history_testnet.json files from data dirs and backups into a single
repo-level trade_history_testnet.json file. Deduplicates by orderId and timestamp.

Run from project root. After merging you can POST /api/notify_update to refresh the UI.
"""
import json
import os
import glob
from collections import OrderedDict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def find_candidate_files():
    # repo-level candidates
    candidates = []
    repo_candidate = os.path.join(ROOT, 'trade_history_testnet.json')
    if os.path.exists(repo_candidate):
        candidates.append(repo_candidate)
    # any backups in repo root
    for p in glob.glob(os.path.join(ROOT, 'trade_history_testnet.json.bak*')):
        candidates.append(p)
    # data/*/trade_history_testnet.json
    for p in glob.glob(os.path.join(ROOT, 'data', '*', 'trade_history_testnet.json')):
        candidates.append(p)
    return list(OrderedDict.fromkeys(candidates))

def load_trades(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict):
                # some files may wrap trades under a key
                trades = data.get('trades') or data.get('history') or data.get('trade_history') or []
            elif isinstance(data, list):
                trades = data
            else:
                trades = []
            return trades
    except Exception:
        return []

def merge_trades(files):
    seen = {}
    merged = []
    for f in files:
        trades = load_trades(f)
        for t in trades:
            # prefer orderId if present, else timestamp+side+entry as key
            key = None
            try:
                key = str(t.get('orderId'))
            except Exception:
                key = None
            if not key or key in ('None',''):
                key = f"{t.get('timestamp')}-{t.get('side')}-{t.get('entry')}"
            if key in seen:
                # attempt to merge missing fields (prefer existing non-empty values)
                prev = seen[key]
                for k,v in (t or {}).items():
                    if (prev.get(k) is None or prev.get(k) == '') and v not in (None, ''):
                        prev[k] = v
                continue
            seen[key] = dict(t)
            merged.append(seen[key])
    # sort by timestamp if available
    try:
        merged.sort(key=lambda x: x.get('timestamp') or x.get('closed_at') or '')
    except Exception:
        pass
    return merged

def write_repo_history(trades):
    out = os.path.join(ROOT, 'trade_history_testnet.json')
    try:
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(trades, f, indent=2, ensure_ascii=False)
        print('Wrote', out, 'with', len(trades), 'trades')
    except Exception as e:
        print('Failed to write', out, e)

def main():
    files = find_candidate_files()
    if not files:
        print('No candidate trade_history_testnet.json files found')
        return
    print('Found candidate files:')
    for f in files:
        print(' -', f)
    merged = merge_trades(files)
    write_repo_history(merged)

if __name__ == '__main__':
    main()
