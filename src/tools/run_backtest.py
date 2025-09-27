#!/usr/bin/env python3
"""Simple backtest runner for strategy performance using klines_BTCUSDT_5m.csv

Produces win%, total trades, and simple P/L assuming equal notional per trade.
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json

import pandas as pd
from copy import deepcopy

from src.strategy import apply_strategy
from src.indicators import add_indicators
from src.config import Config

ROOT = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(ROOT, 'klines_BTCUSDT_5m.csv')

def load_klines():
    df = pd.read_csv(CSV)
    # expect columns: timestamp, open, high, low, close, volume
    df = df[['timestamp','open','high','low','close','volume']].copy()
    for c in ['open','high','low','close','volume']:
        df[c] = df[c].astype(float)
    return df


def run_backtest(cfg_overrides=None):
    # apply config overrides dict to Config class temporarily
    cfg_backup = {}
    if cfg_overrides:
        for k,v in cfg_overrides.items():
            cfg_backup[k] = getattr(Config, k)
            setattr(Config, k, v)

    df = load_klines()
    trades = []
    df = add_indicators(df).dropna().reset_index(drop=True)
    for i in range(len(df)):
        sub = df.iloc[:i+1].copy().reset_index(drop=True)
        sig = apply_strategy(sub)
        if sig:
            # assume trade executed at close price of that candle
            entry = sub.iloc[-1]['close']
            sl = sig['sl']
            tp = sig['tp']
            side = sig['side']
            # find next candle where sl or tp hit (use following rows)
            result = 'open'
            for j in range(i+1, len(df)):
                low = df.iloc[j]['low']
                high = df.iloc[j]['high']
                if side == 'BUY':
                    if low <= sl:
                        result = 'loss'
                        break
                    if high >= tp:
                        result = 'win'
                        break
                else:
                    if high >= sl:
                        result = 'loss'
                        break
                    if low <= tp:
                        result = 'win'
                        break
            trades.append({'side':side,'entry':entry,'sl':sl,'tp':tp,'result':result})
    # restore config
    if cfg_overrides:
        for k,v in cfg_backup.items():
            setattr(Config, k, v)

    wins = sum(1 for t in trades if t['result']=='win')
    losses = sum(1 for t in trades if t['result']=='loss')
    total = len(trades)
    win_pct = (wins/total*100) if total>0 else 0
    return {'wins':wins,'losses':losses,'total':total,'win_pct':win_pct,'trades':trades}


if __name__ == '__main__':
    print('Running baseline backtest...')
    base = run_backtest()
    print(f"Baseline: total={base['total']} wins={base['wins']} losses={base['losses']} win%={base['win_pct']:.2f}")

    # Conservative filters to try to raise win% (safety first)
    print('\nTesting conservative filters...')
    scenarios = [
        {'name':'Raise ATR_SL_MULT to 2.5','cfg':{'ATR_SL_MULT':2.5}},
        {'name':'Require 3 indicators','cfg':{}},
        {'name':'Increase AI confidence to 0.8','cfg':{'AI_MIN_CONFIDENCE_OVERRIDE':0.8}},
        {'name':'Combine strict filters','cfg':{'ATR_SL_MULT':3.0,'AI_MIN_CONFIDENCE_OVERRIDE':0.85}}
    ]
    # For 'Require 3 indicators' tweak strategy function by monkeypatching
    import types
    from src import strategy as strat_mod
    original_indicator_confirms = strat_mod.indicator_confirms if hasattr(strat_mod,'indicator_confirms') else None

    for s in scenarios:
        print('\n--', s['name'])
        if s['name']=='Require 3 indicators':
            # monkeypatch to require 3
            def require_three(row, side):
                checks = [
                    (row['sma_fast']>row['sma_slow']) if side=='BUY' else (row['sma_fast']<row['sma_slow']),
                    (row['rsi']>50) if side=='BUY' else (row['rsi']<50),
                    (row['macd']>row['macd_signal']) if side=='BUY' else (row['macd']<row['macd_signal'])
                ]
                return sum(1 for c in checks if c) >= 3
            # monkeypatch into module
            strat_mod.indicator_confirms = require_three
            res = run_backtest(s.get('cfg'))
            # restore
            if original_indicator_confirms is not None:
                strat_mod.indicator_confirms = original_indicator_confirms
        else:
            res = run_backtest(s.get('cfg'))
        print(f"total={res['total']} wins={res['wins']} losses={res['losses']} win%={res['win_pct']:.2f}")

    print('\nDone')
