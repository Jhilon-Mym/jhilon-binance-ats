#!/usr/bin/env python3
"""Fast backtest runner that avoids ML model by stubbing predict_signal.

Useful when the full XGBoost model is slow or unavailable.
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.indicators import add_indicators
from src import strategy as strat_mod
from src.config import Config

ROOT = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(ROOT, 'klines_BTCUSDT_5m.csv')


def load_klines():
    df = pd.read_csv(CSV)
    df = df[['timestamp','open','high','low','close','volume']].copy()
    for c in ['open','high','low','close','volume']:
        df[c] = df[c].astype(float)
    return df


def stub_predict_signal(df):
    """Deterministic lightweight predictor based on indicators.

    Returns ('BUY'|'SELL'|None, prob)
    """
    last = df.iloc[-1]
    # Simple strict rules to favor high-precision signals
    if last['sma_fast'] > last['sma_slow'] and last['rsi'] > 55 and last['macd'] > last['macd_signal']:
        return 'BUY', 0.9
    if last['sma_fast'] < last['sma_slow'] and last['rsi'] < 45 and last['macd'] < last['macd_signal']:
        return 'SELL', 0.9
    return None, 0.0


def run_backtest(cfg_overrides=None):
    # temporarily apply overrides
    cfg_backup = {}
    if cfg_overrides:
        for k,v in cfg_overrides.items():
            cfg_backup[k] = getattr(Config, k)
            setattr(Config, k, v)

    df = load_klines()
    df = add_indicators(df).dropna().reset_index(drop=True)

    # monkeypatch strategy.predict_signal to our stub
    original_predict = getattr(strat_mod, 'predict_signal', None)
    strat_mod.predict_signal = stub_predict_signal

    trades = []
    for i in range(len(df)):
        sub = df.iloc[:i+1].copy().reset_index(drop=True)
        sig = strat_mod.apply_strategy(sub)
        if sig:
            entry = sub.iloc[-1]['close']
            sl = sig['sl']
            tp = sig['tp']
            side = sig['side']
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

    # restore
    if original_predict is not None:
        strat_mod.predict_signal = original_predict
    if cfg_overrides:
        for k,v in cfg_backup.items():
            setattr(Config, k, v)

    wins = sum(1 for t in trades if t['result']=='win')
    losses = sum(1 for t in trades if t['result']=='loss')
    total = len(trades)
    win_pct = (wins/total*100) if total>0 else 0
    return {'wins':wins,'losses':losses,'total':total,'win_pct':win_pct}


if __name__ == '__main__':
    print('Running fast backtest baseline...')
    base = run_backtest()
    print(f"Baseline: total={base['total']} wins={base['wins']} losses={base['losses']} win%={base['win_pct']:.2f}")

    print('\nTesting conservative filters...')
    scenarios = [
        ('Raise ATR_SL_MULT to 2.5', {'ATR_SL_MULT':2.5}),
        ('Require 3 indicators (stricter)', None),
        ('Increase AI confidence to 0.8', {'AI_MIN_CONFIDENCE_OVERRIDE':0.8}),
        ('Combine strict filters', {'ATR_SL_MULT':3.0,'AI_MIN_CONFIDENCE_OVERRIDE':0.85})
    ]

    # Helper to monkeypatch indicator_confirms to require 3
    original_indicator_confirms = getattr(strat_mod, 'indicator_confirms', None)

    for name, cfg in scenarios:
        print('\n--', name)
        if name.startswith('Require 3 indicators'):
            def require_three(row, side):
                checks = [
                    (row['sma_fast']>row['sma_slow']) if side=='BUY' else (row['sma_fast']<row['sma_slow']),
                    (row['rsi']>50) if side=='BUY' else (row['rsi']<50),
                    (row['macd']>row['macd_signal']) if side=='BUY' else (row['macd']<row['macd_signal'])
                ]
                return sum(1 for c in checks if c) >= 3
            setattr(strat_mod, 'indicator_confirms', require_three)
            res = run_backtest()
            # restore
            if original_indicator_confirms is not None:
                setattr(strat_mod, 'indicator_confirms', original_indicator_confirms)
        else:
            res = run_backtest(cfg)
        print(f"total={res['total']} wins={res['wins']} losses={res['losses']} win%={res['win_pct']:.2f}")

    print('\nDone')
