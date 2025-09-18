#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import json
from dotenv import load_dotenv
load_dotenv()
from src.config import Config
from pathlib import Path
from statistics import mean, median
ROOT = Path(__file__).resolve().parent.parent
BOT_LOG = ROOT / 'bot_run.log'
ORDERS_LOG = ROOT / 'orders.log'
DEBUG_JSONL = ROOT / 'model_debug.jsonl'

def tail_lines(path: Path, n=200):
    if not path.exists():
        return []
    with path.open('rb') as f:
        f.seek(0, 2)
        size = f.tell()
        block = -1
        data = b''
        while True:
            step = 1024 * 4
            if -step * block > size:
                f.seek(0)
                data = f.read(size)
                break
            f.seek(step * block, 2)
            data = f.read(-step * block) + data
            if data.count(b'\n') > n:
                break
            block -= 1
        lines = data.splitlines()[-n:]
        return [ln.decode('utf-8', errors='replace') for ln in lines]


def parse_debug_jsonl(path: Path):
    if not path.exists():
        return []
    entries = []
    with path.open('r', encoding='utf-8', errors='replace') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
                entries.append(obj)
            except Exception:
                continue
    return entries


def parse_orders(path: Path, n=100):
    if not path.exists():
        return []
    orders = []
    with path.open('r', encoding='utf-8', errors='replace') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
                # some lines have wrapper {"ts":.., "order": {...}}
                if isinstance(obj, dict) and 'order' in obj:
                    orders.append(obj['order'])
                else:
                    orders.append(obj)
            except Exception:
                continue
    return orders[-n:]


def summarize_debug(entries):
    total = len(entries)
    apply_count = 0
    accepted = 0
    rejected = 0
    combined_scores = []
    win_probs = []
    conflicts = 0
    recent_applies = []
    for e in entries:
        apply = e.get('apply_strategy') or e.get('apply') or {}
        ai = e.get('ai_signal') or {}
        if apply:
            apply_count += 1
            side = apply.get('side')
            if side:
                accepted += 1
            else:
                rejected += 1
            cs = apply.get('combined_score') or (e.get('weighted') or {}).get('combined_score')
            if cs is not None:
                try:
                    combined_scores.append(float(cs))
                except Exception:
                    pass
            wp = (apply.get('win_prob') or ai.get('win_prob'))
            if wp is not None:
                try:
                    win_probs.append(float(wp))
                except Exception:
                    pass
            if ai.get('side') and apply.get('side') and ai.get('side') != apply.get('side'):
                conflicts += 1
            if len(recent_applies) < 10 and apply:
                recent_applies.append({
                    'side': apply.get('side'),
                    'reason': apply.get('reason') or e.get('event'),
                    'win_prob': wp,
                    'combined_score': cs,
                    'ai_side': ai.get('side')
                })
    return {
        'total': total,
        'apply_count': apply_count,
        'accepted': accepted,
        'rejected': rejected,
        'combined_mean': mean(combined_scores) if combined_scores else None,
        'combined_median': median(combined_scores) if combined_scores else None,
        'combined_min': min(combined_scores) if combined_scores else None,
        'combined_max': max(combined_scores) if combined_scores else None,
        'win_prob_mean': mean(win_probs) if win_probs else None,
        'recent_applies': recent_applies,
        'conflicts': conflicts
    }


def summarize_orders(orders):
    by_side = {}
    total_qty = 0.0
    for o in orders:
        side = o.get('side') or o.get('side')
        qty = o.get('executedQty') or o.get('origQty') or o.get('origQuoteOrderQty')
        try:
            qtyf = float(qty) if qty is not None else 0.0
        except Exception:
            qtyf = 0.0
        by_side[side] = by_side.get(side, 0) + 1
        total_qty += qtyf
    return {'count': len(orders), 'by_side': by_side, 'total_exec_qty': total_qty}


def compute_pnl_from_orders(orders):
    # FIFO inventory for buys. Realized PnL in quote asset (USDT).
    lots = []  # list of (qty_base, price_quote_per_base)
    realized = 0.0
    wins = 0
    losses = 0
    last_buy_price = None
    last_sell_price = None
    for o in orders:
        side = (o.get('side') or '').upper()
        fills = o.get('fills') or []
        # derive price and qty from fills if available (take VWAP across fills)
        total_qty = 0.0
        total_quote = 0.0
        for f in fills:
            try:
                fq = float(f.get('qty', 0))
                fp = float(f.get('price', 0))
            except Exception:
                fq = 0.0
                fp = 0.0
            total_qty += fq
            total_quote += fq * fp
        # fallback to order-level fields
        if total_qty <= 0:
            try:
                total_qty = float(o.get('executedQty') or o.get('origQty') or 0)
            except Exception:
                total_qty = 0.0
        avg_price = (total_quote / total_qty) if (total_qty > 0) else None
        if side == 'BUY':
            if avg_price is None:
                # try cummulativeQuoteQty / qty
                try:
                    avg_price = float(o.get('cummulativeQuoteQty') or 0) / (total_qty or 1)
                except Exception:
                    avg_price = 0.0
            lots.append([total_qty, avg_price])
            last_buy_price = avg_price
        elif side == 'SELL':
            if avg_price is None:
                try:
                    avg_price = float(o.get('cummulativeQuoteQty') or 0) / (total_qty or 1)
                except Exception:
                    avg_price = 0.0
            last_sell_price = avg_price
            qty_to_sell = total_qty
            sell_price = avg_price or 0.0
            # consume lots FIFO
            profit_for_order = 0.0
            while qty_to_sell > 1e-12 and lots:
                lot_qty, lot_price = lots[0]
                take = min(lot_qty, qty_to_sell)
                proceeds = take * sell_price
                cost = take * lot_price
                profit = proceeds - cost
                profit_for_order += profit
                lot_qty -= take
                qty_to_sell -= take
                if lot_qty <= 1e-12:
                    lots.pop(0)
                else:
                    lots[0][0] = lot_qty
            realized += profit_for_order
            if profit_for_order > 0:
                wins += 1
            elif profit_for_order < 0:
                losses += 1
    # holdings base asset remaining
    holding_qty = sum(l[0] for l in lots)
    avg_cost = (sum(l[0] * l[1] for l in lots) / holding_qty) if holding_qty > 0 else None
    return {
        'realized_usdt': realized,
        'holding_base': holding_qty,
        'avg_cost_quote': avg_cost,
        'wins': wins,
        'losses': losses,
        'last_buy_price': last_buy_price,
        'last_sell_price': last_sell_price
    }


def summarize_botlog(lines):
    errors = [l for l in lines if 'ERROR' in l.upper()]
    warns = [l for l in lines if 'WARNING' in l.upper()]
    trades = [l for l in lines if '[TRADE]' in l or 'TRADE' in l]
    recent = lines[-20:]
    return {'errors': len(errors), 'warnings': len(warns), 'trade_lines': len(trades), 'recent': recent}


def print_summary():
    print('\n' + '='*60)
    print(time.strftime('%Y-%m-%d %H:%M:%S'), 'LIVE MONITOR SUMMARY')
    print('-'*60)
    debug_entries = parse_debug_jsonl(DEBUG_JSONL)
    dsum = summarize_debug(debug_entries)
    print(f"model_debug.jsonl: total={dsum['total']} apply={dsum['apply_count']} accepted={dsum['accepted']} rejected={dsum['rejected']} conflicts={dsum['conflicts']}")
    if dsum['combined_mean'] is not None:
        print(f"combined_score mean/median/min/max: {dsum['combined_mean']:.4f}/{dsum['combined_median']:.4f}/{dsum['combined_min']:.4f}/{dsum['combined_max']:.4f}")
        print(f"win_prob mean: {dsum['win_prob_mean']:.3f}")
    if dsum['recent_applies']:
        print('\nRecent applies:')
        for r in dsum['recent_applies']:
            print(' ', r)

    orders = parse_orders(ORDERS_LOG, n=200)
    osum = summarize_orders(orders)
    pnl = compute_pnl_from_orders(orders)
    print('\norders.log: count_last={count} total_exec_qty={total_exec_qty:.6f} by_side={by_side}'.format(**osum))
    print('Realized PnL (USDT): {0:.6f} | Holdings: {1:.8f} BTC | avg_cost: {2}'.format(
        pnl['realized_usdt'], pnl['holding_base'], (f"{pnl['avg_cost_quote']:.2f}" if pnl['avg_cost_quote'] else 'N/A')))
    print('Wins/Losses (sell orders): {wins}/{losses} | last_buy_price: {lb} last_sell_price: {ls}'.format(
        wins=pnl['wins'], losses=pnl['losses'], lb=(f"{pnl['last_buy_price']:.2f}" if pnl['last_buy_price'] else 'N/A'), ls=(f"{pnl['last_sell_price']:.2f}" if pnl['last_sell_price'] else 'N/A')))

    # try to extract latest balance line from bot_run.log
    bal_usdt = None
    bal_btc = None
    for ln in reversed(tail_lines(BOT_LOG, 800)):
        if '[BALANCE]' in ln or 'USDT=' in ln:
            # example: [BALANCE] USDT=20000.00 | BTC=2.000000
            try:
                parts = ln.split('USDT=')[-1]
                parts = parts.replace('|', ' ').replace(',', ' ')
                usdt_part = parts.split()[0]
                bal_usdt = float(usdt_part)
                if 'BTC=' in ln:
                    btc_part = ln.split('BTC=')[-1].split()[0]
                    bal_btc = float(btc_part)
            except Exception:
                pass
            break
    print('Live balance: USDT={us} | BTC={btc}'.format(us=(f"{bal_usdt:.2f}" if bal_usdt is not None else 'N/A'), btc=(f"{bal_btc:.6f}" if bal_btc is not None else 'N/A')))

    bot_lines = tail_lines(BOT_LOG, 400)
    bsum = summarize_botlog(bot_lines)
    print('\nbot_run.log: errors={errors} warnings={warnings} trade_lines={trade_lines}'.format(**bsum))
    print('\nRecent bot log lines:')
    for ln in bsum['recent']:
        print(' ', ln)
    print('='*60 + '\n', flush=True)


if __name__ == '__main__':
    print('Starting live monitor. Reading:')
    print(' ', BOT_LOG)
    print(' ', ORDERS_LOG)
    print(' ', DEBUG_JSONL)
    print('Press Ctrl-C to stop.')
    try:
        while True:
            print_summary()
            time.sleep(60)
    except KeyboardInterrupt:
        print('Live monitor stopped.')
