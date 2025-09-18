#!/usr/bin/env python3
"""Test order qty calculation and execute_trade path without placing live orders.

This script imports src.bot, configures safe test parameters, fetches LOT_SIZE info,
and runs a few synthetic BUY/SELL signals to exercise qty rounding and minQty checks.

Run from project root:
  python tools/test_order_path.py
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decimal import Decimal

import time
import warnings

# Suppress DeprecationWarning from external libs during our local test runs
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Ensure test mode
os.environ['USE_TESTNET'] = os.environ.get('USE_TESTNET', 'true')
os.environ['LIVE_TRADES'] = os.environ.get('LIVE_TRADES', 'false')

from src import bot

def setup():
    # Use a small USDT-per-trade for tests
    bot.BUY_USDT_PER_TRADE = 10.0
    # Fetch symbol lot info from exchange (safe, read-only)
    try:
        min_q, step = bot.get_symbol_lot_info(bot.SYMBOL)
        bot.MIN_QTY, bot.STEP_SIZE = min_q, step
        print(f'Set LOT_SIZE: minQty={bot.MIN_QTY}, stepSize={bot.STEP_SIZE}')
    except Exception as e:
        print('Could not fetch symbol info, using defaults:', e)
        bot.MIN_QTY = Decimal('0.00001')
        bot.STEP_SIZE = Decimal('0.00001')

def run_trade(price, side='BUY'):
    # Create a fake strategy signal
    sig = {
        'side': side,
        'sl': float(Decimal(str(price)) - Decimal('100')), 
        'tp': float(Decimal(str(price)) + Decimal('100')),
        'win_prob': 0.9
    }
    print('\n--- Testing', side, 'at price', price)
    bot.execute_trade(sig, price)

def main():
    print('Running order-path tests')
    setup()
    # Test at a typical BTC price
    run_trade(112000.0, 'BUY')
    # Test at a much lower price (bigger qty)
    run_trade(30000.0, 'BUY')
    # Test SELL path
    run_trade(112000.0, 'SELL')
    # Test with BUY_USDT_PER_TRADE too small to meet minQty
    bot.BUY_USDT_PER_TRADE = 0.0001
    print('\nSetting BUY_USDT_PER_TRADE to very small value to check minQty skip')
    run_trade(112000.0, 'BUY')
    print('\nFinal stats:')
    print(bot.stats.summary())

if __name__ == '__main__':
    main()
