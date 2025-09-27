#!/usr/bin/env python3
import os
import sys
import json

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv()

os.environ['USE_TESTNET'] = 'true'
os.environ['LIVE_TRADES'] = 'true'

from src import bot

def main():
    # Synthetic SELL signal — attempt to sell entire free base balance (pre-flight should quantize)
    signal = {'side': 'SELL', 'sl': 99999.0, 'tp': 99999.0}
    price = 112000.0
    print('Calling execute_trade with SELL signal (testnet)')
    bot.execute_trade(signal, price)

    print('\nPending after SELL test:')
    try:
        with open('pending_trades.json', 'r') as f:
            print(json.dumps(json.load(f), indent=2))
    except Exception as e:
        print('Could not read pending_trades.json:', e)

if __name__ == '__main__':
    main()
