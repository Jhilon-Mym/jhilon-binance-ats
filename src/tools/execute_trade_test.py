#!/usr/bin/env python3
import os
import sys
import json
from decimal import Decimal

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv()

# Force testnet + live trades for this controlled test
os.environ['USE_TESTNET'] = 'true'
os.environ['LIVE_TRADES'] = 'true'

from src import bot

def main():
    # Prepare a synthetic BUY signal
    signal = {
        'side': 'BUY',
        'sl': 111900.0,
        'tp': 112100.0
    }
    price = 112000.0
    print('Calling execute_trade with signal:', signal, 'price=', price)
    bot.execute_trade(signal, price)

    print('\nOrder/pending state after execute_trade:')
    try:
        with open('pending_trades.json', 'r') as f:
            print(json.dumps(json.load(f), indent=2))
    except Exception as e:
        print('Could not read pending_trades.json:', e)

if __name__ == '__main__':
    main()
