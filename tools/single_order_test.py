#!/usr/bin/env python3
import os
import sys
BASE_DIR = r"D:\binance_ats_clone\obaidur-binance-ats-main"
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
import json

from dotenv import load_dotenv
load_dotenv()

# Force testnet + live trades for a single test
os.environ['USE_TESTNET'] = 'true'
os.environ['LIVE_TRADES'] = 'true'

from src import bot

def main():
    symbol = bot.Config.SYMBOL
    price = 112000.0
    # compute qty from BUY_USDT_PER_TRADE
    amount = bot.BUY_USDT_PER_TRADE
    from decimal import Decimal
    qty = Decimal(str(amount)) / Decimal(str(price))
    # place one BUY order (pass amount_usdt to prefer quoteOrderQty)
    print('Placing single test BUY order:', symbol, 'qty (raw decimal)=', qty)
    order = bot.place_order_live(symbol, 'BUY', qty, amount_usdt=amount, testnet=True)
    print('Order response:')
    print(order)
    # Print pending_trades.json
    try:
        with open('pending_trades.json', 'r') as f:
            print('\nPending trades file contents:')
            print(json.dumps(json.load(f), indent=2))
    except Exception as e:
        print('Could not read pending_trades.json:', e)

if __name__ == '__main__':
    main()
