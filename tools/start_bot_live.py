#!/usr/bin/env python3
import os
import sys
BASE_DIR = r"D:\binance_ats_clone\obaidur-binance-ats-main"
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Force testnet + live trades for this run
os.environ['USE_TESTNET'] = 'true'
os.environ['LIVE_TRADES'] = 'true'

from src import bot

def main():
    print('Starting bot in testnet LIVE_TRADES mode...')
    bot.main()

if __name__ == '__main__':
    main()
