#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Force testnet + live trades for this run
os.environ['USE_TESTNET'] = 'true'
os.environ['LIVE_TRADES'] = 'true'

from src import bot

def main():
    print('Starting bot in testnet LIVE_TRADES mode...')
    bot.main()

if __name__ == '__main__':
    main()
