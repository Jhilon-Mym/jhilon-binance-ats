"""Simple script to run the bot in paper mode for manual testing.

Usage: python scripts/run_bot.py

This script sets conservative defaults for environment variables to run
the bot in paper mode (no live trades). It assumes a working virtualenv
and that dependencies are installed.
"""
import os
import subprocess
import sys


def main():
    # ensure paper mode defaults
    os.environ.setdefault('LIVE_TRADES', 'false')
    os.environ.setdefault('USE_TESTNET', 'true')
    os.environ.setdefault('SYMBOL', 'BTCUSDT')
    os.environ.setdefault('INTERVAL', '5m')
    os.environ.setdefault('BUY_USDT_PER_TRADE', '20')

    # run the bot module
    here = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    py = sys.executable
    cmd = [py, '-m', 'src.bot']
    print('Running bot in paper mode (LIVE_TRADES=false). Command:', ' '.join(cmd))
    subprocess.run(cmd)


if __name__ == '__main__':
    main()
