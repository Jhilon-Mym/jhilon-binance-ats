#!/usr/bin/env python3
"""Run a short, safe smoke test of src.bot.main() in daemon mode.

This starts bot.main() in a daemon thread, waits for `RUN_SECONDS`, then exits.
It relies on the bot printing its startup messages (seeded history, LOT_SIZE, etc.).

Usage: from project root:
  python tools/smoke_run.py
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import threading

from dotenv import load_dotenv
load_dotenv()
from src.config import Config

# Force test settings for a safe run
os.environ['USE_TESTNET'] = os.environ.get('USE_TESTNET', 'true')
os.environ['LIVE_TRADES'] = os.environ.get('LIVE_TRADES', 'false')

RUN_SECONDS = 70

def run_bot_for_seconds(seconds=RUN_SECONDS):
    try:
        import src.bot as bot
    except Exception as e:
        print('Failed to import src.bot:', e)
        raise

    th = threading.Thread(target=bot.main, daemon=True)
    print(f'Smoke-run: starting bot.main() in daemon thread for ~{seconds}s')
    th.start()
    try:
        remaining = seconds
        while remaining > 0:
            time.sleep(1)
            remaining -= 1
    except KeyboardInterrupt:
        print('Smoke-run: interrupted by user')
    print('Smoke-run: stopping (thread is daemon; process will exit)')

if __name__ == '__main__':
    run_bot_for_seconds()
import os
import sys
import threading
import time

# ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# force testnet in environment
os.environ['USE_TESTNET'] = 'true'

from src import bot

print('Starting bot.main() in background thread (smoke-run ~70s)...')
thr = threading.Thread(target=bot.main, daemon=True)
thr.start()
# Let it run for 70 seconds
for i in range(7):
    print(f'smoke-run: tick {i+1}/7')
    time.sleep(10)
print('Smoke-run complete; exiting.')
