#!/usr/bin/env python3
"""Start the bot (testnet live) and monitor trade_history.json for closed trades.

Usage: run from repo root:
  python tools/live_run_and_monitor.py

The script will start `python -m src.bot` as a child process (with environment vars
USE_TESTNET=true and LIVE_TRADES=true) and poll `trade_history.json` every 10s.
It reports progress every 10 new closed trades and exits after 50 new closed trades
or when the user interrupts (Ctrl-C). It will then terminate the bot process.
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time, json, subprocess, signal
from dotenv import load_dotenv
load_dotenv()
from src.config import Config

# Prefer netmode-specific history files via src.utils.get_history_file
try:
    # ensure repo root on path already added above
    from src.utils import get_history_file
    def get_monitor_history_path():
        return get_history_file()
except Exception:
    def get_monitor_history_path():
        return os.path.join(ROOT, 'trade_history.json')

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(ROOT)

TRADE_HISTORY = get_monitor_history_path()

def read_history_count():
    if not os.path.exists(TRADE_HISTORY):
        return 0
    try:
        with open(TRADE_HISTORY, 'r') as f:
            data = json.load(f)
            return len(data) if isinstance(data, list) else 0
    except Exception:
        return 0

def start_bot():
    env = os.environ.copy()
    env['USE_TESTNET'] = 'true'
    env['LIVE_TRADES'] = 'true'
    env['BUY_USDT_PER_TRADE'] = env.get('BUY_USDT_PER_TRADE', '10')
    cmd = [sys.executable, '-m', 'src.bot']
    print('Starting bot:', ' '.join(cmd))
    p = subprocess.Popen(cmd, env=env)
    return p

def main():
    print('Live monitor starting in repo:', ROOT)
    before = read_history_count()
    print('Existing closed trades in history:', before)
    target_new = 50
    report_every = 10

    bot_proc = start_bot()
    try:
        last_report = 0
        while True:
            time.sleep(10)
            now = read_history_count()
            new = now - before
            if new >= last_report + report_every:
                print(f'Progress: {new} new closed trades (total history={now})')
                last_report = new - (new % report_every)
            # also print small heartbeat
            print('.', end='', flush=True)
            if new >= target_new:
                print('\nTarget reached: new closed trades =', new)
                break
            # check bot process
            if bot_proc.poll() is not None:
                print('\nBot process exited unexpectedly with code', bot_proc.returncode)
                break
    except KeyboardInterrupt:
        print('\nInterrupted by user')
    finally:
        try:
            if bot_proc.poll() is None:
                print('Terminating bot process...')
                bot_proc.terminate()
                time.sleep(2)
                if bot_proc.poll() is None:
                    bot_proc.kill()
        except Exception as e:
            print('Error terminating bot:', e)

    # Final summary
    final = read_history_count()
    print('Final closed trades count in history:', final)
    print('Exiting monitor.')

if __name__ == '__main__':
    main()
