from flask import Flask, jsonify, request


app = Flask(__name__)
bot_proc = None

@app.route('/api/pending_trades')
def pending_trades():
    """Return netmode-specific pending trades as JSON object with 'pendings' key (empty list if missing)."""
    try:
        pending_file = get_pending_file()
        pendings = []
        print(f"[DEBUG] pending_trades API: loading from {pending_file}")
        if os.path.exists(pending_file):
            with open(pending_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                pendings = data
            # some older formats may store a dict; try to extract list
            elif isinstance(data, dict) and 'open_trades' in data and isinstance(data['open_trades'], list):
                pendings = data['open_trades']
        print(f"[DEBUG] pending_trades API: found {len(pendings)} trades")
        # Ensure all fields (including timestamp) are present
        from src.utils import ensure_pending_fields
        pendings = [ensure_pending_fields(t) for t in pendings]
        # Calculate Pending Qty (only open trades)
        pending_qty = sum(t.get('qty', 0) for t in pendings if t.get('status') == 'open')
        return jsonify({'pendings': pendings, 'pending_qty': pending_qty})
    except Exception as e:
        print('[ERROR] pending_trades API:', e)
        return jsonify({'pendings': [], 'pending_qty': 0}), 500

import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import json
from src.utils import get_profit_tracker, get_pending_file, get_netmode

import threading
_status_emit_lock = threading.Lock()
_last_status_emit = 0
def emit_status_update():
    global _last_status_emit
    try:
        with _status_emit_lock:
            now = time.time()
            # debounce: 1.5s interval (hardened)
            if now - _last_status_emit < 1.5:
                return
            # Only emit status if bot_proc is running or just started
            global bot_proc
            running = bot_proc is not None and bot_proc.poll() is None
            if not running:
                # If not running, only emit status once every 10s
                if now - _last_status_emit < 10.0:
                    return
            _last_status_emit = now
            pending_file = get_pending_file()
            profit_tracker = get_profit_tracker()
            # Read pending trades from file
            if os.path.exists(pending_file):
                with open(pending_file, 'r') as f:
                    pending_data = json.load(f)
                if isinstance(pending_data, dict) and 'open_trades' in pending_data:
                    pending_list = pending_data['open_trades']
                elif isinstance(pending_data, list):
                    pending_list = pending_data
                else:
                    pending_list = []
            else:
                pending_list = []
            pending = len([t for t in pending_list if t.get('status') == 'open'])
            pending_amt = sum(t.get('amount_usdt', 0) for t in pending_list if t.get('status') == 'open')
            win = getattr(profit_tracker, 'win', 0)
            loss = getattr(profit_tracker, 'loss', 0)
            pnl = getattr(profit_tracker, 'balance', 0)
            # Balance format always numeric or 'N/A'
            # --- Balance caching: None হলে আগের মান দেখাও ---
            if not hasattr(emit_status_update, '_last_usdt'): emit_status_update._last_usdt = None
            if not hasattr(emit_status_update, '_last_btc'): emit_status_update._last_btc = None
            usdt_val = profit_tracker.cached_balances.get('USDT', None)
            btc_val = profit_tracker.cached_balances.get('BTC', None)
            # --- আরও শক্তিশালী caching ---
            def valid_balance(val):
                return isinstance(val, (int, float)) and val is not None and not (isinstance(val, float) and (val != val or val == float('inf') or val == float('-inf')))
            # --- আরও robust caching: একবার N/A হলে আর ব্যালেন্সে ফ্লিপ করবে না ---
            if valid_balance(usdt_val):
                usdt = f"{usdt_val:.2f}"
                emit_status_update._last_usdt = usdt
            else:
                # যদি আগের মান valid (N/A না), তাহলে সেটাই দেখাও, নইলে শুধু প্রথমবার N/A
                if emit_status_update._last_usdt and emit_status_update._last_usdt != "N/A":
                    usdt = emit_status_update._last_usdt
                else:
                    usdt = "N/A"
            if valid_balance(btc_val):
                btc = f"{btc_val:.6f}"
                emit_status_update._last_btc = btc
            else:
                if emit_status_update._last_btc and emit_status_update._last_btc != "N/A":
                    btc = emit_status_update._last_btc
                else:
                    btc = "N/A"
            status_data = {
                "win": win,
                "loss": loss,
                "pnl": pnl,
                "pending": pending,
                "pending_amt": pending_amt,
                "usdt": usdt,
                "btc": btc,
                "use_testnet": get_netmode(),
                "running": running
            }
            socketio.emit('status_update', status_data)
    except Exception as e:
        print('[ERROR] emit_status_update failed:', e)
from flask import send_from_directory
from src.utils import get_history_file

# webui/backend/app.py — Flask control plane for the bot
import os, threading, json, sys, subprocess
# Ensure src folder is in sys.path for imports
SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src'))
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

## Removed direct import of profit_tracker; use get_profit_tracker() everywhere
from flask_socketio import SocketIO, emit
from src.bot import run as bot_run
socketio = SocketIO(app, cors_allowed_origins="*")

# --- Real-time balance update background thread ---
import time
def emit_realtime_balance():
    last_usdt = last_btc = None
    while True:
        try:
            tracker = get_profit_tracker()
            tracker.print_balances()
            # Emit status update after balance refresh (debounced)
            emit_status_update()
            usdt = tracker.cached_balances.get('USDT', None)
            btc = tracker.cached_balances.get('BTC', None)
            last_usdt, last_btc = usdt, btc
        except Exception:
            pass
        time.sleep(2)  # Poll every 2 seconds

def start_balance_thread():
    t = threading.Thread(target=emit_realtime_balance, daemon=True)
    t.start()

start_balance_thread()
from webui.backend.profit_api import profit_api
from webui.backend.auth_api import auth_api
from webui.backend.run_script_api import run_script_api
from webui.backend.config_api import config_api
app.register_blueprint(profit_api)
app.register_blueprint(auth_api)
app.register_blueprint(run_script_api)
app.register_blueprint(config_api)
bot_thread = None
NETMODE_PATH = os.path.join(os.path.dirname(__file__), '../../.env')

def set_netmode_env(use_testnet: bool):
    # Update .env file and process env
    # Read all lines, update USE_TESTNET, write back
    try:
        with open(NETMODE_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        new_lines = []
        found = False
        for line in lines:
            if line.strip().startswith('USE_TESTNET='):
                new_lines.append(f'USE_TESTNET={"true" if use_testnet else "false"}\n')
                found = True
            else:
                new_lines.append(line)
        if not found:
            new_lines.append(f'USE_TESTNET={"true" if use_testnet else "false"}\n')
        with open(NETMODE_PATH, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        os.environ['USE_TESTNET'] = 'true' if use_testnet else 'false'
    except Exception as e:
        print('Failed to update netmode in .env:', e)

def get_netmode():
    # Read USE_TESTNET from .env
    try:
        with open(NETMODE_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip().startswith('USE_TESTNET='):
                    val = line.strip().split('=',1)[1].lower()
                    return val == 'true'
    except Exception:
        pass
    return True

@app.route("/api/status")
def status():
    # Debug: print pending_list and pending values for diagnosis
    running = bot_proc is not None and bot_proc.poll() is None
    error_msg = ''
    usdt = btc = win = loss = pnl = pending = pending_amt = '...'
    terminal = ''
    # Read bot_run.log, filter only important tags (timestamp preserved)
    log_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../bot_run.log'))
    important_tags = ['[INFO]', '[ERROR]', '[BALANCE]', '[SIGNAL]', '[TRADE]', '[SKIP-LIVE]', '[SHUTDOWN]', '[DONE]', '[CANDLES]', '[WebSocket]']
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
                all_lines = f.readlines()
                filtered_lines = [l for l in all_lines if any(tag in l for tag in important_tags)]
                terminal = ''.join(filtered_lines[-40:])
                # Ensure the 'Bot started | netmode=...' line (often on startup) is included.
                # If present in the raw log, map netmode=true->testnet and false->mainnet and prepend it.
                import re
                single_line = ''
                # Prefer the explicit third line of the log when available (user requested)
                try:
                    if len(all_lines) >= 3:
                        third = all_lines[2].strip()
                        if 'netmode=' in third or 'Bot started' in third:
                            safe = re.sub(r'API_KEY=[^|\s]+', 'API_KEY=***', third)
                            safe = re.sub(r'API_SECRET=[^|\s]+', 'API_SECRET=***', safe)
                            safe = safe.replace('netmode=true', 'netmode=testnet').replace('netmode=false', 'netmode=mainnet')
                            single_line = safe
                except Exception:
                    single_line = ''
                # fallback: if third-line not suitable, try older detection
                if not single_line:
                    for l in all_lines:
                        if 'Bot started' in l and 'netmode=' in l:
                            safe = re.sub(r'API_KEY=[^|\s]+', 'API_KEY=***', l.strip())
                            safe = re.sub(r'API_SECRET=[^|\s]+', 'API_SECRET=***', safe)
                            safe = safe.replace('netmode=true', 'netmode=testnet').replace('netmode=false', 'netmode=mainnet')
                            single_line = safe
                            break
                if single_line:
                    # avoid duplicating if it's already present in terminal
                    if single_line not in terminal:
                        terminal = single_line + '\n' + terminal
                # Get last balance
                for line in reversed(all_lines):
                    if '[BALANCE]' in line:
                        parts = line.split('USDT=')
                        if len(parts)>1:
                            usdt = parts[1].split('|')[0].strip()
                        if 'BTC=' in line:
                            btc = line.split('BTC=')[1].split()[0].strip()
                        break
                # Get last error
                for line in reversed(filtered_lines):
                    if '[ERROR]' in line:
                        error_msg = line.strip()
                        break
        except Exception:
            pass
    # Read summary using netmode-specific profit_tracker and pending file
    pending_file = get_pending_file()
    profit_tracker = get_profit_tracker()
    pending_list = []
    pending = 0
    pending_amt = 0
    win = 0
    loss = 0
    pnl = 0
    try:
        if os.path.exists(pending_file):
            with open(pending_file, 'r') as f:
                data = json.load(f)
            if isinstance(data, dict) and 'open_trades' in data:
                pending_list = data['open_trades']
            elif isinstance(data, list):
                pending_list = data
            pending = len([t for t in pending_list if t.get('status') == 'open'])
            pending_amt = sum(t.get('amount_usdt', 0) for t in pending_list if t.get('status') == 'open')
        win = getattr(profit_tracker, 'win', 0)
        loss = getattr(profit_tracker, 'loss', 0)
        pnl = getattr(profit_tracker, 'balance', 0)
    except Exception as e:
        print('[ERROR] summary logic:', e)
    use_testnet = get_netmode()
    status_data = {
        "running": running,
        "usdt": usdt,
        "btc": btc,
        "win": win,
        "loss": loss,
        "pnl": pnl,
        "pending": pending,
        "pending_amt": pending_amt,
        "terminal": terminal,
        "use_testnet": use_testnet,
        "error_msg": error_msg
    }
    # Emit status update to all connected clients
    socketio.emit('status_update', status_data)
    return jsonify(status_data)


@app.route('/api/trade_history')
def trade_history():
    """Return netmode-specific trade history as JSON object with 'trades' key (empty list if missing)."""
    try:
        history_file = get_history_file()
        trades = []
        if os.path.exists(history_file):
            with open(history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                trades = data
            # some older formats may store a dict; try to extract list
            elif isinstance(data, dict) and 'history' in data and isinstance(data['history'], list):
                trades = data['history']
        return jsonify({'trades': trades})
    except Exception as e:
        print('[ERROR] trade_history API:', e)
        return jsonify({'trades': []}), 500

@app.route("/api/start", methods=["POST"])
def start():
    global bot_proc
    data = request.get_json(force=True, silent=True) or {}
    netmode = data.get('netmode', None)
    if netmode:
        set_netmode_env(netmode == 'testnet')
    use_testnet = get_netmode()
    import dotenv
    dotenv.load_dotenv(NETMODE_PATH, override=True)
    if use_testnet:
        os.environ['BINANCE_API_KEY'] = os.environ.get('TESTNET_API_KEY','')
        os.environ['BINANCE_API_SECRET'] = os.environ.get('TESTNET_API_SECRET','')
    else:
        os.environ['BINANCE_API_KEY'] = os.environ.get('MAINNET_API_KEY','')
        os.environ['BINANCE_API_SECRET'] = os.environ.get('MAINNET_API_SECRET','')
    if not os.environ['BINANCE_API_KEY'] or not os.environ['BINANCE_API_SECRET']:
        return jsonify({"ok": False, "msg": "API KEY/SECRET missing for selected netmode!"})
    if bot_proc is not None and bot_proc.poll() is None:
        return jsonify({"ok": True, "msg": "Already running"})
    # Run bot.py as subprocess, log output to bot_run.log
    bot_log = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../bot_run.log'))
    # Truncate log for new session
    emit_status_update()  # Emit status after bot start
    with open(bot_log, 'w', encoding='utf-8') as f:
        f.write('')
    logf = open(bot_log, 'a', encoding='utf-8', errors='replace', buffering=1)
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    bot_proc = subprocess.Popen([
        sys.executable, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/bot.py'))
    ], env=env, stdout=logf, stderr=logf, bufsize=1, cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    import atexit
    def close_log():
        try:
            logf.close()
        except Exception:
            pass
    atexit.register(close_log)
    # Wait briefly to ensure the bot actually starts (avoid UI flipping running/stopped)
    import time
    started = False
    start_wait_secs = 6
    try:
        for _ in range(int(start_wait_secs * 2)):
            time.sleep(0.5)
            # if process exited early, break
            if bot_proc.poll() is not None:
                break
            try:
                with open(bot_log, 'r', encoding='utf-8', errors='replace') as bf:
                    tail = bf.read()
                if 'Bot started' in tail or 'Bot started |' in tail:
                    started = True
                    break
            except Exception:
                pass
    except Exception:
        pass

    # If process exited immediately, capture tail and return error
    if bot_proc.poll() is not None and not started:
        log_tail = ''
        try:
            with open(bot_log, 'r', encoding='utf-8', errors='replace') as bf:
                lines = bf.readlines()
                log_tail = ''.join(lines[-80:])
        except Exception:
            log_tail = ''
        try:
            logf.close()
        except Exception:
            pass
        bot_proc = None
        return jsonify({"ok": False, "msg": "Bot process exited immediately", "log_tail": log_tail}), 500

    # started may be False but process alive - accept as started with a warning
    return jsonify({"ok": True, "msg": "Bot started", "started_log_confirmed": bool(started)})

@app.route("/api/stop", methods=["POST"])
def stop():
    global bot_proc
    if bot_proc is not None and bot_proc.poll() is None:
        try:
            bot_proc.terminate()
        except Exception:
            pass
        bot_proc = None
        emit_status_update()  # Push status to all clients after stop
        return jsonify({"ok": True, "msg": "Bot stopped."})
    emit_status_update()  # Push status even if not running
    return jsonify({"ok": False, "msg": "Bot not running."})

@app.route("/api/set_netmode", methods=["POST"])
def set_netmode():
    data = request.get_json(force=True, silent=True) or {}
    netmode = data.get('netmode', None)
    if netmode:
        set_netmode_env(netmode == 'testnet')
        return jsonify({"ok": True, "msg": f"Netmode set to {netmode}"})
    return jsonify({"ok": False, "msg": "Missing netmode"})


# Static/frontend serve
@app.route("/")
def root():
    return send_from_directory(os.path.abspath(os.path.join(os.path.dirname(__file__), '../frontend')), 'index.html')

@app.route('/<path:path>')
def static_proxy(path):
    static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../frontend'))
    return send_from_directory(static_dir, path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
