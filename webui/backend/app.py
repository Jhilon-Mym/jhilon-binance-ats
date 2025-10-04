from flask import Flask, jsonify, request, session
import sys as _sys

app = Flask(__name__)

@app.route('/api/force_sell', methods=['POST'])
def force_sell():
    """Manually close all pending trades with profit >= MIN_PROFIT_TO_CLOSE."""
    try:
        from src.config import Config
        from src.utils import get_pending_file, get_history_file, ensure_pending_fields, ensure_history_fields
        import json, os, time
        min_profit = getattr(Config, 'MIN_PROFIT_TO_CLOSE', 0.015)
        pending_file = get_pending_file()
        history_file = get_history_file()
        closed = 0
        # Load pending trades
        if os.path.exists(pending_file):
            with open(pending_file, 'r', encoding='utf-8') as f:
                pendings = json.load(f)
        else:
            pendings = []
        new_pending = []
        history = []
        if os.path.exists(history_file):
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        for t in pendings:
            t = ensure_pending_fields(t)
            entry = float(t.get('entry', 0))
            price = float(t.get('tp', entry))
            amount = float(t.get('amount_usdt', 0))
            side = t.get('side', 'BUY')
            # For BUY: profit = (price-entry)/entry; For SELL: profit = (entry-price)/entry
            if side == 'BUY':
                pnl_pct = (price-entry)/entry if entry else 0.0
            else:
                pnl_pct = (entry-price)/entry if entry else 0.0
            if pnl_pct >= min_profit:
                t['status'] = 'win'
                t['close_price'] = price
                t['realized_pnl_usdt'] = pnl_pct * amount
                t['timestamp'] = time.strftime('%Y-%m-%dT%H:%M:%S')
                history.append(ensure_history_fields(t))
                closed += 1
            else:
                new_pending.append(t)
        # Save updated pending and history
        with open(pending_file, 'w', encoding='utf-8') as f:
            json.dump(new_pending, f, indent=2, ensure_ascii=False)
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        # Refresh profit tracker cached balances so UI status is updated quickly
        try:
            from src.utils import get_profit_tracker
            pt = get_profit_tracker()
            try:
                pt.print_balances()
            except Exception:
                pass
        except Exception:
            pass
        return jsonify({'ok': True, 'closed': closed})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})

# Require modern Python (f-strings and other syntax used). Provide clear error
# if an old interpreter (e.g., Python 2.x) is used so users don't see a vague
# SyntaxError. Adjust minimum version if needed.
if _sys.version_info < (3, 8):
    raise RuntimeError(f"Python 3.8+ is required to run this web UI (found {_sys.version_info.major}.{_sys.version_info.minor}). Please use 'py -3' or a Python 3.8+ interpreter.")


app = Flask(__name__)
bot_proc = None
# Use a secret key for Flask session. Preferably set FLASK_SECRET in environment for production.
import os as _os
app.secret_key = _os.getenv('FLASK_SECRET', 'change-me-in-prod')
from datetime import timedelta as _timedelta
# Session lifetime: default 2 hours
app.permanent_session_lifetime = _timedelta(hours=int(_os.getenv('SESSION_HOURS', '2')))

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
                "pending_amt": pending_amt,
                "usdt": usdt,
                "btc": btc,
                # balance_event=True indicates this payload was triggered by a bot
                # action (e.g. trade executed/closed) and balances should be
                # applied by clients. Periodic emits will set this False.
                "balance_event": True,
                "use_testnet": get_netmode(),
                "running": running
            }
            socketio.emit('status_update', status_data)
    except Exception as e:
        print('[ERROR] emit_status_update failed:', e)
@app.route('/api/notify_update', methods=['POST'])
def notify_update():
    """Lightweight endpoint for external processes (bot) to request an immediate status emit.

    The bot runs in a separate process; after it writes pending/history files it can POST
    to this endpoint (localhost) and the backend will emit an immediate status update
    via Socket.IO so the UI reflects changes without waiting for the next poll.
    """
    try:
        # Rebuild in-memory profit counts from persisted history so web UI shows
        # accurate win/loss/pnl even though the bot runs in a separate process.
        try:
            from src.utils import get_history_file, get_profit_tracker
            import json
            hf = get_history_file()
            if os.path.exists(hf):
                try:
                    with open(hf, 'r', encoding='utf-8') as f:
                        trades = json.load(f)
                except Exception:
                    trades = []
                win = sum(1 for t in trades if str(t.get('status')).lower() == 'win')
                loss = sum(1 for t in trades if str(t.get('status')).lower() == 'loss')
                realized_sum = 0.0
                total_fees = 0.0
                for t in trades:
                    try:
                        realized_sum += float(t.get('realized_pnl_usdt') or 0)
                    except Exception:
                        pass
                    try:
                        total_fees += float(t.get('fee') or 0)
                    except Exception:
                        pass
                # Debug print to help diagnose why UI summary may remain zero
                try:
                    print(f"[NOTIFY] trades={len(trades)} win={win} loss={loss} realized_sum={realized_sum} total_fees={total_fees}")
                except Exception:
                    pass
                tracker = get_profit_tracker()
                try:
                    tracker.win = int(win)
                    tracker.loss = int(loss)
                    # Set balance to realized_sum to show cumulative PnL in UI summary
                    tracker.balance = float(realized_sum)
                    # Also update embedded ProfitTracker if present
                    if hasattr(tracker, 'profit_tracker') and tracker.profit_tracker is not None:
                        tracker.profit_tracker.realized_pnl = float(realized_sum)
                        tracker.profit_tracker.total_fees = float(total_fees)
                        # Build a simple equity_curve from trades
                        try:
                            tracker.profit_tracker.equity_curve = []
                            cum = 0.0
                            import datetime as _dt
                            for t in trades:
                                try:
                                    rp = float(t.get('realized_pnl_usdt') or 0)
                                except Exception:
                                    rp = 0.0
                                cum += rp
                                ts = t.get('timestamp') or t.get('closed_at')
                                try:
                                    dt = _dt.datetime.fromisoformat(ts) if ts else _dt.datetime.now()
                                except Exception:
                                    dt = _dt.datetime.now()
                                tracker.profit_tracker.equity_curve.append({'dt': dt, 'equity': cum, 'mark_price': None})
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception:
            # Non-fatal: proceed to emitting status
            pass

        try:
            # Build an explicit status payload from persisted files so the web UI
            # receives accurate win/loss/pnl immediately (don't only rely on
            # profit_tracker which may be stale across processes).
            try:
                pending_file = get_pending_file()
                pending_list = []
                if os.path.exists(pending_file):
                    with open(pending_file, 'r', encoding='utf-8') as pf:
                        pdata = json.load(pf)
                    if isinstance(pdata, dict) and 'open_trades' in pdata:
                        pending_list = pdata['open_trades']
                    elif isinstance(pdata, list):
                        pending_list = pdata
                pending_cnt = len([t for t in pending_list if t.get('status') == 'open'])
                pending_amt = sum(t.get('amount_usdt', 0) for t in pending_list if t.get('status') == 'open')
            except Exception:
                pending_cnt = 0
                pending_amt = 0

            # Use tracker cached balances when available
            tracker = None
            try:
                tracker = get_profit_tracker()
            except Exception:
                tracker = None

            usdt = btc = 'N/A'
            win_val = win
            loss_val = loss
            pnl_val = realized_sum
            try:
                if tracker is not None:
                    ub = tracker.cached_balances.get('USDT', None)
                    bb = tracker.cached_balances.get('BTC', None)
                    if isinstance(ub, (int, float)):
                        usdt = f"{ub:.2f}"
                    if isinstance(bb, (int, float)):
                        btc = f"{bb:.6f}"
            except Exception:
                pass

            status_data = {
                'win': int(win_val),
                'loss': int(loss_val),
                'pnl': float(pnl_val),
                'pending': int(pending_cnt),
                'pending_amt': float(pending_amt),
                'usdt': usdt,
                'btc': btc,
                # This notify endpoint was triggered by the bot writing files,
                # so mark balance_event True so the frontend updates balances.
                'balance_event': True,
                'use_testnet': get_netmode(),
                'running': bot_proc is not None and bot_proc.poll() is None
            }
            try:
                # Log the exact payload we emit so it can be inspected in server logs
                try:
                    print('[NOTIFY_EMIT]', json.dumps(status_data, default=str))
                except Exception:
                    try: print('[NOTIFY_EMIT]', status_data)
                    except Exception: pass
                socketio.emit('status_update', status_data)
            except Exception:
                # fallback to the existing emit function which handles debouncing
                try:
                    emit_status_update()
                except Exception:
                    pass
        except Exception:
            pass
        return jsonify({'ok': True})
    except Exception:
        pass

@app.route('/api/status_from_history')
def status_from_history():
    """Return a status payload computed from persisted history and pending files.
    This is a diagnostic helper so clients can query the server for the exact
    values the server would emit.
    """
    try:
        from src.utils import get_history_file, get_pending_file
        # compute trades summary
        hf = get_history_file()
        trades = []
        if os.path.exists(hf):
            try:
                with open(hf, 'r', encoding='utf-8') as f:
                    trades = json.load(f)
            except Exception:
                trades = []
        win = sum(1 for t in trades if str(t.get('status')).lower() == 'win')
        loss = sum(1 for t in trades if str(t.get('status')).lower() == 'loss')
        realized_sum = 0.0
        for t in trades:
            try:
                realized_sum += float(t.get('realized_pnl_usdt') or 0)
            except Exception:
                pass
        # pending
        pending_file = get_pending_file()
        pending_list = []
        if os.path.exists(pending_file):
            try:
                with open(pending_file, 'r', encoding='utf-8') as pf:
                    pdata = json.load(pf)
                if isinstance(pdata, dict) and 'open_trades' in pdata:
                    pending_list = pdata['open_trades']
                elif isinstance(pdata, list):
                    pending_list = pdata
            except Exception:
                pending_list = []
        pending_cnt = len([t for t in pending_list if t.get('status') == 'open'])
        pending_amt = sum(t.get('amount_usdt', 0) for t in pending_list if t.get('status') == 'open')
        # balances best-effort
        usdt = btc = 'N/A'
        try:
            tracker = get_profit_tracker()
            ub = tracker.cached_balances.get('USDT', None)
            bb = tracker.cached_balances.get('BTC', None)
            if isinstance(ub, (int, float)): usdt = f"{ub:.2f}"
            if isinstance(bb, (int, float)): btc = f"{bb:.6f}"
        except Exception:
            pass
        payload = {
            'win': int(win), 'loss': int(loss), 'pnl': float(realized_sum),
            'pending': int(pending_cnt), 'pending_amt': float(pending_amt),
            'usdt': usdt, 'btc': btc, 'use_testnet': get_netmode(),
            'running': bot_proc is not None and bot_proc.poll() is None
        }
        return jsonify(payload)
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

from flask import send_from_directory
from src.utils import get_history_file
from flask import request

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

# Reconcile synchronization primitives
import threading as _thr
_reconcile_lock = _thr.Lock()

def _run_reconcile_with_timeout(timeout=20, symbol=None):
    """Run reconcile_pending_with_exchange in a background thread with timeout.

    Returns a summary dict from the utils wrapper or a timeout/error dict.
    """
    from src.utils import reconcile_pending_with_exchange
    result = {'updated': 0, 'removed': 0}
    def _target():
        try:
            # build client using env (reuse the utility in utils if needed)
            from binance.client import Client
            import os
            api_key = os.getenv('BINANCE_API_KEY')
            api_secret = os.getenv('BINANCE_API_SECRET')
            use_testnet = os.getenv('USE_TESTNET', 'true').lower() == 'true'
            client = Client(api_key, api_secret, testnet=use_testnet)
            r = reconcile_pending_with_exchange(client, symbol=symbol)
            if isinstance(r, dict):
                result.update(r)
        except Exception as e:
            result['error'] = str(e)

    if not _reconcile_lock.acquire(blocking=False):
        return {'updated': 0, 'removed': 0, 'error': 'reconcile busy'}
    try:
        th = _thr.Thread(target=_target, daemon=True)
        th.start()
        th.join(timeout)
        if th.is_alive():
            return {'updated': result.get('updated',0), 'removed': result.get('removed',0), 'error': 'timeout'}
        return result
    finally:
        try:
            _reconcile_lock.release()
        except Exception:
            pass

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
    from src.utils import get_data_file
    log_path = get_data_file('bot_run.log')
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
    # Run a guarded reconcile before starting the bot to sync pending/history
    try:
        recon = _run_reconcile_with_timeout(timeout=20, symbol=None)
        print(f"[START] Reconcile result before bot start: {recon}")
    except Exception as e:
        print(f"[START] Reconcile failed (ignored): {e}")

    # Run bot.py as subprocess, log output to bot_run.log
    from src.utils import get_data_file
    bot_log = get_data_file('bot_run.log')
    # Truncate log for new session
    emit_status_update()  # Emit status after bot start
    try:
        with open(bot_log, 'w', encoding='utf-8') as f:
            f.write('')
    except Exception:
        pass
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
    log_tail = ''
    if bot_proc.poll() is not None and not started:
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
        return jsonify({"ok": False, "msg": "Bot process exited immediately", "log_tail": log_tail, "reconcile": recon}), 500

    # started may be False but process alive - accept as started with a warning
    return jsonify({"ok": True, "msg": "Bot started", "started_log_confirmed": bool(started), "reconcile": recon})


@app.route('/api/reconcile', methods=['POST'])
def api_reconcile():
    """Manual reconcile endpoint. Builds a client from env and runs reconcile with a timeout."""
    # minimal auth could be added here
    data = request.get_json(force=True, silent=True) or {}
    timeout = int(data.get('timeout', 20))
    # Authentication: prefer session-based login. For backwards-compat, if
    # RECONCILE_TOKEN is configured then a matching token is accepted when the
    # session is not present.
    import os
    expected = os.getenv('RECONCILE_TOKEN', None)
    # If user not logged-in via session, require token when configured.
    if not session.get('logged_in'):
        if expected:
            token = request.headers.get('X-RECONCILE-TOKEN') or data.get('token')
            if not token or token != expected:
                return jsonify({'ok': False, 'error': 'unauthorized'}), 401
        else:
            return jsonify({'ok': False, 'error': 'unauthorized'}), 401
    try:
        res = _run_reconcile_with_timeout(timeout=timeout, symbol=None)
        return jsonify({'ok': True, 'result': res})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

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


@app.after_request
def _maybe_add_dev_csp(response):
    """Development helper: allow 'unsafe-eval' for localhost requests only.

    This mitigates the browser 'Content Security Policy blocks eval' issue
    when running the UI locally for development. It does NOT apply to remote
    hosts. Do NOT enable this in production.
    """
    try:
        host = (request.host or '').split(':')[0]
        remote = request.remote_addr or ''
        if host in ('127.0.0.1', 'localhost') or remote in ('127.0.0.1', '::1'):
            # Dev-only CSP: permit unsafe-eval and inline scripts for quick local debugging
            csp = (
                "default-src 'self' http: https:; "
                "script-src 'self' 'unsafe-eval' 'unsafe-inline' https:; "
                "connect-src 'self' ws: wss: http: https:;"
            )
            response.headers['Content-Security-Policy'] = csp
    except Exception:
        pass
    return response
