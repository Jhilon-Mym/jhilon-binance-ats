from flask import send_from_directory

# webui/backend/app.py — Flask control plane for the bot
import os, threading, json, sys, subprocess
from flask import Flask, jsonify, request
# Ensure project root is on sys.path for src imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from src.bot import run as bot_run

bot_proc = None

app = Flask(__name__)
from profit_api import profit_api
from auth_api import auth_api
from run_script_api import run_script_api
from config_api import config_api
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
    # Read summary from trade_history.json and pending_trades.json
    try:
        import json
        trade_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../trade_history.json'))
        pending_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../pending_trades.json'))
        if os.path.exists(trade_path):
            with open(trade_path, 'r', encoding='utf-8', errors='replace') as f:
                trades = json.load(f)
                win = sum(1 for t in trades if t.get('result')=='win')
                loss = sum(1 for t in trades if t.get('result')=='loss')
                pnl = sum(float(t.get('pnl',0)) for t in trades if 'pnl' in t)
        if os.path.exists(pending_path):
            with open(pending_path, 'r', encoding='utf-8', errors='replace') as f:
                pendings = json.load(f)
                pending = len(pendings)
                pending_amt = sum(float(t.get('amount',0)) for t in pendings if 'amount' in t)
    except Exception:
        pass
    use_testnet = get_netmode()
    return jsonify({
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
    })

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
    return jsonify({"ok": True, "msg": "Bot started"})

@app.route("/api/stop", methods=["POST"])
def stop():
    global bot_proc
    if bot_proc is not None and bot_proc.poll() is None:
        try:
            bot_proc.terminate()
        except Exception:
            pass
        bot_proc = None
        return jsonify({"ok": True, "msg": "Bot stopped."})
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
