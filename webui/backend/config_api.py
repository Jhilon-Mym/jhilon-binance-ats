# config_api.py - Flask API for config get/set
import os, json
from flask import Blueprint, request, jsonify

config_api = Blueprint('config_api', __name__)
CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../.env'))

# Only allow these keys to be set from UI
ALLOWED_KEYS = [
    'SYMBOL','INTERVAL','HISTORY_PRELOAD','BUY_USDT_PER_TRADE','MIN_USDT_BAL','MIN_PROFIT_TO_CLOSE',
    'FAST_SMA','SLOW_SMA','ATR_LEN','EMA_HTF','ATR_SL_MULT','ATR_TP_MULT',
    'AI_MIN_CONFIDENCE_OVERRIDE','AI_OVERRIDE_PROB','INDICATOR_CONFIRM_COUNT',
    'MIN_COMBINED_SCORE','MIN_AI_WEIGHT','WIN_PROB_MIN',
    'SIGNAL_THRESHOLD'
]

def parse_env():
    cfg = {}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                if '=' in line and not line.strip().startswith('#'):
                    k,v = line.strip().split('=',1)
                    v = v.split('#',1)[0].strip()  # Remove inline comments
                    if k in ALLOWED_KEYS:
                        try:
                            v = float(v) if '.' in v or 'e' in v.lower() else int(v)
                        except Exception:
                            pass
                        cfg[k] = v
    return cfg

def update_env(new_cfg):
    # Read all lines, update allowed keys, write back
    if not os.path.exists(CONFIG_PATH):
        return False
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        if '=' in line and not line.strip().startswith('#'):
            k = line.strip().split('=',1)[0]
            if k in new_cfg:
                new_lines.append(f'{k}={new_cfg[k]}'+'\n')
                continue
        new_lines.append(line)
    # Add any new keys not present
    for k,v in new_cfg.items():
        if not any(l.strip().startswith(f'{k}=') for l in lines):
            new_lines.append(f'{k}={v}\n')
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    return True

@config_api.route('/api/config', methods=['GET'])
def get_config():
    return jsonify(parse_env())

@config_api.route('/api/config', methods=['POST'])
def set_config():
    data = request.get_json(force=True, silent=True) or {}
    # Only allow allowed keys
    new_cfg = {k: data[k] for k in ALLOWED_KEYS if k in data}
    if not new_cfg:
        return jsonify({'ok': False, 'msg': 'No valid config keys'}), 400
    ok = update_env(new_cfg)
    return jsonify({'ok': ok})
