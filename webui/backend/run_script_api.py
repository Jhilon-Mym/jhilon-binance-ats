import subprocess, os
from flask import Blueprint, jsonify

run_script_api = Blueprint('run_script_api', __name__)

BASE = r"D:\binance_ats_clone\obaidur-binance-ats-main"

def run_script(script_path):
    try:
        proc = subprocess.Popen([os.environ.get('PYTHON_EXEC','python'), script_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=BASE)
        # Allow longer-running retrain tasks (e.g., epochs) — 20 minutes timeout
        out, _ = proc.communicate(timeout=20*60)
        return out.decode('utf-8', errors='replace')
    except Exception as e:
        return f'Error: {e}'

@run_script_api.route('/api/run_debug_signal')
def run_debug_signal():
    script = os.path.join(BASE, 'tools', 'debug_signal.py')
    output = run_script(script)
    return jsonify({'output': output})

@run_script_api.route('/api/run_retrain')
def run_retrain():
    script = os.path.join(BASE, 'training', 'train_hybrid.py')
    output = run_script(script)
    return jsonify({'output': output})
