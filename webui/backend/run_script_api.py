import subprocess, os
from flask import Blueprint, jsonify

run_script_api = Blueprint('run_script_api', __name__)

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

def run_script(script_path):
    try:
        proc = subprocess.Popen([os.environ.get('PYTHON_EXEC','python'), script_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=BASE)
        out, _ = proc.communicate(timeout=60)
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
    script = os.path.join(BASE, 'training', 'train_model.py')
    output = run_script(script)
    return jsonify({'output': output})
