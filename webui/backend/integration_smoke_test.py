"""Integration smoke test: start -> reconcile -> stop.

This test simulates starting the bot by monkeypatching subprocess.Popen so
the app behaves as if the bot started (it writes a 'Bot started' marker to
bot_run.log). It then calls /api/start, /api/reconcile and /api/stop using the
Flask test client. If real Binance keys are present in .env, reconcile will run
against the testnet keys.
"""
import os, sys, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

import dotenv
dotenv.load_dotenv(os.path.join(ROOT, '.env'))

from webui.backend import app as flask_app_mod

class DummyProc:
    def __init__(self, logpath):
        self._log = logpath
        self._alive = True
        # write a startup line so start() can detect it
        try:
            with open(self._log, 'a', encoding='utf-8') as f:
                f.write('2025-10-01T00:00:00Z Bot started | netmode=testnet\n')
        except Exception:
            pass
    def poll(self):
        return None if self._alive else 0
    def terminate(self):
        self._alive = False

def fake_popen(cmd, env=None, stdout=None, stderr=None, bufsize=1, cwd=None):
    # return a dummy process that writes a Bot started line to per-API bot_run.log
    try:
        from src.utils import get_data_file
        logp = get_data_file('bot_run.log')
    except Exception:
        logp = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../bot_run.log'))
    return DummyProc(logp)

def main():
    # Monkeypatch subprocess.Popen in the app module to avoid launching real bot
    import subprocess
    subprocess_Popen_orig = subprocess.Popen
    subprocess.Popen = fake_popen

    client = flask_app_mod.app.test_client()

    # Login first
    auth_file = os.path.join(os.path.dirname(__file__), 'user_auth.txt')
    email = 'admin@example.com'; password = 'password'
    if os.path.exists(auth_file):
        with open(auth_file, 'r', encoding='utf-8') as f:
            last = [l.strip() for l in f if l.strip()][-1]
            parts = last.split('|')
            if len(parts)>=2:
                email, password = parts[0], parts[1]
    print('Logging in', email)
    r = client.post('/api/login', json={'email': email, 'password': password})
    print('login:', r.status_code, r.get_json())

    # Call start
    print('Calling /api/start')
    r2 = client.post('/api/start', json={'netmode': 'testnet'})
    print('/api/start ->', r2.status_code, r2.get_json())

    # Call reconcile (real if keys available in .env)
    print('Calling /api/reconcile')
    r3 = client.post('/api/reconcile', json={'timeout': 20})
    print('/api/reconcile ->', r3.status_code, r3.get_json())

    # Call stop
    print('Calling /api/stop')
    r4 = client.post('/api/stop')
    print('/api/stop ->', r4.status_code, r4.get_json())

    # restore
    subprocess.Popen = subprocess_Popen_orig

if __name__ == '__main__':
    main()
