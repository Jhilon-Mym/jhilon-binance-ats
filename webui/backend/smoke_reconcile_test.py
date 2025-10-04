"""Smoke test for reconcile endpoint.

Usage: py -3 webui\backend\smoke_reconcile_test.py

If BINANCE_API_KEY and BINANCE_API_SECRET are set in environment or .env, this
script will attempt a real reconcile (with a short timeout) via the app test
client using existing session-based login. Otherwise, it will skip the real
call and inform the user.
"""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if ROOT not in sys.path: sys.path.insert(0, ROOT)
from webui.backend import app as flask_app_mod

def main():
    client = flask_app_mod.app.test_client()

    # Read credentials
    auth_file = os.path.join(os.path.dirname(__file__), 'user_auth.txt')
    email = 'admin@example.com'; password = 'password'
    if os.path.exists(auth_file):
        with open(auth_file, 'r', encoding='utf-8') as f:
            last = [l.strip() for l in f if l.strip()][-1]
            parts = last.split('|')
            if len(parts)>=2:
                email, password = parts[0], parts[1]

    print('Logging in as', email)
    r = client.post('/api/login', json={'email': email, 'password': password})
    print('login:', r.status_code, r.get_json())

    # If no API keys available, do not perform a real reconcile
    if not (os.getenv('BINANCE_API_KEY') and os.getenv('BINANCE_API_SECRET')):
        print('BINANCE_API_KEY/SECRET not set — skipping real reconcile. Use RECONCILE_TOKEN or set API keys to test real flow.')
        return

    print('Calling real reconcile (short timeout)...')
    r2 = client.post('/api/reconcile', json={'timeout': 10})
    print('reconcile:', r2.status_code, r2.get_json())

if __name__ == '__main__': main()
