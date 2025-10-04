"""Simple test harness that verifies login -> reconcile session flow.

Run with: py -3 webui\backend\test_reconcile_flow.py

This script imports the Flask app, uses the test client to POST to /api/login
and then /api/reconcile while preserving cookies. To avoid external network
dependencies it monkeypatches _run_reconcile_with_timeout to return a fake
result.
"""
import importlib, sys, os

# Ensure project root is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from webui.backend import app as flask_app_mod

def fake_reconcile(timeout=20, symbol=None):
    return {'updated': 2, 'removed': 1}

def main():
    print('Using Flask app from:', flask_app_mod.__file__)
    # Monkeypatch the reconcile helper
    flask_app_mod._run_reconcile_with_timeout = fake_reconcile

    client = flask_app_mod.app.test_client()

    # Read credentials from user_auth.txt if present
    auth_file = os.path.join(os.path.dirname(__file__), 'user_auth.txt')
    email = 'admin@example.com'
    password = 'password'
    if os.path.exists(auth_file):
        with open(auth_file, 'r', encoding='utf-8') as f:
            last = [l.strip() for l in f if l.strip()][-1]
            parts = last.split('|')
            if len(parts) >= 2:
                email, password = parts[0], parts[1]

    print('Logging in as', email)
    r = client.post('/api/login', json={'email': email, 'password': password})
    print('Login status:', r.status_code, r.get_json())
    if r.status_code != 200:
        print('Login failed; cannot proceed to reconcile.')
        return

    # Now call reconcile using the same client (session cookie preserved)
    r2 = client.post('/api/reconcile', json={'timeout':5})
    print('Reconcile status:', r2.status_code, r2.get_json())

if __name__ == '__main__':
    main()
