import os
import time
from flask import Blueprint, request, jsonify

AUTH_FILE = os.path.join(os.path.dirname(__file__), 'user_auth.txt')
auth_api = Blueprint('auth_api', __name__)

def read_latest_user():
    if not os.path.exists(AUTH_FILE):
        return None, None, None
    with open(AUTH_FILE, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
        if not lines:
            return None, None, None
        last = lines[-1]
        parts = last.split('|')
        if len(parts) == 3:
            return parts[0], parts[1], parts[2]
    return None, None, None

def save_user(email, password):
    ts = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    with open(AUTH_FILE, 'a', encoding='utf-8') as f:
        f.write(f'{email}|{password}|{ts}\n')

@auth_api.route('/api/login', methods=['POST'])
def login():
    data = request.get_json(force=True, silent=True) or {}
    email = data.get('email')
    password = data.get('password')
    saved_email, saved_password, _ = read_latest_user()
    if email == saved_email and password == saved_password:
        return jsonify({'success': True, 'message': 'Login successful'})
    return jsonify({'success': False, 'message': 'Invalid email or password'}), 401

@auth_api.route('/api/change_password', methods=['POST'])
def change_password():
    data = request.get_json(force=True, silent=True) or {}
    email = data.get('email')
    old_password = data.get('old_password')
    new_password = data.get('new_password')
    saved_email, saved_password, _ = read_latest_user()
    if email == saved_email and old_password == saved_password:
        save_user(email, new_password)
        return jsonify({'success': True, 'message': 'Password updated'})
    return jsonify({'success': False, 'message': 'Invalid credentials'}), 401
