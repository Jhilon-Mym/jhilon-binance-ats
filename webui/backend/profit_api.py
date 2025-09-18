import os
import json
from flask import Blueprint, send_file, jsonify
from src.profit_tracker import ProfitTracker

profit_api = Blueprint('profit_api', __name__)

# Use the same global instance as in utils.py
from src.utils import profit_tracker

@profit_api.route('/api/profit_report', methods=['GET'])
def download_report():
    filename = 'profit_report.csv'
    profit_tracker.report_csv(filename)
    if os.path.exists(filename):
        return send_file(filename, as_attachment=True)
    return jsonify({'ok': False, 'msg': 'Report not found'}), 404

@profit_api.route('/api/profit_equity', methods=['GET'])
def get_equity_curve():
    curve = profit_tracker.get_report().get('equity_curve', [])
    return jsonify({'equity_curve': curve})
