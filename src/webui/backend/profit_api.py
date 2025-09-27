import os
import json
from flask import Blueprint, send_file, jsonify
import traceback
import tempfile
import pathlib
import time
from src.profit_tracker import ProfitTracker

profit_api = Blueprint('profit_api', __name__)

# Use the same global instance as in utils.py
from src.utils import get_profit_tracker, get_netmode

@profit_api.route('/api/profit_report', methods=['GET'])
def download_report():
    tracker = get_profit_tracker()
    try:
        # Determine netmode (testnet/mainnet) and include it in the filename so UI users
        # downloading reports can keep files separate per environment.
        use_testnet = get_netmode()
        net = 'testnet' if use_testnet else 'mainnet'
        # Derive symbol for filename: prefer Config.SYMBOL (stable), otherwise fall back
        # to the embedded ProfitTracker symbol or a sensible default.
        # Prefer a stable symbol from Config to avoid reading attributes from
        # the running tracker instance which may not expose the attribute.
        try:
            from src.config import Config
            symbol = getattr(Config, 'SYMBOL', 'BTCUSDT')
        except Exception:
            symbol = 'BTCUSDT'
        # Create a temp file path in the system temp dir to avoid CWD issues
        tmp_dir = pathlib.Path(tempfile.gettempdir())
        tmp_path = tmp_dir / f"profit_report_{net}_{symbol}.csv"
        tracker.report_csv(str(tmp_path))
        if tmp_path.exists():
            return send_file(str(tmp_path), as_attachment=True)
        return jsonify({'ok': False, 'msg': 'Report generation failed'}), 500
    except Exception as e:
        # Write full traceback to a small log file for debugging
        try:
            logp = pathlib.Path(tempfile.gettempdir()) / 'profit_api_error.log'
            with open(logp, 'a', encoding='utf-8') as lf:
                lf.write('\n---\n')
                lf.write(f'Time: {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
                lf.write(traceback.format_exc())
                lf.write('\n')
        except Exception:
            pass
        return jsonify({'ok': False, 'msg': f'Error generating report: {e}'}), 500

@profit_api.route('/api/profit_equity', methods=['GET'])
def get_equity_curve():
    tracker = get_profit_tracker()
    curve = tracker.get_report().get('equity_curve', [])
    return jsonify({'equity_curve': curve})
