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
    try:
        # Generate the profit CSV from the persisted trade history JSON so the web UI
        # (which runs in a separate process from the bot) can always produce a
        # meaningful report even if its in-memory ProfitTracker is empty.
        use_testnet = get_netmode()
        net = 'testnet' if use_testnet else 'mainnet'
        try:
            from src.config import Config
            symbol = getattr(Config, 'SYMBOL', 'BTCUSDT')
        except Exception:
            symbol = 'BTCUSDT'
        # Load history file
        from src.utils import get_history_file
        history_file = get_history_file()
        if not os.path.exists(history_file):
            return jsonify({'ok': False, 'msg': 'No trade history found'}), 404
        with open(history_file, 'r', encoding='utf-8') as f:
            trades = json.load(f)
        # Build CSV in temp dir
        tmp_dir = pathlib.Path(tempfile.gettempdir())
        tmp_path = tmp_dir / f"profit_report_{net}_{symbol}.csv"
        # Write CSV: include timestamp, side, qty, entry, close_price, fee (if any), realized_pnl_usdt
        import csv
        realized_sum = 0.0
        total_fees = 0.0
        with open(tmp_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'orderId', 'side', 'qty', 'entry', 'close_price', 'fee', 'realized_pnl_usdt'])
            for t in trades:
                try:
                    ts = t.get('timestamp') or t.get('closed_at') or ''
                    oid = t.get('orderId', '')
                    side = t.get('side', '')
                    qty = t.get('qty', '')
                    entry = t.get('entry', '')
                    close_price = t.get('close_price', '')
                    fee = t.get('fee', '')
                    rp = float(t.get('realized_pnl_usdt', 0) or 0)
                except Exception:
                    ts = ''
                    oid = ''
                    side = ''
                    qty = ''
                    entry = ''
                    close_price = ''
                    fee = ''
                    rp = 0.0
                writer.writerow([ts, oid, side, qty, entry, close_price, fee, rp])
                try:
                    realized_sum += float(rp)
                except Exception:
                    pass
                try:
                    total_fees += float(t.get('fee', 0) or 0)
                except Exception:
                    pass
            writer.writerow([])
            writer.writerow(['realized_pnl', realized_sum])
            writer.writerow(['unrealized_pnl', 0])
            writer.writerow(['total_fees', total_fees])
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
