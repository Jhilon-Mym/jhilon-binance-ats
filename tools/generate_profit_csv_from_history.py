"""Quick helper to generate profit CSV from persisted history (same logic as profit_api).
Robust to being run from any working directory inside the repo.
"""
import json, os, pathlib, tempfile, csv, sys

# Ensure repo root and src/ are on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# Also make sure src is importable
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

try:
    from src.utils import get_history_file
except Exception as e:
    print('Failed to import src.utils; sys.path is:', sys.path[:5])
    raise

history_file = get_history_file()
print('Using history file:', history_file)
if not os.path.exists(history_file):
    print('No history file found')
    raise SystemExit(1)
with open(history_file, 'r', encoding='utf-8') as f:
    trades = json.load(f)
net = 'testnet' if 'testnet' in history_file else 'mainnet'
symbol = 'BTCUSDT'
# write tmp csv
tmp_path = pathlib.Path(tempfile.gettempdir()) / f"profit_report_{net}_{symbol}.csv"
realized_sum = 0.0
total_fees = 0.0
with open(tmp_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['timestamp', 'orderId', 'side', 'qty', 'entry', 'close_price', 'fee', 'realized_pnl_usdt'])
    for t in trades:
        ts = t.get('timestamp') or t.get('closed_at') or ''
        oid = t.get('orderId', '')
        side = t.get('side', '')
        qty = t.get('qty', '')
        entry = t.get('entry', '')
        close_price = t.get('close_price', '')
        fee = t.get('fee', '')
        try:
            rp = float(t.get('realized_pnl_usdt', 0) or 0)
        except Exception:
            rp = 0.0
        writer.writerow([ts, oid, side, qty, entry, close_price, fee, rp])
        realized_sum += rp
        try:
            total_fees += float(t.get('fee', 0) or 0)
        except Exception:
            pass
    writer.writerow([])
    writer.writerow(['realized_pnl', realized_sum])
    writer.writerow(['unrealized_pnl', 0])
    writer.writerow(['total_fees', total_fees])
print('Wrote', tmp_path)
