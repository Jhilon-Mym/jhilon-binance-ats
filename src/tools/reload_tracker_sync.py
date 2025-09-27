import os
import sys
import json

# ensure repo root is on sys.path so we can import src
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from src.utils import get_profit_tracker, get_pending_file
except Exception as e:
    print('ERROR importing src.utils:', e)
    raise


pf = get_pending_file()
print('Pending file:', pf, 'exists', os.path.exists(pf))
with open(pf, 'r', encoding='utf-8') as f:
    data = json.load(f)
print('Entries in file:', len(data))

tr = get_profit_tracker()
# reload pending into tracker if method exists
if hasattr(tr, 'load_pending_from_file'):
    tr.load_pending_from_file()
elif hasattr(tr, 'sync_pending'):
    tr.sync_pending()
else:
    # fallback: set tracker.pending to file contents
    tr.pending = data

print('Tracker.pending len after sync:', len(tr.pending))
print('Tracker.cached_balances:', getattr(tr, 'cached_balances', None))
print('Done')
