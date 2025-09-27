import os
import sys
import json
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from src.utils import get_pending_file, get_profit_tracker
except Exception as e:
    print('ERROR importing src.utils:', e)
    raise

pf = get_pending_file()
print('PENDING_FILE ->', pf)
print('exists:', os.path.exists(pf))
if os.path.exists(pf):
    try:
        with open(pf, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict) and 'open_trades' in data:
            pending_list = data['open_trades']
        elif isinstance(data, list):
            pending_list = data
        else:
            pending_list = []
        print('Pending file entries:', len(pending_list))
        print(json.dumps(pending_list[:10], default=str, indent=2))
    except Exception as e:
        print('ERROR reading pending file:', e)

try:
    tracker = get_profit_tracker()
    print('\nTracker pending length:', len(tracker.pending))
    try:
        print('Tracker cached_balances:', tracker.cached_balances)
    except Exception:
        print('Tracker has no cached_balances attribute')
    try:
        sample = tracker.pending[:10]
        print('Tracker pending sample:')
        print(json.dumps(sample, default=str, indent=2))
    except Exception as e:
        print('Error dumping tracker.pending:', e)
except Exception as e:
    print('ERROR getting tracker:', e)
    raise
