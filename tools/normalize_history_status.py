"""Normalize legacy history status values.

This script updates any trade history entries with "status": "closed" to more
useful values:
  - If realized_pnl_usdt > 0 -> 'win'
  - If realized_pnl_usdt < 0 -> 'loss'
  - If realized_pnl_usdt == 0 -> keep as 'closed'

It operates on the same history files as the backfill script and creates a
timestamped backup before writing.
"""
import json, os, datetime
ROOT = r"D:\binance_ats_clone\obaidur-binance-ats-main"
FILES = [
    os.path.join(ROOT, 'trade_history_mainnet.json'),
    os.path.join(ROOT, 'trade_history_testnet.json'),
    os.path.join(ROOT, 'trade_history.json'),
]
updated_summary = {}
for fp in FILES:
    if not os.path.exists(fp):
        continue
    try:
        with open(fp, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[SKIP] Could not read {fp}: {e}")
        continue
    if not isinstance(data, list):
        print(f"[SKIP] {fp} does not contain a list of trades; skipping")
        continue
    updated = 0
    for t in data:
        if not isinstance(t, dict):
            continue
        st = t.get('status')
        if st == 'closed':
            try:
                pnl = float(t.get('realized_pnl_usdt', 0) or 0)
            except Exception:
                pnl = 0
            if pnl > 0:
                t['status'] = 'win'
                updated += 1
            elif pnl < 0:
                t['status'] = 'loss'
                updated += 1
            else:
                # keep as 'closed' when zero PnL
                pass
    if updated > 0:
        bak = fp + '.bak.' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        try:
            os.replace(fp, bak)
            print(f"[BACKUP] {fp} -> {bak}")
        except Exception as e:
            print(f"[WARN] Could not create backup for {fp}: {e}; will attempt write with .new file")
            bak = fp + '.new'
        try:
            with open(fp, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"[UPDATED] {fp}: {updated} statuses updated")
            updated_summary[fp] = updated
        except Exception as e:
            print(f"[ERROR] Failed to write {fp}: {e}")
            try:
                os.replace(bak, fp)
                print(f"[RESTORE] Restored {fp} from backup")
            except Exception:
                print(f"[CRITICAL] Could not restore backup for {fp}")

if not updated_summary:
    print("No statuses required normalization.")
else:
    print("Summary:")
    for k, v in updated_summary.items():
        print(f"  {k}: {v} statuses updated")
