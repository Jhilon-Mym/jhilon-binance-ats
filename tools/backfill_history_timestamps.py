"""Backfill trade history timestamps.

This script safely updates trade history JSON files in the repo root:
- trade_history_mainnet.json
- trade_history_testnet.json
- trade_history.json (legacy)

For each trade entry where 'timestamp' is missing or 'N/A', it will:
- prefer 'closed_at' if present and a string
- otherwise set to current time (ISO, seconds)

It creates a timestamped backup next to each file before overwriting.

Run this from repo root (the script uses absolute paths matching get_history_file()).
"""
import json
import os
import datetime

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
        ts = t.get('timestamp')
        if not ts or ts == 'N/A':
            closed_at = t.get('closed_at')
            if closed_at and isinstance(closed_at, str) and closed_at != 'N/A':
                t['timestamp'] = closed_at
            else:
                t['timestamp'] = datetime.datetime.now().isoformat(timespec='seconds')
            updated += 1
    if updated > 0:
        # backup
        bak = fp + '.bak.' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        try:
            os.replace(fp, bak)
            print(f"[BACKUP] {fp} -> {bak}")
        except Exception as e:
            print(f"[WARN] Could not create backup for {fp}: {e}; will attempt write with .new file")
            bak = fp + '.new'
        # write updated file
        try:
            with open(fp, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"[UPDATED] {fp}: {updated} entries updated")
            updated_summary[fp] = updated
        except Exception as e:
            print(f"[ERROR] Failed to write {fp}: {e}")
            # attempt to restore backup
            try:
                os.replace(bak, fp)
                print(f"[RESTORE] Restored {fp} from backup")
            except Exception:
                print(f"[CRITICAL] Could not restore backup for {fp}")

if not updated_summary:
    print("No files updated. No missing timestamps found or no history files present.")
else:
    print("Summary:")
    for k, v in updated_summary.items():
        print(f"  {k}: {v} entries updated")
