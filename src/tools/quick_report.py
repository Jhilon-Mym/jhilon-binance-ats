import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json, statistics, pathlib
from dotenv import load_dotenv
load_dotenv()
from src.config import Config
p = pathlib.Path(__file__).with_name('..').resolve().joinpath('model_debug.jsonl')
lines = [l.strip() for l in open(p, 'r', encoding='utf-8').read().splitlines() if l.strip()]
entries = [json.loads(l) for l in lines]
# classify
total = len(entries)
weighted = [e for e in entries if e.get('event') == 'weighted_decision']
apply = [e for e in entries if e.get('apply_strategy') or e.get('event')=='weighted_decision' and 'final_side' in e]
# normalize apply entries: some are apply_strategy dicts or weighted_decision events later
apply_entries = []
for e in entries:
    if 'apply_strategy' in e:
        rec = e['apply_strategy']
        ai = e.get('ai_signal')
        lr = e.get('last_row')
        apply_entries.append({'side': rec.get('side'), 'reason': rec.get('reason'), 'win_prob': float(rec.get('win_prob',0.0)), 'combined_score': rec.get('combined_score'), 'w_ai': rec.get('w_ai'), 'ai_side': ai.get('side') if ai else None})
    elif e.get('event')=='weighted_decision':
        wd = e
        apply_entries.append({'side': wd.get('final_side'), 'reason': 'weighted_decision', 'win_prob': wd.get('ai_signal',{}).get('win_prob',0.0), 'combined_score': wd.get('weighted',{}).get('combined_score'), 'w_ai': wd.get('weighted',{}).get('w_ai'), 'ai_side': wd.get('ai_signal',{}).get('side')})

accepted = [a for a in apply_entries if a['side'] is not None]
rejected = [a for a in apply_entries if a['side'] is None]

# counts
c_accepted = len(accepted)
c_rejected = len(rejected)

# reasons
from collections import Counter
reason_counts = Counter([a['reason'] for a in apply_entries if a.get('reason')])

# combined_score stats
scores = [float(a['combined_score']) for a in apply_entries if a.get('combined_score') is not None]
wai = [float(a['w_ai']) for a in apply_entries if a.get('w_ai') is not None]
winps = [float(a['win_prob']) for a in apply_entries if a.get('win_prob') is not None]
conflicts = [1 for a in apply_entries if a.get('ai_side') and a.get('side') and a.get('ai_side')!=a.get('side')]

print('TOTAL_ENTRIES', total)
print('APPLY_ENTRIES', len(apply_entries))
print('ACCEPTED', c_accepted, 'REJECTED', c_rejected)
print('REASON_COUNTS', reason_counts)
if scores:
    print('COMBINED_SCORE mean/median/min/max', statistics.mean(scores), statistics.median(scores), min(scores), max(scores))
else:
    print('COMBINED_SCORE none')
if wai:
    print('W_AI mean/median', statistics.mean(wai), statistics.median(wai))
if winps:
    print('WIN_PROB mean/median/min/max', statistics.mean(winps), statistics.median(winps), min(winps), max(winps))
print('CONFLICTS (ai_side != final side)', sum(conflicts))

# show top recent entries
print('\nRECENT APPLY ENTRIES (last 8):')
for a in apply_entries[-8:]:
    print(a)
