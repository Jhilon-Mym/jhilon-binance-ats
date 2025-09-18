"""Quick reporter for model_debug.jsonl
Produces a short summary: total entries, apply_strategy counts, acceptance rate,
weighted decision stats (avg w_ai, tuned_thresh), override counts, and agreement rates.
"""
import json
from pathlib import Path
from statistics import mean, median

p = Path(__file__).resolve().parents[1] / 'model_debug.jsonl'
if not p.exists():
    print('model_debug.jsonl not found at', p)
    raise SystemExit(1)

lines = p.read_text(encoding='utf-8').strip().splitlines()
if not lines:
    print('no entries')
    raise SystemExit(0)

total = 0
apply_count = 0
accepted = 0
rejected = 0
override_events = 0
weighted_count = 0
w_ai_vals = []
tuned_thresh_vals = []
combined_scores = []
ai_win_probs = []
ai_vs_final_agree = 0
ai_vs_final_total = 0

weighted_samples = []

for L in lines:
    try:
        rec = json.loads(L)
    except Exception:
        continue
    total += 1
    # detect override events
    if rec.get('event') == 'ai_override' or rec.get('ai_override'):
        override_events += 1
    # apply_strategy style (from debug_signal)
    if 'apply_strategy' in rec:
        apply_count += 1
        app = rec['apply_strategy']
        if app.get('side') is not None:
            accepted += 1
        else:
            rejected += 1
        ai = rec.get('ai_signal') or rec.get('ai')
        if ai:
            ai_win_probs.append(float(ai.get('win_prob', 0.0)))
            if app.get('side') is not None:
                ai_vs_final_total += 1
                if str(ai.get('side')) == str(app.get('side')):
                    ai_vs_final_agree += 1
    # weighted decisions / tuning logs
    if rec.get('event') == 'weighted_decision' or rec.get('weighted'):
        weighted_count += 1
        w = rec.get('weighted') or {}
        w_ai = w.get('w_ai') if isinstance(w, dict) else None
        tuned = w.get('tuned_thresh') if isinstance(w, dict) else None
        comb = w.get('combined_score') if isinstance(w, dict) else None
        if w_ai is not None:
            try:
                w_ai_vals.append(float(w_ai))
            except Exception:
                pass
        if tuned is not None:
            try:
                tuned_thresh_vals.append(float(tuned))
            except Exception:
                pass
        if comb is not None:
            try:
                combined_scores.append(float(comb))
            except Exception:
                pass
        weighted_samples.append(rec)

# safe aggregates
print('model_debug.jsonl summary')
print('------------------------')
print(f'total lines parsed: {total}')
print(f'apply_strategy entries: {apply_count}')
print(f'  accepted: {accepted}')
print(f'  rejected: {rejected}')
print(f'acceptance rate: {accepted / apply_count:.2%}' if apply_count else 'acceptance rate: N/A')
print(f'override events logged: {override_events}')

print()
print(f'weighted decisions: {weighted_count}')
if weighted_count:
    print(f'  avg w_ai: {mean(w_ai_vals):.3f}  (n={len(w_ai_vals)})' if w_ai_vals else '  avg w_ai: N/A')
    print(f'  median tuned_thresh: {median(tuned_thresh_vals):.3f}  (n={len(tuned_thresh_vals)})' if tuned_thresh_vals else '  median tuned_thresh: N/A')
    print(f'  avg combined_score: {mean(combined_scores):.3f}' if combined_scores else '  avg combined_score: N/A')

print()
if ai_win_probs:
    print(f'AI win_prob  mean: {mean(ai_win_probs):.3f}  median: {median(ai_win_probs):.3f}  n={len(ai_win_probs)}')
if ai_vs_final_total:
    print(f'AI vs final agreement: {ai_vs_final_agree}/{ai_vs_final_total} ({ai_vs_final_agree/ai_vs_final_total:.2%})')

# show last few weighted samples
if weighted_samples:
    print('\nlast weighted decisions (most recent first):')
    for rec in weighted_samples[-5:]:
        ts = rec.get('timestamp')
        w = rec.get('weighted') or {}
        print(f'  {ts}  w_ai={w.get("w_ai")}  combined_score={w.get("combined_score")}  tuned={w.get("tuned_thresh")}  ind_major={rec.get("ind_major")} final={rec.get("final_side")}')

print('\nDone.')
