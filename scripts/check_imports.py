import importlib, sys, os

# Ensure repo root is on sys.path so 'training' and 'src' packages import correctly
repo_root = os.path.dirname(os.path.dirname(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

mods = ['training.train_hybrid', 'src.predict_hybrid']
ok = True
for m in mods:
    try:
        importlib.import_module(m)
        print('OK', m)
    except Exception as e:
        ok = False
        print('ERR', m, type(e).__name__, e)

sys.exit(0 if ok else 2)
