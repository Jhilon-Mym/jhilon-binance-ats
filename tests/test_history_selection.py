import os
import json
import shutil
import importlib

def test_get_best_history_prefers_repo(tmp_path):
    # Import inside the test so the autouse tmp_path chdir has already been applied
    import webui.backend.app as app_mod
    from src.utils import get_history_file

    # determine repository root used by the app
    PROJECT_ROOT = getattr(app_mod, 'PROJECT_ROOT')
    assert os.path.isdir(PROJECT_ROOT)

    # repo candidate path (testnet)
    repo_path = os.path.join(PROJECT_ROOT, 'trade_history_testnet.json')

    # per-API candidate as returned by get_history_file()
    api_candidate = get_history_file()

    # Ensure we have writable locations
    # Create both files so the selection logic must choose repo_path
    try:
        with open(repo_path, 'w', encoding='utf-8') as f:
            json.dump([{'status': 'win', 'realized_pnl_usdt': 1}], f)
        # create api candidate if missing
        api_dir = os.path.dirname(api_candidate)
        os.makedirs(api_dir, exist_ok=True)
        with open(api_candidate, 'w', encoding='utf-8') as f:
            json.dump([{'status': 'loss', 'realized_pnl_usdt': -1}], f)

        # reload module to ensure fresh behavior (if needed)
        importlib.reload(app_mod)

        best = app_mod.get_best_history_file()
        assert best is not None
        # It should prefer the repo-level file when present
        assert os.path.abspath(best) == os.path.abspath(repo_path)
    finally:
        # cleanup
        try:
            if os.path.exists(repo_path):
                os.remove(repo_path)
        except Exception:
            pass
        try:
            if os.path.exists(api_candidate):
                os.remove(api_candidate)
        except Exception:
            pass