import os
from src.config import Config
from src.utils import compute_recent_win_rate, compute_recent_volatility, get_history_file


def test_adaptive_toggle_env():
    os.environ['ADAPTIVE_ENABLED'] = 'true'
    assert getattr(Config, 'ADAPTIVE_ENABLED') is True
    os.environ['ADAPTIVE_ENABLED'] = 'false'
    assert getattr(Config, 'ADAPTIVE_ENABLED') is False


def test_compute_recent_helpers_no_history(tmp_path, monkeypatch):
    # ensure history file points to a temp location with no data
    hf = tmp_path / 'trade_history_testnet.json'
    # ensure file is created empty
    hf.write_text('[]')
    # monkeypatch get_history_file to return this path
    monkeypatch.setenv('USE_TESTNET', 'true')
    # patch get_history_file by monkeypatching src.utils.get_history_file via import
    import importlib
    import src.utils as u
    monkeypatch.setattr(u, 'get_history_file', lambda: str(hf))
    assert compute_recent_win_rate(n=10) is None
    assert compute_recent_volatility(n=10) is None
