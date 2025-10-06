import os
import json
import tempfile
from decimal import Decimal

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
import sys
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils import TradeStats


@pytest.fixture(autouse=True)
def chdir_tmp(monkeypatch, tmp_path):
    # Run tests in a temporary directory so file IO doesn't affect repo files
    monkeypatch.chdir(tmp_path)
    yield


def test_open_trade_deduplication():
    ts = TradeStats()
    # ensure clean
    from src.utils import get_pending_file
    pfile = get_pending_file()
    if os.path.exists(pfile):
        os.remove(pfile)
    sig = {"side": "BUY", "sl": 100.0, "tp": 200.0}
    ts.open_trade(sig, 100.0, Decimal('0.001'), 10.0)
    # duplicate (same side & entry & amount)
    ts.open_trade(sig, 100.0, Decimal('0.001'), 10.0)
    pending = ts.load_pending()
    assert len(pending) == 1


def test_update_pending_closing_path():
    ts = TradeStats()
    from src.utils import get_pending_file, get_history_file
    pfile = get_pending_file()
    hfile = get_history_file()
    if os.path.exists(pfile):
        os.remove(pfile)
    if os.path.exists(hfile):
        os.remove(hfile)
    # Guarantee no in-memory pending leftover from other tests
    try:
        ts.pending = []
    except Exception:
        pass
    sig = {"side": "BUY", "sl": 90.0, "tp": 120.0}
    ts.open_trade(sig, 100.0, Decimal('0.001'), 10.0)
    # Simulate a candle that hits TP
    candle = {"high": 121.0, "low": 99.0}
    closed = ts.update_pending(candle)
    assert len(closed) == 1
    # history file should contain the closed trade
    hfile = get_history_file()
    with open(hfile, 'r') as f:
        history = json.load(f)
    assert len(history) == 1