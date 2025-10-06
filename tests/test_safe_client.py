import pytest
import asyncio
import time
from src.safe_client import SafeClient

class DummyClient:
    def __init__(self):
        self.call_count = 0
        self.last_response_headers = {}

    def get_symbol_ticker(self, symbol=None):
        self.call_count += 1
        # emulate response headers attached to client
        self.last_http_response = type('R', (), {'headers': {'X-MBX-USED-WEIGHT-1M': str(self.call_count)}})()
        return {'symbol': symbol, 'price': '100'}

    def fail_once_then_succeed(self, symbol=None):
        self.call_count += 1
        if self.call_count == 1:
            raise RuntimeError('transient')
        return {'ok': True}


def test_sync_reservation_and_header_sync():
    c = SafeClient.from_client(DummyClient())
    # First call consumes weight 1
    r = c.get_symbol_ticker(symbol='BTCUSDT')
    assert r['symbol'] == 'BTCUSDT'
    # After call, header-based sync should have updated usage
    # current usage should be >= 1
    assert c._current_usage() >= 1


def test_retry_backoff_sync():
    dummy = DummyClient()
    c = SafeClient.from_client(dummy)
    # call the method that fails once then succeeds
    res = c.sync_request(dummy.fail_once_then_succeed)
    assert res.get('ok') is True


def test_async_call_and_header_sync():
    dummy = DummyClient()
    c = SafeClient.from_client(dummy)
    # Run the async call via asyncio.run so pytest-asyncio is not required
    res = asyncio.run(c.async_call('get_symbol_ticker', symbol='BTCUSDT'))
    assert res['symbol'] == 'BTCUSDT'
    # ensure usage updated from dummy header
    assert c._current_usage() >= 1
