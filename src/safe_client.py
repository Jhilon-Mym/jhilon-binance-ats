import time
import threading
import logging
import random
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional, Any, Union

logger = logging.getLogger('safe_client')


class SafeClient:
    """A professional, rate-limit aware wrapper for the Binance Client.

    Features:
    - Unified per-minute weight tracking (default 1200/min) shared across sync
      and async callers.
    - Auto-sleep when the 1-minute window would be exceeded.
    - Exponential backoff with jitter on transient errors.
    - Detailed logging for reservations, successes and failures.
    - Async-friendly: use `async_request` or `async_call` to schedule calls
      concurrently while the wrapper governs rate-limiting.
    - Easy testnet/mainnet toggle (passed to underlying Client).

    Usage (sync):
        client = SafeClient(api_key, api_secret, testnet=True)
        res = client.get_symbol_ticker(symbol='BTCUSDT')

    Usage (async):
        client = SafeClient(...)
        res = await client.async_request('get_symbol_ticker', symbol='BTCUSDT')

    Notes:
    - The wrapper runs sync Binance calls in a threadpool for async usage.
    - You can pass an existing `binance.client.Client` instance with the
      `raw_client` kwarg.
    """

    DEFAULT_WEIGHT_LIMIT = 1200
    DEFAULT_MAX_RETRIES = 5
    DEFAULT_RETRY_BACKOFF_BASE = 0.5
    DEFAULT_MAX_WORKERS = 8

    # best-effort weights for commonly used client methods (extendable)
    DEFAULT_WEIGHTS = {
        'get_klines': 1,
        'get_avg_price': 1,
        'get_symbol_ticker': 1,
        'get_order': 1,
        'get_all_orders': 40,
        'create_order': 1,
        'get_account': 5,
        'get_asset_balance': 1,
        'get_symbol_info': 1,
    }

    def __init__(self,
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 testnet: bool = True,
                 *,
                 raw_client: Any = None,
                 weight_limit: Optional[int] = None,
                 max_retries: Optional[int] = None,
                 backoff_base: Optional[float] = None,
                 executor: Optional[ThreadPoolExecutor] = None,
                 weights: Optional[dict] = None,
                 max_workers: Optional[int] = None):
        # lazy import so module import stays fast for tools that don't need binance
        if raw_client is not None:
            self._client = raw_client
        else:
            from binance.client import Client
            self._client = Client(api_key, api_secret, testnet=testnet)

        self.API_KEY = getattr(self._client, 'API_KEY', api_key)
        self.API_SECRET = getattr(self._client, 'API_SECRET', api_secret)
        self._testnet = testnet
        self._weight_limit = weight_limit or self.DEFAULT_WEIGHT_LIMIT
        self._max_retries = max_retries or self.DEFAULT_MAX_RETRIES
        self._backoff_base = backoff_base or self.DEFAULT_RETRY_BACKOFF_BASE
        self._weights = dict(self.DEFAULT_WEIGHTS)
        if isinstance(weights, dict):
            self._weights.update(weights)

        # store (timestamp, weight) tuples for last-minute window
        self._usage = []  # list of (ts_monotonic, weight)
        self._lock = threading.Lock()

        # executor used for async wrappers; can supply your own
        self._executor = executor or ThreadPoolExecutor(max_workers=(max_workers or self.DEFAULT_MAX_WORKERS))

    # ---------------------- weight bookkeeping ----------------------
    def _prune_usage(self) -> None:
        cutoff = time.monotonic() - 60.0
        with self._lock:
            self._usage = [t for t in self._usage if t[0] >= cutoff]

    def _current_usage(self) -> int:
        self._prune_usage()
        with self._lock:
            return sum(w for _, w in self._usage)

    def _resolve_weight(self, name: str, override: Optional[int] = None) -> int:
        if override is not None:
            return int(override)
        return int(self._weights.get(name, 1))

    def _next_safe_delay(self, weight: int) -> float:
        """Return seconds to wait until enough weight ages out so `weight` can be reserved."""
        self._prune_usage()
        with self._lock:
            cur = sum(w for _, w in self._usage)
            if cur + weight <= self._weight_limit:
                return 0.0
            if not self._usage:
                return 1.0
            oldest_ts = min(t for t, _ in self._usage)
            return max(0.1, 60.0 - (time.monotonic() - oldest_ts))

    def _reserve_now(self, weight: int) -> None:
        with self._lock:
            self._usage.append((time.monotonic(), weight))

    def _reserve_or_wait_sync(self, weight: int) -> None:
        while True:
            delay = self._next_safe_delay(weight)
            if delay <= 0:
                self._reserve_now(weight)
                logger.debug("Reserved weight %s (current %s/%s)", weight, self._current_usage(), self._weight_limit)
                return
            logger.warning("API weight approaching (%s/%s). Sleeping %.2fs before next attempt.", self._current_usage(), self._weight_limit, delay)
            time.sleep(delay)

    async def _reserve_or_wait_async(self, weight: int) -> None:
        # non-blocking version for async callers
        while True:
            delay = self._next_safe_delay(weight)
            if delay <= 0:
                self._reserve_now(weight)
                logger.debug("(async) Reserved weight %s (current %s/%s)", weight, self._current_usage(), self._weight_limit)
                return
            logger.warning("(async) API weight approaching (%s/%s). Sleeping %.2fs before next attempt.", self._current_usage(), self._weight_limit, delay)
            await asyncio.sleep(delay)

    # ---------------------- retry/backoff ----------------------
    def _call_with_retries_sync(self, func: Callable, weight: int, *args, **kwargs):
        attempt = 0
        while True:
            attempt += 1
            try:
                result = func(*args, **kwargs)
                logger.info("API call %s used weight=%s (attempt=%s)", getattr(func, '__name__', str(func)), weight, attempt)
                return result
            except Exception as e:
                if attempt > self._max_retries:
                    logger.exception("API call %s failed after %s attempts", getattr(func, '__name__', str(func)), attempt - 1)
                    raise
                backoff = self._backoff_base * (2 ** (attempt - 1))
                backoff = backoff * (0.8 + random.random() * 0.4)
                logger.warning("API call %s failed (attempt %s). Retrying in %.2fs: %s", getattr(func, '__name__', str(func)), attempt, backoff, e)
                time.sleep(backoff)

    async def _call_with_retries_async(self, func: Callable, weight: int, *args, **kwargs):
        # runs the sync func in an executor but performs async sleeps for backoff
        attempt = 0
        loop = asyncio.get_running_loop()
        while True:
            attempt += 1
            try:
                result = await loop.run_in_executor(self._executor, lambda: func(*args, **kwargs))
                logger.info("(async) API call %s used weight=%s (attempt=%s)", getattr(func, '__name__', str(func)), weight, attempt)
                return result
            except Exception as e:
                if attempt > self._max_retries:
                    logger.exception("(async) API call %s failed after %s attempts", getattr(func, '__name__', str(func)), attempt - 1)
                    raise
                backoff = self._backoff_base * (2 ** (attempt - 1))
                backoff = backoff * (0.8 + random.random() * 0.4)
                logger.warning("(async) API call %s failed (attempt %s). Retrying in %.2fs: %s", getattr(func, '__name__', str(func)), attempt, backoff, e)
                await asyncio.sleep(backoff)
            finally:
                # After each attempt (success or failure), try to sync usage from response headers
                try:
                    headers = self._extract_headers_from_owner(func)
                    if headers:
                        self._update_usage_from_header_headers(headers)
                except Exception:
                    pass

    # ---------------------- sync / async interfaces ----------------------
    def __getattr__(self, item: str):
        # forward attribute access to underlying client, wrap callables for
        # sync usage so they go through reservation + retry logic
        target = getattr(self._client, item)
        if callable(target):
            def wrapper(*args, **kwargs):
                override = None
                if '_weight' in kwargs:
                    override = kwargs.pop('_weight')
                weight = self._resolve_weight(item, override)
                self._reserve_or_wait_sync(weight)
                result = self._call_with_retries_sync(target, weight, *args, **kwargs)
                # best-effort: try to extract headers from client after the call
                try:
                    headers = self._extract_headers_from_owner(target)
                    if headers:
                        self._update_usage_from_header_headers(headers)
                except Exception:
                    pass
                return result
            return wrapper
        return target

    def raw_client(self) -> Any:
        return self._client

    # Async-friendly request: provide the method name (string) or callable
    async def async_request(self, func: Union[str, Callable], *args, _weight: Optional[int] = None, **kwargs) -> Any:
        """Call `func` (or method named `func`) in a threadpool while respecting
        the shared per-minute weight limit. Returns the underlying call result.

        Example: await client.async_request('get_symbol_ticker', symbol='BTCUSDT')
        """
        if isinstance(func, str):
            target = getattr(self._client, func)
            name = func
        else:
            target = func
            name = getattr(func, '__name__', str(func))

        override = _weight
        weight = self._resolve_weight(name, override)
        await self._reserve_or_wait_async(weight)
        return await self._call_with_retries_async(target, weight, *args, **kwargs)

    async def async_call(self, method_name: str, *args, _weight: Optional[int] = None, **kwargs) -> Any:
        return await self.async_request(method_name, *args, _weight=_weight, **kwargs)

    def sync_request(self, func: Union[str, Callable], *args, _weight: Optional[int] = None, **kwargs) -> Any:
        """Explicit sync request wrapper (identical to calling methods directly)."""
        if isinstance(func, str):
            target = getattr(self._client, func)
            name = func
        else:
            target = func
            name = getattr(func, '__name__', str(func))
        weight = self._resolve_weight(name, _weight)
        self._reserve_or_wait_sync(weight)
        return self._call_with_retries_sync(target, weight, *args, **kwargs)

    # Convenience factory
    @classmethod
    def from_client(cls, client, **kwargs):
        return cls(raw_client=client, **kwargs)

    # ---------------------- header-based weight sync ----------------------
    def _extract_headers_from_owner(self, owner) -> Optional[dict]:
        """Try to extract last-response headers from the underlying client or
        method wrapper. Different binance client implementations may attach
        the HTTP response to the client instance or the method. We try a
        few likely attribute names and return a dict-like mapping or None.
        """
        try:
            # If the client stores last_response or last_response_headers
            cand = getattr(self._client, 'last_response', None) or getattr(self._client, 'last_response_headers', None)
            if cand and isinstance(cand, dict):
                return cand
        except Exception:
            pass
        try:
            # Some wrappers attach a `response` attribute to the method result
            cand = getattr(owner, 'response', None)
            if cand and hasattr(cand, 'headers'):
                return dict(cand.headers)
        except Exception:
            pass
        try:
            # Some clients store last_http_response on the client
            cand = getattr(self._client, 'last_http_response', None)
            if cand and hasattr(cand, 'headers'):
                return dict(cand.headers)
        except Exception:
            pass
        return None

    def _update_usage_from_header_headers(self, headers: dict) -> None:
        """Look for common Binance rate-limit headers and update local usage
        accounting if we can parse an 'used weight per minute' value.

        Common header names:
         - X-MBX-USED-WEIGHT-1M
         - x-mbx-used-weight-1m
        The header value is expected to be an integer representing used weight
        in the last minute. We'll compute delta = reported_used - current and,
        if positive, append a synthetic reservation entry with that delta so
        our local accounting matches the server.
        """
        if not headers:
            return
        keys = [k for k in headers.keys()]
        found = None
        for candidate in ('X-MBX-USED-WEIGHT-1M', 'x-mbx-used-weight-1m', 'X-MBX-USED-WEIGHT', 'x-mbx-used-weight'):
            for k in keys:
                if k.lower() == candidate.lower():
                    found = headers.get(k)
                    break
            if found is not None:
                break
        if found is None:
            return
        try:
            reported = int(found)
        except Exception:
            try:
                # Sometimes header may contain comma-separated values; take first
                reported = int(str(found).split(',')[0])
            except Exception:
                return
        # Compute delta between server-reported and our local sum
        local = self._current_usage()
        delta = reported - local
        if delta <= 0:
            # nothing to do
            return
        # Append a synthetic reservation with timestamp slightly in the past
        # so it ages out in the normal window. Use monotonic now.
        with self._lock:
            ts = time.monotonic()
            # distribute the delta as a single entry
            self._usage.append((ts, int(delta)))
        logger.debug("Synced local weight usage from headers: reported=%s local=%s delta=%s", reported, local, delta)

