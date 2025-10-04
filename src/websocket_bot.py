
# websocket_bot.py — WS runner + REST preload (no CSV fallback)
import logging
import pandas as pd
import threading, time, os
import traceback
from typing import Callable, Optional
from binance import ThreadedWebsocketManager

logger = logging.getLogger(__name__)

class WSRunner:
    def __init__(self, client, symbol: str, interval: str, on_candle_closed: Callable[[dict], None]):
        # keep a lightweight runner that prefers websockets but falls back to REST polling
        self.client = client
        self.symbol = symbol
        self.interval = interval
        self.on_candle_closed = on_candle_closed
        self.twm: Optional[ThreadedWebsocketManager] = None
        self._started = False
        self._poll_thread: Optional[threading.Thread] = None
        self._last_close_ts = None
        self.max_attempts = None
        self.last_exc = None

    def start(self):
        self.max_attempts = 5
        try:
            self.max_attempts = int(os.getenv('WS_MAX_START_RETRIES', '5'))
        except Exception:
            pass
        attempt = 0
        backoff = 1
        self.last_exc = None
        while attempt < self.max_attempts:
            attempt += 1
            try:
                self.twm = ThreadedWebsocketManager(api_key=self.client.API_KEY, api_secret=self.client.API_SECRET)
                self.twm.start()
                # subscribe kline socket
                try:
                    self.twm.start_kline_socket(callback=self._handle_msg, symbol=self.symbol, interval=self.interval)
                except Exception as e:
                    logger.error(f"[ERROR] start_kline_socket: {e}")
                    try:
                        from src.utils import get_data_file
                        with open(get_data_file('bot_run.log'), 'a', encoding='utf-8') as logf:
                            logf.write(f"[ERROR] start_kline_socket: {e}\n")
                            logf.write(traceback.format_exc())
                    except Exception:
                        pass
                self._started = True
                logger.info(f"[WebSocket] Connection opened (attempt {attempt})")
                break
            except Exception as e:
                self.last_exc = e
                logger.error(f"[ERROR] WSRunner.start attempt {attempt}/{self.max_attempts} failed: {e}")
                try:
                    from src.utils import get_data_file
                    with open(get_data_file('bot_run.log'), 'a', encoding='utf-8') as logf:
                        logf.write(f"[ERROR] WSRunner.start attempt {attempt}/{self.max_attempts} failed: {e}\n")
                        logf.write(traceback.format_exc())
                except Exception:
                    pass
                try:
                    time.sleep(backoff)
                except Exception:
                    pass
                backoff = min(backoff * 2, 30)
                self.twm = None
                backoff = min(backoff * 2, 30)
                self.twm = None
    def _handle_msg(self, msg):
        import traceback
        try:
            # ...existing code...
            if self.on_candle_closed:
                self.on_candle_closed(msg)
        except Exception as e:
            logger.error(f"[ERROR] _handle_msg: {e}")
            try:
                from src.utils import get_data_file
                with open(get_data_file('bot_run.log'), 'a', encoding='utf-8') as logf:
                    logf.write(f"[ERROR] _handle_msg: {e}\n")
                    logf.write(traceback.format_exc())
            except Exception:
                pass
            logger.warning(f"[WS] Websocket manager failed to start after {self.max_attempts} attempts ({self.last_exc!r}), falling back to REST polling")
        if not self._started:
            logger.warning(f"[WS] Websocket manager failed to start after {self.max_attempts} attempts ({self.last_exc!r}), falling back to REST polling")

            # start REST-based polling fallback
            self._started = True
            self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._poll_thread.start()
            logger.info("[WS-REST] Polling fallback started")

            # start monitor to attempt restarts if possible
            try:
                self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
                self._monitor_thread.start()
                logger.info("[WS] monitor started")
            except Exception:
                pass

    def stop(self):
        if self.twm and self._started:
            try:
                self.twm.stop()
            except Exception:
                pass
            self._started = False
            logger.info("[WebSocket] Connection closed")
        # stop polling thread if used
        if self._poll_thread and self._poll_thread.is_alive():
            try:
                self._started = False
                self._poll_thread.join(timeout=2)
            except Exception:
                pass
            logger.info("[WS-REST] Polling stopped")

    def _handle_msg(self, msg):
        try:
            if msg.get("e") != "kline":
                return
            k = msg.get("k", {})
            if k.get("x"):  # closed candle
                candle = {
                    "timestamp": pd.to_datetime(k.get("t"), unit="ms"),
                    "open": float(k.get("o")),
                    "high": float(k.get("h")),
                    "low": float(k.get("l")),
                    "close": float(k.get("c")),
                    "volume": float(k.get("v")),
                }
                self.on_candle_closed(candle)
        except Exception as e:
            logger.error(f"[WS ERROR] {e}")

    def _interval_to_seconds(self, interval: str) -> int:
        # handle common intervals
        if interval.endswith('m'):
            return int(interval[:-1]) * 60
        if interval.endswith('h'):
            return int(interval[:-1]) * 3600
        if interval.endswith('d'):
            return int(interval[:-1]) * 86400
        # default fallback
        return 60

    def _poll_loop(self):
        # simple REST polling that checks last kline close time and calls on_candle_closed when it changes
        sleep_sec = max(5, self._interval_to_seconds(self.interval) // 6)
        try:
            while self._started:
                try:
                    klines = self.client.get_klines(symbol=self.symbol, interval=self.interval, limit=2)
                    if not klines:
                        time.sleep(sleep_sec)
                        continue
                    last = klines[-1]
                    close_time = int(last[6])
                    if self._last_close_ts is None or close_time != self._last_close_ts:
                        # build candle dict similar to WS handler
                        candle = {
                            "timestamp": pd.to_datetime(int(last[0]), unit='ms'),
                            "open": float(last[1]),
                            "high": float(last[2]),
                            "low": float(last[3]),
                            "close": float(last[4]),
                            "volume": float(last[5]),
                        }
                        try:
                            self.on_candle_closed(candle)
                        except Exception as e:
                            logger.error(f"[WS-REST ERROR] on_candle_closed failed: {e}")
                        self._last_close_ts = close_time
                    time.sleep(sleep_sec)
                except Exception as e:
                    logger.error(f"[WS-REST] polling error: {e}")
                    time.sleep(2)
        finally:
            logger.info("[WS-REST] poll loop exiting")

def preload_history(client, symbol: str, interval: str = "5m", limit: int = 600):
    """
    Fetch historical klines using REST and support pagination when `limit` is larger
    than the exchange max per-request (commonly 1000). Returns a DataFrame with up
    to `limit` rows (most recent first).
    """
    try:
        max_per_call = 1000
        remaining = int(limit)
        all_klines = []
        end_time = None
        # Fetch in reverse chronological order: request the most recent `batch` klines,
        # prepend to list and continue until we've collected `limit` or no more data.
        attempt = 0
        max_retries = 3
        while remaining > 0:
            batch = min(remaining, max_per_call)
            try:
                # Binance client supports `limit` and optional `endTime` to page backwards
                if end_time:
                    klines = client.get_klines(symbol=symbol, interval=interval, limit=batch, endTime=end_time)
                else:
                    klines = client.get_klines(symbol=symbol, interval=interval, limit=batch)
                # Log what we requested and what we got
                logger.info(f"[PRELOAD] requested={batch} end_time={end_time} got={len(klines) if klines else 0}")
                if not klines:
                    break
                # Prepend older batch before existing entries (maintain chronological order)
                all_klines = klines + all_klines
                earliest_open = int(all_klines[0][0]) if all_klines and all_klines[0] else None
                # If server returned fewer than asked and it's not the first fetch, assume no more historical data
                if earliest_open is None:
                    break
                if len(klines) < batch:
                    # fewer returned than requested — likely no more data available; stop
                    logger.info(f"[PRELOAD] fewer returned than requested (got {len(klines)} < {batch}), stopping pagination")
                    break
                # Prepare for next page: request older candles than earliest_open
                end_time = earliest_open - 1
                remaining = limit - len(all_klines)
                if remaining <= 0:
                    break
                # small pause to avoid rate-limits
                time.sleep(0.2)
                attempt = 0
            except Exception as e:
                attempt += 1
                logger.warning(f"[PRELOAD] attempt {attempt} failed: {e}")
                if attempt >= max_retries:
                    logger.error(f"[PRELOAD] failed after {attempt} attempts: {e}")
                    break
                # exponential backoff
                time.sleep(0.5 * (2 ** (attempt-1)))
                continue
        if not all_klines:
            return None
        # Trim to requested limit (keep most recent `limit` candles)
        if len(all_klines) > limit:
            all_klines = all_klines[-limit:]
        df = pd.DataFrame(all_klines, columns=[
            "open_time","open","high","low","close","volume","close_time",
            "quote_asset_volume","number_of_trades","taker_buy_base","taker_buy_quote","ignore"]
        )
        # types
        df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
        df = df[["timestamp","open","high","low","close","volume"]]
        return df
    except Exception as e:
        logger.error(f"[ERROR] preload_history failed: {e}")
        return None


def _monitor_loop(self):
    """Best-effort monitor used when running in REST-fallback mode: detect dead twm and try restart."""
    try:
        while getattr(self, '_started', False):
            try:
                twm = getattr(self, 'twm', None)
                alive = True
                if twm is not None:
                    try:
                        if hasattr(twm, 'is_alive'):
                            alive = twm.is_alive()
                        elif getattr(twm, '_thread', None) is not None:
                            alive = getattr(twm, '_thread').is_alive()
                    except Exception:
                        alive = True

                if not alive:
                    logger.warning("[WS] detected dead websocket manager in monitor, attempting restart")
                    try:
                        try:
                            if twm is not None:
                                twm.stop()
                        except Exception:
                            pass
                        # attempt simple restart
                        new = ThreadedWebsocketManager(api_key=self.client.API_KEY, api_secret=self.client.API_SECRET)
                        new.start()
                        try:
                            new.start_kline_socket(callback=self._handle_msg, symbol=self.symbol, interval=self.interval)
                        except Exception:
                            pass
                        self.twm = new
                        logger.info("[WS] monitor restarted websocket manager")
                    except Exception as e:
                        logger.error(f"[WS] monitor restart failed: {e}")
                time.sleep(15)
            except Exception:
                time.sleep(5)
    except Exception:
        pass
