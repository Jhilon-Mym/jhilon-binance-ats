
# websocket_bot.py — WS runner + REST preload (no CSV fallback)
import logging
import pandas as pd
import threading, time, os
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

    def start(self):
        # Try starting the threaded websocket manager with retry/backoff
        try:
            max_attempts = int(os.getenv('WS_MAX_START_RETRIES', '5'))
        except Exception:
            max_attempts = 5
        attempt = 0
        backoff = 1
        last_exc = None
        while attempt < max_attempts:
            attempt += 1
            try:
                self.twm = ThreadedWebsocketManager(api_key=self.client.API_KEY, api_secret=self.client.API_SECRET)
                self.twm.start()
                # subscribe kline socket
                try:
                    self.twm.start_kline_socket(callback=self._handle_msg, symbol=self.symbol, interval=self.interval)
                except Exception:
                    pass
                self._started = True
                logger.info(f"[WebSocket] Connection opened (attempt {attempt})")
                break
            except Exception as e:
                last_exc = e
                logger.warning(f"[WS] start attempt {attempt}/{max_attempts} failed: {e!r}")
                try:
                    time.sleep(backoff)
                except Exception:
                    pass
                backoff = min(backoff * 2, 30)
                self.twm = None

        if not self._started:
            logger.warning(f"[WS] Websocket manager failed to start after {max_attempts} attempts ({last_exc!r}), falling back to REST polling")

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
    try:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=[
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
