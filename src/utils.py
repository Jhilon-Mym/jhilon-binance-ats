from src.profit_tracker import ProfitTracker

# Global profit tracker instance (can be moved to main bot context)
profit_tracker = ProfitTracker(symbol='BTCUSDT')
import json
import os
import numbers
import threading
import tempfile
import time
from decimal import Decimal, ROUND_DOWN
from binance.client import Client
import importlib
try:
    _dotenv = importlib.import_module("dotenv")
    load_dotenv = getattr(_dotenv, "load_dotenv", None)
    if load_dotenv is None:
        def load_dotenv(*args, **kwargs):
            # No-op fallback when python-dotenv does not expose load_dotenv
            return None
except Exception:
    def load_dotenv(*args, **kwargs):
        # No-op fallback when python-dotenv is not installed or import cannot be resolved.
        return None

PENDING_FILE = "pending_trades.json"
HISTORY_FILE = "trade_history.json"
ORDERS_LOG = "orders.log"


def place_order_live(client: Client, side: str, symbol: str, quantity: float, price=None, order_type="MARKET"):
    """
    Place a live order on Binance (MARKET by default).
    client: an instance of binance.client.Client
    side: 'BUY' or 'SELL'
    symbol: trading pair e.g. 'BTCUSDT'
    quantity: base asset quantity (float or Decimal)
    price: price for LIMIT orders
    order_type: 'MARKET' or 'LIMIT'
    """
    try:
        if order_type == "MARKET":
            order = client.create_order(
                symbol=symbol,
                side=side,
                type="MARKET",
                quantity=str(quantity)
            )
        elif order_type == "LIMIT" and price:
            order = client.create_order(
                symbol=symbol,
                side=side,
                type="LIMIT",
                timeInForce="GTC",
                quantity=str(quantity),
                price=str(price)
            )
        else:
            raise ValueError("Invalid order type or missing price for LIMIT order")

        # Extract fee and commission info from order response
        fee = 0.0
        commissionAsset = 'USDT'
        if order and 'fills' in order:
            for fill in order['fills']:
                fee += float(fill.get('commission', 0.0))
                commissionAsset = fill.get('commissionAsset', 'USDT')
        # Add trade to profit tracker
        mark_price = price if price else None
        profit_tracker.add_trade(side, float(quantity), float(order.get('price', price if price else 0)), fee, commissionAsset)
        if mark_price:
            profit_tracker.update_unrealized(mark_price)
        return order
    except Exception as e:
        print(f"[ERROR] Failed to place {side} order: {e}")
        return None


def load_env():
    """Convenience wrapper to load environment variables from .env for external modules."""
    try:
        # Use the module-level load_dotenv created at import time (which falls back to a no-op
        # when python-dotenv is not installed or doesn't expose load_dotenv).
        if callable(load_dotenv):
            load_dotenv()
    except Exception:
        pass


def reconcile_pending_with_exchange(client: Client, symbol: str = None):
    """Wrapper to reconcile pending trades using a provided Binance client instance.

    This function will instantiate a TradeStats, attach the provided client and call
    the existing TradeStats.reconcile_pending_with_exchange implementation.
    """
    try:
        stats = TradeStats(use_real_client=False)
        stats.client = client
        stats.reconcile_pending_with_exchange(symbol=symbol)
    except Exception as e:
        print(f"[ERROR] reconcile_pending_with_exchange wrapper failed: {e}")
        return None


class TradeStats:
    def update_profit_tracker(self, trade, mark_price=None):
        """
        Update profit tracker after trade execution or close.
        """
        side = trade.get('side')
        qty = float(trade.get('qty', 0))
        price = float(trade.get('entry', 0))
        fee = float(trade.get('fee', 0.0))
        commissionAsset = trade.get('commissionAsset', 'USDT')
        profit_tracker.add_trade(side, qty, price, fee, commissionAsset)
        if mark_price:
            profit_tracker.update_unrealized(mark_price)
    def __init__(self, use_real_client=True):
        # If use_real_client is False, avoid creating Binance Client (useful for tests)
        load_dotenv()
        self.api_key = os.getenv("BINANCE_API_KEY")
        self.api_secret = os.getenv("BINANCE_API_SECRET")
        self.use_testnet = os.getenv("USE_TESTNET", "true").lower() == "true"
        self.client = None
        if use_real_client:
            try:
                self.client = Client(self.api_key, self.api_secret, testnet=self.use_testnet)
            except Exception:
                self.client = None

        self.total = 0
        self.win = 0
        self.loss = 0
        # protect file reads/writes
        self._lock = threading.Lock()
        # initialize balances
        self.balance = self.get_real_balance()
        # cached balances map (asset -> free amount) updated by print_balances()
        self.cached_balances = {}
        self.pending = self.load_pending()


    def load_pending(self):
        if os.path.exists(PENDING_FILE):
            try:
                with open(PENDING_FILE, "r") as f:
                    data = json.load(f)
            except Exception:
                # corrupted file -> reset safely
                print("[WARN] pending_trades.json malformed, resetting to empty list")
                try:
                    with self._lock:
                        with tempfile.NamedTemporaryFile('w', delete=False, dir='.', prefix='pending_', suffix='.json') as tf:
                            json.dump([], tf)
                        os.replace(tf.name, PENDING_FILE)
                except Exception:
                    pass
                return []
            # Support both list and dict format for backward compatibility
            if isinstance(data, dict) and "open_trades" in data:
                return data["open_trades"]
            elif isinstance(data, list):
                return data
            else:
                print("[WARN] Unknown pending_trades.json format, resetting pending trades.")
                return []
        return []

    def get_real_balance(self):
        if not self.client:
            return 0.0
        try:
            bal = self.client.get_asset_balance(asset='USDT')
            return float(bal['free'])
        except Exception as e:
            print(f"[ERROR] Could not fetch real balance: {e}")
            return 0.0

    def _to_native(self, obj):
        # Convert numpy and Decimal types to native Python for JSON
        if isinstance(obj, dict):
            return {k: self._to_native(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._to_native(v) for v in obj]
        if isinstance(obj, numbers.Number):
            return obj.item() if hasattr(obj, 'item') else obj
        return obj

    def save_pending(self):
        # Atomic write to avoid partial/corrupt files
        try:
            with self._lock:
                with tempfile.NamedTemporaryFile('w', delete=False, dir='.', prefix='pending_', suffix='.json') as tf:
                    json.dump(self._to_native(self.pending), tf, indent=2)
                os.replace(tf.name, PENDING_FILE)
        except Exception as e:
            print(f"[ERROR] Failed to save pending trades atomically: {e}")


    def open_trade(self, signal, entry_price, qty, amount_usdt, order_info=None):
        # Restrict total pending trade amount to 75% of current balance.
        # In test environments or when balance==0 (no real client), allow opening trades
        # so unit tests can run deterministically.
        pending_usdt = sum(t.get('amount_usdt', 0) for t in self.pending if t.get('status') == 'open')
        if self.balance and float(self.balance) > 0:
            max_allowed = 0.75 * float(self.balance)
            if pending_usdt + float(amount_usdt) > max_allowed:
                print(f"[WARN] Pending trades would exceed 75% of balance: {pending_usdt + float(amount_usdt):.2f} > {max_allowed:.2f}. Skipping new trade.")
                import logging
                logging.warning(f"Pending trades would exceed 75% of balance: {pending_usdt + float(amount_usdt):.2f} > {max_allowed:.2f}. Skipping new trade.")
                return
        else:
            # No balance info (likely in tests) — allow opening trades
            max_allowed = None
        # Record an open trade with quantity and USD notional. Optionally attach
        # exchange order response data (orderId, executedQty, cummulativeQuoteQty, fills).
        trade = {
            "side": signal["side"],
            "entry": float(entry_price),
            "qty": float(qty),
            "amount_usdt": float(amount_usdt),
            "sl": float(signal["sl"]),
            "tp": float(signal["tp"]),
            "status": "open",
        }

        # Always assign a unique orderId if not present (for test/live)
        import time, random
        if order_info and isinstance(order_info, dict) and 'orderId' in order_info:
            trade['orderId'] = order_info.get('orderId')
        else:
            # Generate a unique orderId for test trades or if exchange did not return one
            trade['orderId'] = int(time.time() * 1000) + random.randint(0, 999)

        # Attach order response details when available and normalize
        if order_info and isinstance(order_info, dict):
            try:
                # Log raw order to orders.log for an audit trail
                try:
                    with open(ORDERS_LOG, 'a') as ol:
                        import json, time
                        ol.write(json.dumps({'ts': time.time(), 'order': self._to_native(order_info)}) + "\n")
                except Exception:
                    pass

                if 'orderId' in order_info:
                    trade['orderId'] = order_info.get('orderId')
                # If exchange returned executedQty/cummulativeQuoteQty, prefer those values
                if 'executedQty' in order_info:
                    try:
                        exec_q = float(order_info.get('executedQty'))
                        trade['qty'] = exec_q
                        trade['executedQty'] = exec_q
                    except Exception:
                        trade['executedQty'] = order_info.get('executedQty')
                if 'cummulativeQuoteQty' in order_info:
                    try:
                        cq = float(order_info.get('cummulativeQuoteQty'))
                        trade['amount_usdt'] = cq
                        trade['cummulativeQuoteQty'] = cq
                    except Exception:
                        trade['cummulativeQuoteQty'] = order_info.get('cummulativeQuoteQty')
                if 'fills' in order_info:
                    # normalize fills into native structures
                    trade['fills'] = self._to_native(order_info.get('fills'))
            except Exception:
                # Non-critical: ignore order_info extraction failures
                pass
        # Prevent accidental duplicate open trades.
        # If order_info provided with orderId, treat orderId as the uniqueness key.
        is_dup = False
        order_id = None
        if order_info and isinstance(order_info, dict):
            order_id = order_info.get('orderId')
        for t in self.pending:
            try:
                if t.get('status') != 'open':
                    continue
                if order_id is not None:
                    # If existing pending trade already has same orderId, consider duplicate
                    if t.get('orderId') == order_id:
                        is_dup = True
                        break
                else:
                    # Fallback duplicate heuristic: side + entry + amount_usdt
                    if t.get('side') == trade['side'] and abs(t.get('entry', 0) - trade['entry']) < 1e-9 and abs(t.get('amount_usdt', 0) - trade['amount_usdt']) < 1e-9:
                        is_dup = True
                        break
            except Exception:
                continue
        if is_dup:
            print('[WARN] Duplicate open trade detected; skipping append')
            return

        self.pending.append(trade)
        if trade.get("side") == "SELL":
            import logging
            logging.info(f"[DEBUG] SELL trade appended to pending: {trade}")
        self.save_pending()
        # Update profit tracker for new trade
        self.update_profit_tracker(trade, mark_price=entry_price)

    def update_pending(self, candle):
        """
        Update all open trades using trailing/profit-only close logic from ManagedTrade.
        Assumes candle contains 'close', 'high', 'low', and 'atr' (if available).
        """
        from src.config import Config
        try:
            from src.trade_manager import ManagedTrade
        except ImportError:
            print("[ERROR] Could not import ManagedTrade for trailing logic!")
            return []
        closed = []
        for trade in list(self.pending):
            if trade["status"] != "open":
                continue
            # Use close/high/low/atr for trailing/profit logic
            price = candle.get("close")
            high = candle.get("high")
            low = candle.get("low")
            atr = candle.get("atr")
            # Fallback: estimate ATR if not present
            if atr is None:
                atr = abs(high - low) if high is not None and low is not None else 0
            mt = ManagedTrade(
                side=trade["side"],
                entry=trade["entry"],
                sl=trade["sl"],
                tp=trade["tp"],
                trail_active=trade.get("trail_active", False),
                trail_stop=trade.get("trail_stop")
            )
            # Check all price points in candle for hit (simulate intra-candle moves)
            should_close = None
            close_price_override = None
            # For BUY: check high for TP, low for SL/trailing; for SELL: low for TP, high for SL/trailing
            if trade["side"] == "BUY":
                # Simulate price moving: entry → high → low → close
                # 1. TP hit
                if high is not None and high >= trade["tp"]:
                    res = mt.update(trade["tp"], atr, Config)
                    if res == "close":
                        should_close = res
                        close_price_override = float(trade["tp"])
                # 2. Trailing stop hit
                if not should_close and mt.trail_active and low is not None and mt.trail_stop is not None and low <= mt.trail_stop:
                    res = mt.update(mt.trail_stop, atr, Config)
                    if res == "close":
                        should_close = res
                        close_price_override = float(mt.trail_stop)
                # 3. SL hit (only if in profit, handled by mt.update)
                if not should_close and low is not None and low <= trade["sl"]:
                    res = mt.update(trade["sl"], atr, Config)
                    if res == "close":
                        should_close = res
                        close_price_override = float(trade["sl"])
                # 4. Candle close
                if not should_close and price is not None:
                    res = mt.update(price, atr, Config)
                    if res == "close":
                        should_close = res
                        close_price_override = float(price)
            elif trade["side"] == "SELL":
                # 1. TP hit
                if low is not None and low <= trade["tp"]:
                    res = mt.update(trade["tp"], atr, Config)
                    if res == "close":
                        should_close = res
                        close_price_override = float(trade["tp"])
                # 2. Trailing stop hit
                if not should_close and mt.trail_active and high is not None and mt.trail_stop is not None and high >= mt.trail_stop:
                    res = mt.update(mt.trail_stop, atr, Config)
                    if res == "close":
                        should_close = res
                        close_price_override = float(mt.trail_stop)
                # 3. SL hit (only if in profit, handled by mt.update)
                if not should_close and high is not None and high >= trade["sl"]:
                    res = mt.update(trade["sl"], atr, Config)
                    if res == "close":
                        should_close = res
                        close_price_override = float(trade["sl"])
                # 4. Candle close
                if not should_close and price is not None:
                    res = mt.update(price, atr, Config)
                    if res == "close":
                        should_close = res
                        close_price_override = float(price)
            # If should_close, mark as win/loss (profit-only close)
            if should_close == "close":
                # Determine win/loss by entry vs. close and compute realized PnL in USDT
                # Use amount_usdt (notional) to estimate PnL: pnl_pct = (close - entry)/entry for BUY
                try:
                    amount = float(trade.get("amount_usdt", 0))
                    entry = float(trade.get("entry", 0))
                    # choose close price: override if we captured intra-candle trigger, else use candle close
                    price_for_calc = None
                    if close_price_override is not None:
                        price_for_calc = close_price_override
                    else:
                        price_for_calc = price
                    if price_for_calc is None:
                        # fallback: treat as no PnL
                        pnl_pct = 0.0
                    else:
                        if trade["side"] == "BUY":
                            pnl_pct = (price_for_calc - entry) / entry if entry != 0 else 0.0
                        else:
                            pnl_pct = (entry - price_for_calc) / entry if entry != 0 else 0.0
                except Exception:
                    pnl_pct = 0.0
                    amount = float(trade.get("amount_usdt", 0))

                realized_usdt = pnl_pct * amount
                # Mark win/loss by realized PnL > 0
                if realized_usdt > 0:
                    trade["status"] = "win"
                    self.win += 1
                else:
                    trade["status"] = "loss"
                    self.loss += 1

                # Adjust session balance by realized PnL (not by full notional)
                try:
                    self.balance += float(realized_usdt)
                except Exception:
                    pass

                # Save trailing state and realized metrics for audit
                trade["trail_active"] = mt.trail_active
                trade["trail_stop"] = mt.trail_stop
                trade["close_price"] = float(price_for_calc) if price_for_calc is not None else None
                trade["realized_pnl_usdt"] = float(realized_usdt)
                trade["pnl_pct"] = float(pnl_pct)
                closed.append(trade)
                self.save_trade_history(trade)
                if trade.get("side") == "SELL":
                    import logging
                    logging.info(f"[DEBUG] SELL trade closed and written to history: {trade}")
            else:
                # Update trailing state in open trade
                trade["trail_active"] = mt.trail_active
                trade["trail_stop"] = mt.trail_stop
        # Remove closed trades from pending
        self.pending = [t for t in self.pending if t["status"] == "open"]
        self.save_pending()
        # Update profit tracker for closed trades
        for t in closed:
            self.update_profit_tracker(t, mark_price=t.get('close_price'))
        return closed

    def save_trade_history(self, trade):
        try:
            # Use atomic write for history as well
            if os.path.exists(HISTORY_FILE):
                try:
                    with open(HISTORY_FILE, "r") as f:
                        history = json.load(f)
                except Exception:
                    history = []
            else:
                history = []
            history.append(self._to_native(trade))
            with tempfile.NamedTemporaryFile('w', delete=False, dir='.', prefix='history_', suffix='.json') as tf:
                json.dump(history, tf, indent=2)
            os.replace(tf.name, HISTORY_FILE)
        except Exception as e:
            print(f"[ERROR] Saving trade history: {e}")

    def sync_pending_with_orders(self):
        """Scan pending trades and normalize qty/amount_usdt where executedQty or cummulativeQuoteQty exist.
        Useful after manual edits to pending_trades.json or when migrating data.
        """
        changed = False
        for t in self.pending:
            try:
                # If executedQty present as string, convert to float and update qty
                if 'executedQty' in t and not isinstance(t.get('qty'), float):
                    try:
                        t['qty'] = float(t.get('executedQty'))
                        changed = True
                    except Exception:
                        pass
                # If cummulativeQuoteQty present, ensure amount_usdt matches it
                if 'cummulativeQuoteQty' in t:
                    try:
                        cq = float(t.get('cummulativeQuoteQty'))
                        if abs(float(t.get('amount_usdt', 0)) - cq) > 1e-9:
                            t['amount_usdt'] = cq
                            changed = True
                    except Exception:
                        pass
            except Exception:
                continue
        if changed:
            self.save_pending()
        return changed

    def summary(self):
        pending_amt = sum(t.get("amount_usdt", 0) for t in self.pending if t.get("status") == "open")
        # Session P/L: বর্তমান ব্যালান্স - শুরুতে ছিল (ধরা হচ্ছে শুরুতে 0, বা চাইলে initial_balance ফিল্ড যোগ করা যায়)
        session_pl = self.balance
        return (
            f"Win: {self.win}, Loss: {self.loss}, "
            f"Balance: {self.balance:.2f} USDT, Pending: {len(self.pending)}, Pending Amount: {pending_amt} USDT, Session P/L: {session_pl:.2f}"
        )

    def print_balances(self):
        """Fetch and print available USDT and BTC balances from Binance with retry/backoff."""
        if not self.client:
            print("[BALANCE] Binance client not initialized.")
            return
        # retry with exponential backoff
        retries = 3
        attempt = 0
        base_wait = 1.0
        while attempt < retries:
            try:
                usdt = self.client.get_asset_balance(asset='USDT')
                btc = self.client.get_asset_balance(asset='BTC')
                usdt_free = float(usdt['free']) if usdt and 'free' in usdt else 0.0
                btc_free = float(btc['free']) if btc and 'free' in btc else 0.0
                # update cached balances and session balance
                self.cached_balances['USDT'] = usdt_free
                self.cached_balances['BTC'] = btc_free
                # keep stats.balance as available USDT for BUY pre-flight checks
                self.balance = usdt_free
                print(f"[BALANCE] Available: {usdt_free:.2f} USDT, {btc_free:.6f} BTC")
                return
            except Exception as e:
                attempt += 1
                wait = base_wait * (2 ** (attempt - 1))
                print(f"[ERROR] Could not fetch balances (attempt {attempt}/{retries}): {e}; retrying in {wait:.1f}s")
                try:
                    time.sleep(wait)
                except Exception:
                    pass
        # final fallback: use cached balances if available
        print("[ERROR] Failed to fetch balances after retries; using cached balances if available.")
        usdt_free = self.cached_balances.get('USDT', None)
        btc_free = self.cached_balances.get('BTC', None)
        if usdt_free is not None or btc_free is not None:
            usdt_str = f"{usdt_free:.2f}" if usdt_free is not None else "N/A"
            btc_str = f"{btc_free:.6f}" if btc_free is not None else "N/A"
            print(f"[BALANCE] Cached: {usdt_str} USDT, {btc_str} BTC")
        return

    def check_order_status(self, order_id, symbol=None):
        """Query Binance for a specific orderId and return the order dict or None.

        This is a safe read-only helper that can be used to cross-check whether an
        orderId recorded in local logs is actually visible in the exchange account.
        """
        if not self.client:
            print("[ORDER] Binance client not initialized.")
            return None
        try:
            from src.config import Config
            sym = symbol or Config.SYMBOL
            order = self.client.get_order(symbol=sym, orderId=order_id)
            print(f"[ORDER] orderId={order_id} status={order.get('status')} executedQty={order.get('executedQty')} cummulativeQuoteQty={order.get('cummulativeQuoteQty')}")
            return self._to_native(order)
        except Exception as e:
            print(f"[ERROR] Could not fetch order {order_id}: {e}")
            return None

    
    def reconcile_pending_with_exchange(self, symbol=None):
        """
        Cross-check locally stored 'open' trades with Binance order status using orderId.
        - If order not found or status in {CANCELED, REJECTED, EXPIRED}, mark trade 'canceled' and remove from pending.
        - If order is FILLED (market order typical), keep the trade open for trailing/TP management.
        - If partially filled, keep it open but update qty and USDT amount from exchange.
        This helps when the bot restarts or crashed while orders were in-flight.
        """
        removed = 0
        updated = 0
        if not self.client:
            print("[RECONCILE] Binance client not initialized; cannot reconcile pending orders.")
            return

        try:
            sym = symbol
            try:
                if sym is None:
                    from src.config import Config
                    sym = Config.SYMBOL
            except Exception:
                pass

            for trade in list(self.pending):
                if trade.get('status') != 'open':
                    continue
                info = (trade.get('order_info') or {})
                order_id = info.get('orderId') or info.get('order_id')
                if not order_id:
                    # nothing to verify; keep but warn
                    print(f"[RECONCILE] Trade without orderId kept as-is: side={trade.get('side')} entry={trade.get('entry')}")
                    continue

                try:
                    order = self.client.get_order(symbol=sym, orderId=int(order_id))
                except Exception as e:
                    print(f"[RECONCILE] Could not fetch orderId {order_id}: {e}")
                    continue

                status = (order or {}).get('status', 'UNKNOWN')
                if status in ('CANCELED', 'REJECTED', 'EXPIRED'):
                    # remove from pending as it never executed
                    self.pending.remove(trade)
                    removed += 1
                    print(f"[RECONCILE] Removing canceled/rejected orderId={order_id}")
                    continue

                if status in ('NEW', 'PARTIALLY_FILLED'):
                    # update quantities from exchange
                    try:
                        executed_qty = float(order.get('executedQty', 0.0))
                        cquote = float(order.get('cummulativeQuoteQty', 0.0))
                        if executed_qty > 0:
                            trade['qty'] = executed_qty
                        if cquote > 0:
                            trade['amount_usdt'] = cquote
                        trade['order_info'] = self._to_native(order)
                        updated += 1
                    except Exception:
                        pass

                if status == 'FILLED':
                    # ensure order info stored
                    trade['order_info'] = self._to_native(order)
                    updated += 1

            if removed or updated:
                self.save_pending()
            print(f"[RECONCILE] Done. updated={updated} removed={removed}. open={sum(1 for t in self.pending if t.get('status')=='open')}")
        except Exception as e:
            print(f"[RECONCILE] Unexpected error: {e}")
def get_cached_balance(self, asset):
        """Return cached free balance for asset or None if not available."""
        try:
            v = self.cached_balances.get(asset)
            return float(v) if v is not None else None
        except Exception:
            return None

from binance.client import Client
import pandas as pd

def fetch_klines_rest(symbol="BTCUSDT", interval="5m", limit=600):
    """
    Fetch historical klines (candles) via Binance REST API and return DataFrame.
    """
    load_env()
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    use_testnet = os.getenv("USE_TESTNET", "true").lower() == "true"

    client = Client(api_key, api_secret, testnet=use_testnet)

    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)

    df = pd.DataFrame(klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)

    return df
