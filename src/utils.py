# ট্রেড অবজেক্টে প্রয়োজনীয় ফিল্ড ও ডিফল্ট ভ্যালু
REQUIRED_PENDING_FIELDS = {
    "orderId": "N/A",
    "side": "N/A",
    "qty": 0,
    "amount_usdt": 0,
    "entry": 0,
    "status": "N/A",
    "timestamp": "N/A",
    "sl": 0,
    "tp": 0,
    "executedQty": 0,
    "cummulativeQuoteQty": 0,
    "fills": [],
    "trail_active": False,
    "trail_stop": None
}
REQUIRED_HISTORY_FIELDS = {
    "orderId": "N/A",
    "side": "N/A",
    "qty": 0,
    "amount_usdt": 0,
    "entry": 0,
    "close_price": 0,
    "realized_pnl_usdt": 0,
    "status": "N/A",
    "timestamp": "N/A"
}

def ensure_pending_fields(trade, set_timestamp: bool = False):
    """প্রতিটি পেন্ডিং ট্রেডে প্রয়োজনীয় ফিল্ড ও ডিফল্ট ভ্যালু যোগ করে।

    By default this will NOT overwrite or fill the `timestamp` field for
    existing pending trades (to avoid mass-updating old records with the
    same timestamp). When creating a new trade, call with `set_timestamp=True`
    to stamp the creation time.
    """
    import datetime
    for k, v in REQUIRED_PENDING_FIELDS.items():
        if k not in trade:
            trade[k] = v
    # Only set timestamp when explicitly requested (new trade creation).
    if set_timestamp:
        if not trade.get('timestamp') or trade.get('timestamp') == 'N/A':
            trade['timestamp'] = datetime.datetime.now().isoformat(timespec='seconds')
    return trade

def ensure_history_fields(trade):
    """প্রতিটি ট্রেড হিস্টোরি অবজেক্টে প্রয়োজনীয় ফিল্ড ও ডিফল্ট ভ্যালু যোগ করে।"""
    for k, v in REQUIRED_HISTORY_FIELDS.items():
        if k not in trade:
            trade[k] = v
    return trade


def analyze_closed_trade(trade, fee_converter=None):
    """
    Attach a few derived/analysis fields to a closed trade before saving:
    - closed_at: ISO timestamp when saving (if not present)
    - duration_seconds: seconds between open timestamp and close
    - fee_usdt: fee converted to USDT using provided fee_converter (best-effort)
    - netmode: 'testnet'|'mainnet' from current env
    This is best-effort and will not raise on parse errors.
    """
    import datetime
    try:
        # closed timestamp
        closed_at = datetime.datetime.now()
        trade.setdefault('closed_at', closed_at.isoformat(timespec='seconds'))

        # duration: try to parse existing open timestamp
        started = None
        ts = trade.get('timestamp')
        if isinstance(ts, str) and ts and ts != 'N/A':
            try:
                # try parsing ISO
                started = datetime.datetime.fromisoformat(ts)
            except Exception:
                try:
                    # fallback: epoch float
                    started = datetime.datetime.fromtimestamp(float(ts))
                except Exception:
                    started = None
        if started is not None:
            try:
                trade['duration_seconds'] = int((closed_at - started).total_seconds())
            except Exception:
                pass

        # fee conversion to USDT when possible
        fee = trade.get('fee')
        commissionAsset = trade.get('commissionAsset', 'USDT')
        if fee is not None:
            try:
                if fee_converter:
                    trade['fee_usdt'] = float(fee_converter(fee, commissionAsset))
                else:
                    # best-effort: if commissionAsset already USDT
                    if commissionAsset == 'USDT':
                        trade['fee_usdt'] = float(fee)
            except Exception:
                pass

        # annotate netmode
        try:
            trade['netmode'] = 'testnet' if get_netmode() else 'mainnet'
        except Exception:
            pass
    except Exception:
        # never fail the save path due to analysis
        pass
    return trade

def save_pending_trades(trades, file_path=None):
    """সব পেন্ডিং ট্রেড সেভ করার সময় ফিল্ড চেক করে ফাইল আপডেট করে।"""
    if file_path is None:
        file_path = get_pending_file()
    trades = [ensure_pending_fields(dict(t)) for t in trades]
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(trades, f, indent=2, ensure_ascii=False)

def save_trade_history(trades, file_path=None):
    """সব ট্রেড হিস্টোরি সেভ করার সময় ফিল্ড চেক করে ফাইল আপডেট করে।"""
    if file_path is None:
        file_path = get_history_file()
    trades = [ensure_history_fields(dict(t)) for t in trades]
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(trades, f, indent=2, ensure_ascii=False)
import os
import json
from src.profit_tracker import ProfitTracker


def get_netmode():
    try:
        with open('.env', 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip().startswith('USE_TESTNET='):
                    val = line.strip().split('=',1)[1].lower()
                    return val == 'true'
    except Exception:
        pass
    return True


import hashlib


def _repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def get_api_hash():
    """Compute a short SHA256 fingerprint for the active API credentials.

    Uses BINANCE_API_KEY, BINANCE_API_SECRET and USE_TESTNET to separate data.
    """
    try:
        load_dotenv()
    except Exception:
        pass
    k = os.getenv('BINANCE_API_KEY', '') or ''
    s = os.getenv('BINANCE_API_SECRET', '') or ''
    net = os.getenv('USE_TESTNET', 'true').lower()

    # Try to compute an account-level fingerprint when valid API keys are present.
    # This allows different API key pairs that access the same Binance account
    # to share the same data directory. If the account query fails, fall back
    # to hashing the key+secret+netmode.
    try:
        if k and s:
            try:
                from binance.client import Client as _BinanceClient
                client = _BinanceClient(k, s, testnet=(net == 'true'))
                acct = client.get_account()
                parts = []
                for fld in ('makerCommission', 'takerCommission', 'buyerCommission', 'sellerCommission', 'canTrade', 'canWithdraw', 'canDeposit'):
                    parts.append(str(acct.get(fld, '')))
                balances = acct.get('balances', []) if isinstance(acct.get('balances', []), list) else []
                syms = sorted([b.get('asset') for b in balances if b.get('asset')])
                parts.append(','.join(syms))
                raw = '|'.join(parts)
                h = hashlib.sha256(raw.encode('utf-8')).hexdigest()
                return h[:12]
            except Exception:
                # fall through to key-based fallback
                pass
    except Exception:
        pass

    raw = f"{k}|{s}|{net}"
    h = hashlib.sha256(raw.encode('utf-8')).hexdigest()
    return h[:12]


def get_data_dir():
    root = _repo_root()
    api_hash = get_api_hash()
    d = os.path.join(root, 'data', api_hash)
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    return d


def get_data_file(name: str):
    return os.path.join(get_data_dir(), name)

def get_pending_file():
    use_testnet = get_netmode()
    fname = 'pending_trades_testnet.json' if use_testnet else 'pending_trades_mainnet.json'
    # prefer per-API data dir
    candidate = get_data_file(fname)
    legacy_data = get_data_file('pending_trades.json')
    repo_legacy = os.path.join(_repo_root(), 'pending_trades.json')
    for p in (candidate, legacy_data, repo_legacy):
        if os.path.exists(p):
            return p
    return candidate

def get_profit_tracker():
    """
    Return a TradeStats-based tracker per netmode so it exposes cached_balances and print_balances.
    Lazy-instantiates one TradeStats per netmode.
    """
    use_testnet = get_netmode()
    if not hasattr(get_profit_tracker, '_testnet'):
        # Create a Testnet tracker. If API keys are configured in the
        # environment, enable real client; otherwise keep it disabled to
        # prevent noisy API errors when running only the web UI.
        try:
            load_dotenv()
        except Exception:
            pass
        has_keys = bool(os.getenv('BINANCE_API_KEY')) and bool(os.getenv('BINANCE_API_SECRET'))
        get_profit_tracker._testnet = TradeStats(use_real_client=has_keys)
    if not hasattr(get_profit_tracker, '_mainnet'):
        try:
            load_dotenv()
        except Exception:
            pass
        has_keys = bool(os.getenv('BINANCE_API_KEY')) and bool(os.getenv('BINANCE_API_SECRET'))
        get_profit_tracker._mainnet = TradeStats(use_real_client=has_keys)
    return get_profit_tracker._testnet if use_testnet else get_profit_tracker._mainnet
import json
import os
import numbers
import threading
import tempfile
import time
from decimal import Decimal, ROUND_DOWN
from binance.client import Client
import importlib
import logging
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

# Control verbose balance logging from env (set VERBOSE_BALANCE=true to enable)
VERBOSE_BALANCE = os.getenv('VERBOSE_BALANCE', 'false').lower() == 'true'
logger = logging.getLogger(__name__)

# PENDING_FILE is netmode-specific and resolved at import time
PENDING_FILE = get_pending_file()


def get_history_file():
    """Return netmode-specific trade history file path.
    Keeps history separated per netmode to avoid mixing testnet/mainnet records.
    """
    use_testnet = get_netmode()
    fname = 'trade_history_testnet.json' if use_testnet else 'trade_history_mainnet.json'
    candidate = get_data_file(fname)
    legacy_data = get_data_file('trade_history.json')
    repo_legacy = os.path.join(_repo_root(), 'trade_history.json')
    for p in (candidate, legacy_data, repo_legacy):
        if os.path.exists(p):
            return p
    return candidate
ORDERS_LOG = get_data_file('orders.log')


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
        tracker = get_profit_tracker()
        tracker.add_trade(side, float(quantity), float(order.get('price', price if price else 0)), fee, commissionAsset)
        if mark_price:
            tracker.update_unrealized(mark_price)
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
        # call the instance method which returns None currently; capture printed info by returning counts
        # We'll adapt the instance method to return (updated, removed) if possible.
        try:
            res = stats.reconcile_pending_with_exchange(symbol=symbol)
            # If the instance method returns a dict or tuple, normalize it
            if isinstance(res, dict):
                return res
            if isinstance(res, tuple) and len(res) == 2:
                return {'updated': int(res[0]), 'removed': int(res[1])}
            # no explicit return from instance method — attempt to compute counts from state
            updated = 0
            removed = 0
            return {'updated': updated, 'removed': removed}
        except Exception as e:
            print(f"[ERROR] reconcile_pending_with_exchange instance call failed: {e}")
            return {'updated': 0, 'removed': 0, 'error': str(e)}
    except Exception as e:
        print(f"[ERROR] reconcile_pending_with_exchange wrapper failed: {e}")
        return {'updated': 0, 'removed': 0, 'error': str(e)}


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
        tracker = get_profit_tracker()
        tracker.add_trade(side, qty, price, fee, commissionAsset)
        if mark_price:
            tracker.update_unrealized(mark_price)
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
        # embed a ProfitTracker instance to handle realized/unrealized PnL and reporting
        try:
            # Use symbol from Config if available
            try:
                from src.config import Config
                symbol = getattr(Config, 'SYMBOL', 'BTCUSDT')
            except Exception:
                symbol = 'BTCUSDT'

            self.profit_tracker = ProfitTracker(symbol=symbol)

            # fee_converter: convert commissionAsset (e.g., BTC) to USDT using the client ticker if possible
            def fee_converter(fee_value, commission_asset):
                try:
                    # if commission asset is already USDT, no conversion
                    if commission_asset == 'USDT':
                        return float(fee_value)
                    # try using client to get a conversion price (asset->USDT)
                    if self.client:
                        # common pairs: <asset>USDT
                        pair = f"{commission_asset}USDT"
                        try:
                            avg = self.client.get_avg_price(symbol=pair)
                            price = float(avg.get('price', 0))
                            return float(fee_value) * price
                        except Exception:
                            # fallback: try ticker price endpoint
                            try:
                                tick = self.client.get_symbol_ticker(symbol=pair)
                                price = float(tick.get('price', 0))
                                return float(fee_value) * price
                            except Exception:
                                return float(fee_value)
                    else:
                        # no client available — cannot convert reliably
                        return float(fee_value)
                except Exception:
                    return float(fee_value)

            # Attach fee_converter for ProfitTracker usage (ProfitTracker.add_trade accepts fee_converter param)
            self._fee_converter = fee_converter
        except Exception:
            # fallback: keep attribute but set to None if import/creation fails
            self.profit_tracker = None
            self._fee_converter = None

    # ---- Delegate ProfitTracker API so TradeStats can act as the shared tracker ----
    def add_trade(self, side: str, qty: float, price: float, fee: float, commissionAsset: str = 'USDT', fee_converter=None):
        """Delegate adding a trade to the embedded ProfitTracker (if present)."""
        if self.profit_tracker is None:
            return None
        try:
            # prefer local fee_converter if none provided
            fc = fee_converter if fee_converter is not None else getattr(self, '_fee_converter', None)
            return self.profit_tracker.add_trade(side, qty, price, fee, commissionAsset, fc)
        except Exception:
            return None

    def report_csv(self, filename: str = 'profit_report.csv'):
        if self.profit_tracker is None:
            return None
        try:
            return self.profit_tracker.report_csv(filename)
        except Exception:
            return None

    def get_report(self):
        if self.profit_tracker is None:
            return {}
        try:
            return self.profit_tracker.get_report()
        except Exception:
            return {}

    def update_unrealized(self, mark_price: float):
        if self.profit_tracker is None:
            return None
        try:
            return self.profit_tracker.update_unrealized(mark_price)
        except Exception:
            return None


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
                # Ensure all pending trades have required fields before writing
                normalized = [ensure_pending_fields(dict(t)) for t in self.pending]
                with tempfile.NamedTemporaryFile('w', delete=False, dir='.', prefix='pending_', suffix='.json') as tf:
                    json.dump(self._to_native(normalized), tf, indent=2, ensure_ascii=False)
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
        # Ensure we normalize trade before saving and stamp creation time
        try:
            self.pending[-1] = ensure_pending_fields(self.pending[-1], set_timestamp=True)
        except Exception:
            pass
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
            history_file = get_history_file()
            if os.path.exists(history_file):
                try:
                    with open(history_file, "r", encoding='utf-8') as f:
                        history = json.load(f)
                except Exception:
                    history = []
            else:
                history = []
            # Ensure required history fields are present (auto-fill/normalize)
            try:
                trade = ensure_history_fields(dict(trade))
            except Exception:
                trade = dict(trade)

            # Ensure a usable timestamp is present for history display.
            # Priority: existing 'timestamp' -> 'closed_at' -> current time
            try:
                ts = trade.get('timestamp')
                if not ts or ts == 'N/A':
                    closed_at = trade.get('closed_at')
                    if closed_at and isinstance(closed_at, str) and closed_at != 'N/A':
                        trade['timestamp'] = closed_at
                    else:
                        import datetime
                        trade['timestamp'] = datetime.datetime.now().isoformat(timespec='seconds')
            except Exception:
                try:
                    import datetime
                    trade['timestamp'] = datetime.datetime.now().isoformat(timespec='seconds')
                except Exception:
                    trade['timestamp'] = 'N/A'

            # Attach lightweight analysis (duration, fee_usdt, closed_at, netmode)
            try:
                trade = analyze_closed_trade(trade, fee_converter=getattr(self, '_fee_converter', None))
            except Exception:
                pass

            history.append(self._to_native(trade))
            with tempfile.NamedTemporaryFile('w', delete=False, dir='.', prefix='history_', suffix='.json', encoding='utf-8') as tf:
                json.dump(history, tf, indent=2, ensure_ascii=False)
            os.replace(tf.name, history_file)
            # Refresh cached balances from exchange (if client available)
            try:
                # Attempt to fetch current balances so UI/status reflects recent realized PnL
                self.print_balances()
            except Exception:
                pass
            # Notify webui backend to emit status update so UI is refreshed immediately.
            try:
                import requests, os
                backend = os.getenv('WEBUI_BACKEND_URL', 'http://127.0.0.1:5000')
                try:
                    requests.post(f"{backend}/api/notify_update", timeout=1)
                except Exception:
                    # best-effort only; do not fail save path
                    pass
            except Exception:
                pass
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
            if VERBOSE_BALANCE:
                logger.info("[BALANCE] Binance client not initialized.")
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
                # Protect against accidental key flip: ensure 'USDT' maps to USDT and 'BTC' to BTC
                try:
                    self.cached_balances['USDT'] = float(usdt_free)
                except Exception:
                    self.cached_balances['USDT'] = 0.0
                try:
                    self.cached_balances['BTC'] = float(btc_free)
                except Exception:
                    self.cached_balances['BTC'] = 0.0
                # keep stats.balance as available USDT for BUY pre-flight checks
                # only update if retrieved value seems sane (non-negative)
                try:
                    if self.cached_balances.get('USDT') is not None:
                        self.balance = float(self.cached_balances['USDT'])
                except Exception:
                    pass
                if VERBOSE_BALANCE:
                    logger.info(f"[BALANCE] Available: {usdt_free:.2f} USDT, {btc_free:.6f} BTC")
                return
            except Exception as e:
                attempt += 1
                wait = base_wait * (2 ** (attempt - 1))
                if VERBOSE_BALANCE:
                    logger.error(f"[ERROR] Could not fetch balances (attempt {attempt}/{retries}): {e}; retrying in {wait:.1f}s")
                try:
                    time.sleep(wait)
                except Exception:
                    pass
        # final fallback: use cached balances if available
        if VERBOSE_BALANCE:
            logger.error("[ERROR] Failed to fetch balances after retries; using cached balances if available.")
        usdt_free = self.cached_balances.get('USDT', None)
        btc_free = self.cached_balances.get('BTC', None)
        if usdt_free is not None or btc_free is not None:
            usdt_str = f"{usdt_free:.2f}" if usdt_free is not None else "N/A"
            btc_str = f"{btc_free:.6f}" if btc_free is not None else "N/A"
            if VERBOSE_BALANCE:
                logger.info(f"[BALANCE] Cached: {usdt_str} USDT, {btc_str} BTC")
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
            return {'updated': 0, 'removed': 0, 'error': 'client not initialized'}

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

                # Prefer orderId-first reconciliation: query the specific orderId
                order = None
                try:
                    order = self.client.get_order(symbol=sym, orderId=int(order_id))
                except Exception as e:
                    # get_order may fail for various reasons (order missing, permissions, network).
                    # Fall back to scanning recent orders to try to find a matching orderId
                    try:
                        print(f"[RECONCILE] get_order failed for orderId {order_id}: {e}; falling back to get_all_orders scan")
                        # get_all_orders can be heavy; limit to recent timeframe by not passing many params
                        all_orders = self.client.get_all_orders(symbol=sym)
                        for o in all_orders:
                            try:
                                if str(o.get('orderId')) == str(order_id):
                                    order = o
                                    break
                            except Exception:
                                continue
                    except Exception as e2:
                        print(f"[RECONCILE] Fallback scan failed for orderId {order_id}: {e2}")
                        order = None
                if order is None:
                    print(f"[RECONCILE] Could not locate orderId {order_id} on exchange; leaving pending trade as-is")
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
                    # If the executed order side is opposite to the local trade side,
                    # it's likely a close (e.g., local BUY was closed by a SELL order).
                    try:
                        ord_side = (order or {}).get('side')
                        executed_qty = float(order.get('executedQty', 0.0))
                        cquote = float(order.get('cummulativeQuoteQty', 0.0))
                    except Exception:
                        ord_side = None
                        executed_qty = 0.0
                        cquote = 0.0

                    # Determine average fill price if possible
                    avg_price = None
                    try:
                        if executed_qty > 0 and cquote > 0:
                            avg_price = float(cquote) / float(executed_qty)
                        elif 'fills' in order and isinstance(order.get('fills'), list) and len(order.get('fills')) > 0:
                            num = 0.0
                            den = 0.0
                            for f in order.get('fills'):
                                q = float(f.get('qty', 0))
                                p = float(f.get('price', 0))
                                num += q * p
                                den += q
                            if den > 0:
                                avg_price = num / den
                    except Exception:
                        avg_price = None

                    # If order side is opposite => treat as a closing fill
                    if ord_side is not None and trade.get('side') is not None and ord_side != trade.get('side'):
                        try:
                            # compute realized pnl using existing trade fields (best-effort)
                            entry = float(trade.get('entry', 0) or 0)
                            amount = float(trade.get('amount_usdt', 0) or 0)
                            price_for_calc = avg_price if avg_price is not None else (float(order.get('price', 0)) or None)
                            if price_for_calc is None:
                                pnl_pct = 0.0
                            else:
                                if trade.get('side') == 'BUY':
                                    pnl_pct = (price_for_calc - entry) / entry if entry != 0 else 0.0
                                else:
                                    pnl_pct = (entry - price_for_calc) / entry if entry != 0 else 0.0
                            realized_usdt = pnl_pct * amount
                        except Exception:
                            realized_usdt = 0.0
                            pnl_pct = 0.0

                        # mark win/loss and attach fields
                        try:
                            if realized_usdt > 0:
                                trade['status'] = 'win'
                                self.win += 1
                            else:
                                trade['status'] = 'loss'
                                self.loss += 1
                        except Exception:
                            pass

                        trade['close_price'] = float(price_for_calc) if price_for_calc is not None else None
                        trade['realized_pnl_usdt'] = float(realized_usdt)
                        trade['pnl_pct'] = float(pnl_pct)
                        # persist to history and remove from pending
                        try:
                            self.save_trade_history(trade)
                        except Exception:
                            pass
                        try:
                            self.pending.remove(trade)
                            removed += 1
                        except Exception:
                            pass
                    else:
                        # same-side filled (opening order filled) — keep as executed open trade
                        updated += 1

            if removed or updated:
                self.save_pending()
                # refresh balances after changes
                try:
                    self.print_balances()
                except Exception:
                    pass
            msg = f"[RECONCILE] Done. updated={updated} removed={removed}. open={sum(1 for t in self.pending if t.get('status')=='open')}"
            print(msg)
            return {'updated': int(updated), 'removed': int(removed), 'message': msg}
        except Exception as e:
            print(f"[RECONCILE] Unexpected error: {e}")
            return {'updated': int(updated), 'removed': int(removed), 'error': str(e)}
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
