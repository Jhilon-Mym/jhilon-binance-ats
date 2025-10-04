
# bot.py — REST preload only + no-None strategy + graceful shutdown
import os, time, signal, logging
from dotenv import load_dotenv
load_dotenv()
from src.config import Config
import sys

# Ensure project root is on sys.path so `import src.*` works when running
# this file directly (python src/bot.py). This mirrors setting PYTHONPATH.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from typing import List, Dict, Optional
from binance.client import Client
import pandas as pd
from decimal import Decimal, ROUND_DOWN


from src.strategy import apply_strategy, predict_signal
from src.utils import get_profit_tracker, get_pending_file

# populated at startup after fetching symbol info from Binance
SYMBOL_INFO: Optional[dict] = None
from src.websocket_bot import WSRunner, preload_history

# ---- minimal utils inlined if not present ----
def load_env():
    # simple .env loader
    env_path = os.path.join(os.getcwd(), '.env')
    if os.path.exists(env_path):
        for line in open(env_path, 'r', encoding='utf-8'):
            line=line.strip()
            if not line or line.startswith('#') or '=' not in line: 
                continue
            k,v = line.split('=',1)
            os.environ.setdefault(k.strip(), v.strip())

def place_order_live(client, side, symbol, quantity, order_type="MARKET"):
    return client.create_order(symbol=symbol, side=side, type=order_type, quantity=quantity)

def reconcile_pending_with_exchange(client):
    # placeholder; extend if you track pending orders locally
    return

logging.basicConfig(level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

HISTORY: List[Dict] = []
SHUTTING_DOWN = False

def get_balances(client: Client) -> Dict[str, float]:
    acct = client.get_account()
    return {b["asset"]: float(b["free"]) for b in acct["balances"]}


def _parse_env_float(key: str, default: float) -> float:
    """Read environment variable and parse to float while stripping inline comments and whitespace.

    Accepts strings like '0.75  # comment' or '0.5; extra' and returns the numeric part.
    Falls back to `default` on any parse error.
    """
    raw = os.getenv(key, None)
    if raw is None:
        return float(default)
    try:
        # strip inline comments after '#' or ';'
        cleaned = str(raw).split('#', 1)[0].split(';', 1)[0].strip()
        return float(cleaned)
    except Exception:
        try:
            return float(default)
        except Exception:
            return 0.0


def _quantize_qty(symbol_info: dict, qty: float) -> Optional[str]:
    """Quantize `qty` according to symbol_info LOT_SIZE filter.

    Returns a decimal string suitable for Binance `quantity` param, or None
    if qty is below minQty or symbol_info missing required filters.
    """
    if not symbol_info:
        return None
    try:
        filters = {f['filterType']: f for f in symbol_info.get('filters', [])}
        lot = filters.get('LOT_SIZE') or filters.get('LOT') or {}
        minQty = Decimal(str(lot.get('minQty', '0')))
        stepSize = Decimal(str(lot.get('stepSize', '1')))

        qty_dec = Decimal(str(qty))
        if stepSize == 0:
            return None
        # floor to nearest multiple of stepSize
        multiple = (qty_dec // stepSize) * stepSize
        if multiple < minQty or multiple == Decimal('0'):
            return None
        # format with the same number of decimal places as stepSize
        # determine decimal places from stepSize exponent
        exponent = -stepSize.as_tuple().exponent
        places = max(0, exponent)
        # ensure we round down to comply with exchange
        quantized = multiple.quantize(stepSize, rounding=ROUND_DOWN)
        s = f"{quantized:.{places}f}"
        return s
    except Exception:
        return None

def execute_trade(client: Client, signal_out: Dict, symbol: str, last_price: float):
    from src.utils import get_profit_tracker
    import logging
    # Prepare trade parameters
    side = signal_out.get("side")
    if side is None:
        logger.info(f"[SKIP] {signal_out.get('reason')}")
        if signal_out.get('reason','').startswith('indicators_reject') or signal_out.get('reason','').startswith('low_conf'):
            logging.warning(f"[TRADE] Trade rejected: {signal_out}")
        return None

    reason = signal_out.get("reason", "")
    win_prob = float(signal_out.get("win_prob", 0.0))
    combined_score = signal_out.get("combined_score")
    w_ai = signal_out.get("w_ai")
    tuned_thresh = signal_out.get("tuned_thresh")

    ai_min = _parse_env_float("AI_MIN_CONFIDENCE_OVERRIDE", 0.75)
    win_prob_min = _parse_env_float("WIN_PROB_MIN", 0.8)
    min_combined = _parse_env_float("MIN_COMBINED_SCORE", 0.70)
    min_ai_weight = _parse_env_float("MIN_AI_WEIGHT", 0.20)
    live_trades = os.getenv("LIVE_TRADES", "false").lower() == "true"
    buy_usdt = _parse_env_float("BUY_USDT_PER_TRADE", 20.0)
    symbol_base = os.getenv("SYMBOL_BASE", "BTC")

    # Safety checks (same as before)
    if live_trades:
        if win_prob < float(win_prob_min):
            logger.info(f"[SKIP-LIVE] Blocking live trade: win_prob {win_prob:.3f} < WIN_PROB_MIN {win_prob_min}")
            return None
        safety_ok = False
        if reason == "ai_confident" and win_prob >= ai_min:
            safety_ok = True
        if combined_score is not None:
            try:
                if float(combined_score) >= float(min_combined):
                    if w_ai is None or float(w_ai) >= float(min_ai_weight):
                        safety_ok = True
            except Exception:
                safety_ok = False
        if not safety_ok:
            logger.info(f"[SKIP-LIVE] Blocking live trade: reason={reason} win_prob={win_prob:.3f} combined_score={combined_score} w_ai={w_ai} min_combined={min_combined} min_ai_weight={min_ai_weight}")
            return None

    bals = get_balances(client)
    # use the shared, netmode-aware profit tracker so UI and bot see the same cached balances
    trade_stats = get_profit_tracker()
    # BUY
    if side == "BUY":
        if bals.get("USDT", 0.0) < buy_usdt:
            logger.info(f"[SKIP] BUY: insufficient USDT {bals.get('USDT',0.0):.2f} < {buy_usdt}")
            return None
        qty = round(buy_usdt / float(last_price), 8)
        qty_str = format(qty, '.8f').rstrip('0').rstrip('.')
        if qty_str == '' or float(qty_str) == 0.0:
            logger.info(f"[SKIP] BUY: computed qty is zero for last_price={last_price}")
            return None
        order_info = None
        if live_trades:
            q = _quantize_qty(SYMBOL_INFO, float(qty_str)) if SYMBOL_INFO is not None else qty_str
            if q is None:
                logger.info(f"[SKIP] BUY: qty {qty_str} doesn't meet symbol LOT_SIZE/minQty rules")
                return None
            order_info = place_order_live(client, "BUY", symbol, q, order_type="MARKET")
            logger.info(f"[TRADE] BUY {q} {symbol_base} @ ~{last_price} | reason={reason} win_prob={win_prob:.3f} combined_score={combined_score}")
        else:
            logger.info(f"[PAPER] BUY {qty_str} {symbol_base} @ ~{last_price} | reason={reason} win_prob={win_prob:.3f} combined_score={combined_score}")
        trade = trade_stats.open_trade(signal_out, last_price, qty, buy_usdt, order_info=order_info)
        # Refresh live balance after trade execution (updates shared cached_balances)
        try:
            trade_stats.print_balances()
        except Exception:
            pass
        return trade
    # SELL
    elif side == "SELL":
        logging.info(f"[TRADE] Attempting SELL: {signal_out}")
        base_bal = bals.get(symbol_base, 0.0)
        if base_bal <= 0:
            logger.info(f"[SKIP] SELL: no {symbol_base} balance")
            return None
        # Prefer closing tracked pending BUY trades rather than selling the
        # entire wallet unexpectedly. Use the shared TradeStats pending list
        # to determine how much qty we should sell to close managed positions.
        try:
            tracker = get_profit_tracker()
            # sum qty of open BUY pending trades we manage
            tracked_qty = sum(float(t.get('qty', 0)) for t in getattr(tracker, 'pending', []) if t.get('status') == 'open' and t.get('side') == 'BUY')
        except Exception:
            tracked_qty = 0.0

        if tracked_qty and tracked_qty > 0:
            # sell only the tracked pending qty (bounded by available balance)
            sell_qty = min(float(base_bal), float(tracked_qty))
        else:
            # fallback: sell entire available base balance
            sell_qty = float(base_bal)

        qty = round(sell_qty, 8)
        # amount_usdt should reflect notional (qty * last_price)
        amount_usdt_val = float(qty) * float(last_price)
        qty_str = format(qty, '.8f').rstrip('0').rstrip('.')
        if qty_str == '' or float(qty_str) == 0.0:
            logger.info(f"[SKIP] SELL: computed qty is zero for base_bal={base_bal}")
            return None
        order_info = None
        if live_trades:
            q = _quantize_qty(SYMBOL_INFO, float(qty_str)) if SYMBOL_INFO is not None else qty_str
            if q is None:
                logger.info(f"[SKIP] SELL: qty {qty_str} doesn't meet symbol LOT_SIZE/minQty rules")
                return None
            order_info = place_order_live(client, "SELL", symbol, q, order_type="MARKET")
            logger.info(f"[TRADE] SELL {q} {symbol_base} @ ~{last_price} | reason={reason} win_prob={win_prob:.3f} combined_score={combined_score}")
        else:
            logger.info(f"[PAPER] SELL {qty_str} {symbol_base} @ ~{last_price} | reason={reason} win_prob={win_prob:.3f} combined_score={combined_score}")
            trade = trade_stats.open_trade(signal_out, last_price, qty, amount_usdt_val, order_info=order_info)
            logging.info(f"[TRADE] SELL executed: {trade}")
            # Refresh live balance after trade execution (updates shared cached_balances)
            try:
                trade_stats.print_balances()
            except Exception:
                pass
            return trade

def run():
    global HISTORY, SHUTTING_DOWN
    load_env()
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    use_testnet = os.getenv("USE_TESTNET", "true").lower() == "true"
    symbol = os.getenv("SYMBOL", "BTCUSDT")
    interval = os.getenv("INTERVAL", "5m")
    # Read preload limit from dynamic Config so UI/.env updates take effect
    from src.config import Config
    try:
        preload_limit = int(Config.HISTORY_PRELOAD)
    except Exception:
        preload_limit = int(os.getenv("HISTORY_PRELOAD", "600"))
    os.environ.setdefault("SYMBOL_BASE", symbol.replace("USDT", ""))

    try:
        client = Client(api_key, api_secret, testnet=use_testnet)
    except Exception as e:
        import traceback
        logger.error(f"Failed to create client: {e}")
        try:
            from src.utils import get_data_file
            with open(get_data_file('bot_run.log'), 'a', encoding='utf-8') as logf:
                logf.write(f"Failed to create client: {e}\n")
                logf.write(traceback.format_exc())
        except Exception:
            pass
        raise
    # fetch symbol_info once at startup to support LOT_SIZE quantization
    global SYMBOL_INFO
    try:
        info = client.get_symbol_info(symbol)
        SYMBOL_INFO = info or None
        logger.info(f"[INFO] Loaded symbol info for {symbol}: LOT_SIZE step={SYMBOL_INFO.get('filters',[])}")
    except Exception as e:
        SYMBOL_INFO = None
        logger.warning(f"[WARN] Failed to fetch symbol info for {symbol}: {e}")

    netmode = os.getenv('USE_TESTNET', 'true')
    api_key = os.getenv('BINANCE_API_KEY', '')
    api_secret = os.getenv('BINANCE_API_SECRET', '')
    logger.info(f"Bot started | netmode={netmode} | API_KEY={api_key} | API_SECRET={api_secret}")

    bals = get_balances(client)
    logger.info(f"[BALANCE] USDT={bals.get('USDT',0):.2f} | {symbol.replace('USDT','')}={bals.get(symbol.replace('USDT',''),0):.6f}")

    reconcile_pending_with_exchange(client)

    # REST preload only
    df = preload_history(client, symbol, interval, limit=preload_limit)
    if df is None or df.empty:
        raise RuntimeError("REST preload failed — no candles fetched. Check API keys/network/symbol.")
    HISTORY = df.to_dict("records")
    logger.info(f"[INFO] Seeded history with {len(HISTORY)} candles from REST")

    def _shutdown(signum, frame):
        nonlocal ws_runner
        globals()['SHUTTING_DOWN'] = True
        try:
            ws_runner.stop()
        except Exception:
            pass
        logger.info("[SHUTDOWN] Stopping bot...")

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    def on_candle_closed(candle: Dict):
        global HISTORY
        HISTORY.append(candle)
        if len(HISTORY) > 3000:
            HISTORY = HISTORY[-3000:]
        logger.info(f"[CANDLES] {len(HISTORY)} (last close={candle['close']})")
        try:
            # build a DataFrame view for the AI predictor (predict_signal expects a DataFrame)
            try:
                df_hist = pd.DataFrame(HISTORY)
            except Exception:
                df_hist = None

            ai_signal = None
            try:
                if df_hist is not None:
                    ai_signal = predict_signal(df_hist)
            except Exception as e:
                logger.error(f"[ERROR] predict_signal: {e}")

            from src.config import Config
            out = apply_strategy(HISTORY, ai_signal=ai_signal, threshold=Config.SIGNAL_THRESHOLD)
        except Exception as e:
            logger.error(f"[ERROR] apply_strategy: {e}")
            return
        logger.info(f"[SIGNAL] {out}")
        try:
            trade = execute_trade(client, out, symbol, float(candle["close"]))
            if trade and trade.get("side") == "SELL":
                logger.info(f"[DEBUG] SELL trade created and stored: {trade}")
        except Exception as e:
            logger.error(f"[ERROR] execute_trade: {e}")

        # --- Auto-close trades on every new candle ---
        try:
            from src.utils import TradeStats
            trade_stats = TradeStats()
            closed = trade_stats.update_pending(candle)
            for t in closed:
                if t.get("side") == "SELL":
                    logger.info(f"[DEBUG] SELL trade closed and persisted: {t}")
        except Exception as e:
            logger.error(f"[ERROR] update_pending: {e}")

    try:
        ws_runner = WSRunner(client, symbol, interval, on_candle_closed=on_candle_closed)
        ws_runner.start()
    except Exception as e:
        import traceback
        logger.error(f"[ERROR] WSRunner/WebSocket: {e}")
        try:
            from src.utils import get_data_file
            with open(get_data_file('bot_run.log'), 'a', encoding='utf-8') as logf:
                logf.write(f"[ERROR] WSRunner/WebSocket: {e}\n")
                logf.write(traceback.format_exc())
        except Exception:
            pass
        raise

    try:
        while not globals()['SHUTTING_DOWN']:
            time.sleep(0.5)
    finally:
        try:
            ws_runner.stop()
        except Exception:
            pass
        logger.info("[DONE] Bot stopped.")

if __name__ == "__main__":
    print("Bot started", flush=True)
    try:
        run()
    except Exception as e:
        import traceback
        try:
            from src.utils import get_data_file
            with open(get_data_file('bot_run.log'), 'a', encoding='utf-8') as logf:
                logf.write(f"[ERROR] {e}\n")
                logf.write(traceback.format_exc())
        except Exception:
            pass
        print(f"[ERROR] {e}", flush=True)
