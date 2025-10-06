"""Adaptive trading helpers

Functions here compute dynamic win-prob thresholds and position sizing based
on recent performance and market volatility. These are conservative defaults
designed to increase execution frequency when the model is doing well and to
reduce exposure when volatility or recent losses increase.

This is intentionally small and well-tested logic (no network calls). Callers
should treat outputs as suggestions and still perform final safety checks
before placing live orders.
"""
from typing import Optional
import os
import math

def get_env_float(name: str, default: float) -> float:
    try:
        v = os.getenv(name)
        if v is None:
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def dynamic_threshold(base_threshold: float,
                      model_win_prob: float,
                      recent_win_rate: Optional[float] = None,
                      volatility: Optional[float] = None) -> float:
    """Return an adjusted win-prob threshold to decide whether to execute a trade.

    - base_threshold: configured baseline threshold (e.g., 0.8)
    - model_win_prob: probability from model (0..1)
    - recent_win_rate: empirical win-rate from recent closed trades (0..1)
    - volatility: recent market volatility metric (higher means more volatile)

    Behavior:
    - If recent_win_rate is high (> base), lower threshold modestly to allow more trades.
    - If volatility is high, increase threshold (be more selective).
    - Never return threshold below a configured minimum (ENV ADAPT_MIN_THRESH)
      or above configured max (ADAPT_MAX_THRESH).
    """
    MIN_THRESH = get_env_float('ADAPT_MIN_THRESH', 0.55)
    MAX_THRESH = get_env_float('ADAPT_MAX_THRESH', 0.95)

    # start with base
    t = float(base_threshold)

    # reward good recent performance: if recent_win_rate provided, nudge threshold down
    if recent_win_rate is not None:
        # factor in difference between recent and baseline (clamp)
        try:
            diff = float(recent_win_rate) - max(0.0, float(base_threshold) - 0.1)
            # if recent_win_rate > baseline, decrease threshold by up to 10%
            adj = -0.10 * max(0.0, diff)
            t = t * (1.0 + adj)
        except Exception:
            pass

    # penalize high volatility: increase threshold proportional to volatility
    if volatility is not None:
        try:
            # interpret volatility as e.g., stddev of returns (0..inf). scale nicely
            vol = float(volatility)
            # mapping: vol 0.0 -> 0 adj; vol 0.01 -> +2% ; vol 0.02 -> +4% etc (conservative)
            vol_adj = min(0.25, vol * 2.0)
            t = t * (1.0 + vol_adj)
        except Exception:
            pass

    # If model's win prob is much larger than threshold, be willing to relax
    try:
        if model_win_prob - t > 0.10:
            # model very confident, lower required threshold a bit to allow execution
            t = t * 0.9
    except Exception:
        pass

    # clamp
    try:
        t = max(MIN_THRESH, min(MAX_THRESH, float(t)))
    except Exception:
        t = float(base_threshold)

    return float(t)


def compute_position_size(balance_usdt: float,
                          target_risk_usdt: Optional[float] = None,
                          volatility: Optional[float] = None,
                          max_pct_of_balance: float = 0.2) -> float:
    """Return suggested notional USDT size for a trade.

    - balance_usdt: current available USDT balance (float)
    - target_risk_usdt: absolute USDT amount you are willing to risk per trade
    - volatility: recent volatility; higher volatility => reduce size
    - max_pct_of_balance: upper cap as fraction of balance

    Priority: if target_risk_usdt provided, use it (bounded by max_pct_of_balance*balance).
    Otherwise derive from balance and volatility.
    """
    try:
        bal = float(balance_usdt or 0.0)
    except Exception:
        return 0.0

    if bal <= 0:
        return 0.0

    cap = max_pct_of_balance * bal

    if target_risk_usdt is not None:
        try:
            tr = float(target_risk_usdt)
            return max(0.0, min(cap, tr))
        except Exception:
            pass

    # default heuristic: risk a small fraction of balance scaled inversely with volatility
    base_frac = get_env_float('ADAPT_BASE_FRAC', 0.02)  # 2% of balance
    frac = base_frac
    if volatility is not None:
        try:
            vol = float(volatility)
            # higher vol -> reduce fraction; map vol 0->1, 0.02->0.5, 0.05->0.2
            frac = base_frac * max(0.25, 1.0 / (1.0 + vol * 50.0))
        except Exception:
            pass

    size = bal * frac
    # clamp to cap
    return max(0.0, min(cap, float(size)))
