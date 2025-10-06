import math
from src import adaptive


def test_dynamic_threshold_basic():
    base = 0.8
    model_prob = 0.85
    # no recent data -> should return a float between ADAPT_MIN_THRESH and ADAPT_MAX_THRESH
    t = adaptive.dynamic_threshold(base, model_prob, recent_win_rate=None, volatility=None)
    assert isinstance(t, float)
    assert 0.0 < t <= 1.0


def test_dynamic_threshold_rewards_winrate():
    base = 0.8
    model_prob = 0.82
    # high recent win rate should slightly lower the threshold
    t_high = adaptive.dynamic_threshold(base, model_prob, recent_win_rate=0.9, volatility=None)
    t_low = adaptive.dynamic_threshold(base, model_prob, recent_win_rate=0.1, volatility=None)
    assert t_high <= base
    assert t_low >= base or math.isclose(t_low, base)


def test_dynamic_threshold_penalizes_volatility():
    base = 0.75
    model_prob = 0.78
    t_low_vol = adaptive.dynamic_threshold(base, model_prob, recent_win_rate=None, volatility=0.001)
    t_high_vol = adaptive.dynamic_threshold(base, model_prob, recent_win_rate=None, volatility=0.05)
    assert t_high_vol >= t_low_vol


def test_compute_position_size_defaults_and_caps():
    bal = 1000.0
    # default base frac 0.02 => 20 USDT
    s = adaptive.compute_position_size(bal, target_risk_usdt=None, volatility=None, max_pct_of_balance=0.2)
    assert s > 0
    assert s <= bal * 0.2


def test_compute_position_size_with_target_risk():
    bal = 500.0
    s = adaptive.compute_position_size(bal, target_risk_usdt=50.0, volatility=None, max_pct_of_balance=0.1)
    # cap 10% of 500 = 50, so target 50 should be returned
    assert s == 50.0
