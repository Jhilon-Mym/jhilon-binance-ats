from src.trade_manager import ManagedTrade
from src.config import Config
mt = ManagedTrade(side='BUY', entry=100.0, sl=90.0, tp=120.0)
print('initial:', mt)
print('cfg.MIN_PROFIT_TO_CLOSE=', Config.MIN_PROFIT_TO_CLOSE)
print('call update with price=120 (tp) atr=10)')
res = mt.update(120.0, 10.0, Config)
print('result:', res)
print('trail_active:', mt.trail_active, 'trail_stop:', mt.trail_stop)
