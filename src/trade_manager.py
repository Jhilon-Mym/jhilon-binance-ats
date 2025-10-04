

from dataclasses import dataclass
from typing import Optional

@dataclass
class ManagedTrade:
    side: str
    entry: float
    sl: float
    tp: float
    trail_active: bool = False
    trail_stop: Optional[float] = None

    def update(self, price: float, atr: float, cfg) -> Optional[str]:
        """
        Trailing stop and profit-only close logic:
        - Trailing activates after favorable move (TRAIL_ACTIVATE_ATR)
        - Trailing stop distance: TRAIL_OFFSET_ATR
        - Only closes if MIN_PROFIT_TO_CLOSE is met (covers fees)
        - SL ignored unless in profit
        """
        if self.side == "BUY":
            # Activate trailing after favorable move
            if not self.trail_active and price >= self.entry + cfg.TRAIL_ACTIVATE_ATR * atr:
                self.trail_active = True
                self.trail_stop = max(self.entry, price - cfg.TRAIL_OFFSET_ATR * atr)
            # Trail upwards
            if self.trail_active:
                new_trail = price - cfg.TRAIL_OFFSET_ATR * atr
                if new_trail > (self.trail_stop or -1e9):
                    self.trail_stop = new_trail
            # Only close if in profit (covers fees)
            # consider equal-to threshold as in-profit (so exact TP that meets min profit closes)
            in_profit = price >= self.entry * (1 + cfg.MIN_PROFIT_TO_CLOSE)
            # TP or trailing stop hit (and in profit)
            if in_profit and (price >= self.tp or (self.trail_stop and price <= self.trail_stop)):
                return "close"
        elif self.side == "SELL":
            if not self.trail_active and price <= self.entry - cfg.TRAIL_ACTIVATE_ATR * atr:
                self.trail_active = True
                self.trail_stop = min(self.entry, price + cfg.TRAIL_OFFSET_ATR * atr)
            if self.trail_active:
                new_trail = price + cfg.TRAIL_OFFSET_ATR * atr
                if new_trail < (self.trail_stop or 1e9):
                    self.trail_stop = new_trail
            # consider equal-to threshold as in-profit for sells as well
            in_profit = price <= self.entry * (1 - cfg.MIN_PROFIT_TO_CLOSE)
            if in_profit and (price <= self.tp or (self.trail_stop and price >= self.trail_stop)):
                return "close"
        # Never close at SL unless in profit
        return None
