import csv
import datetime
import threading
from collections import deque
from typing import List, Dict, Optional

class ProfitTracker:
    """
    Advanced profit/loss tracker for trading bots.
    Features:
    - FIFO matching for realized P/L
    - Handles commission fees and converts commissionAsset to USDT
    - Tracks realized and unrealized P/L
    - Maintains equity curve
    - Generates CSV reports and plots equity curve
    """
    def __init__(self, symbol: str = 'BTCUSDT'):
        self.symbol = symbol
        self.lock = threading.Lock()
        self.inventory = deque()  # FIFO inventory: [{'side', 'qty', 'price', 'fee'}]
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.total_fees = 0.0
        self.equity_curve: List[Dict] = []
        self.history: List[Dict] = []

    def add_trade(self, side: str, qty: float, price: float, fee: float, commissionAsset: str = 'USDT', fee_converter=None):
        """
        Add a trade to the tracker. Converts fee to USDT if needed.
        fee_converter: function to convert fee from commissionAsset to USDT
        """
        if commissionAsset != 'USDT' and fee_converter:
            fee = fee_converter(fee, commissionAsset)
        trade = {'side': side, 'qty': qty, 'price': price, 'fee': fee, 'dt': datetime.datetime.now()}
        with self.lock:
            self.total_fees += fee
            if side == 'BUY':
                self.inventory.append(trade)
            elif side == 'SELL':
                realized = self._match_fifo(qty, price, fee)
                self.realized_pnl += realized
            self._update_equity(price)
            self.history.append(trade)

    def _match_fifo(self, sell_qty: float, sell_price: float, sell_fee: float) -> float:
        """
        Match SELL trades to BUY inventory using FIFO and calculate realized P/L.
        """
        realized = 0.0
        qty_left = sell_qty
        while qty_left > 0 and self.inventory:
            buy = self.inventory[0]
            match_qty = min(qty_left, buy['qty'])
            pnl = (sell_price - buy['price']) * match_qty
            realized += pnl
            realized -= (buy['fee'] * match_qty / buy['qty'] + sell_fee * match_qty / sell_qty)
            buy['qty'] -= match_qty
            qty_left -= match_qty
            if buy['qty'] <= 1e-8:
                self.inventory.popleft()
        return realized

    def update_unrealized(self, mark_price: float):
        """
        Update unrealized P/L for open positions using latest market price.
        """
        unrealized = 0.0
        for buy in self.inventory:
            unrealized += (mark_price - buy['price']) * buy['qty']
            unrealized -= buy['fee']
        self.unrealized_pnl = unrealized
        self._update_equity(mark_price)

    def _update_equity(self, mark_price: float):
        """
        Update equity curve with current realized/unrealized P/L and mark price.
        """
        equity = self.realized_pnl + self.unrealized_pnl - self.total_fees
        self.equity_curve.append({'dt': datetime.datetime.now(), 'equity': equity, 'mark_price': mark_price})

    def report_csv(self, filename: str = 'profit_report.csv'):
        """
        Save detailed trade and P/L report to CSV file.
        """
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['dt', 'side', 'qty', 'price', 'fee'])
            for t in self.history:
                writer.writerow([t['dt'], t['side'], t['qty'], t['price'], t['fee']])
            writer.writerow([])
            writer.writerow(['realized_pnl', self.realized_pnl])
            writer.writerow(['unrealized_pnl', self.unrealized_pnl])
            writer.writerow(['total_fees', self.total_fees])

    def plot_equity(self):
        """
        Plot the equity curve using matplotlib.
        """
        try:
            import matplotlib.pyplot as plt
            times = [e['dt'] for e in self.equity_curve]
            equities = [e['equity'] for e in self.equity_curve]
            plt.plot(times, equities)
            plt.title('Equity Curve')
            plt.xlabel('Time')
            plt.ylabel('Equity (USDT)')
            plt.show()
        except ImportError:
            print('Please install matplotlib: pip install matplotlib')

    def get_report(self) -> Dict:
        """
        Return a summary report as a dictionary.
        """
        return {
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'total_fees': self.total_fees,
            'equity_curve': self.equity_curve,
            'open_inventory': list(self.inventory)
        }
