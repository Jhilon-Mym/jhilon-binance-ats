#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import TradeStats

def main():
    s = TradeStats()
    changed = s.sync_pending_with_orders()
    print('sync_pending_with_orders changed:', changed)
    print('pending count:', len(s.pending))

if __name__ == '__main__':
    main()
