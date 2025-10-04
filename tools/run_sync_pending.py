#!/usr/bin/env python3
import os
import sys
BASE_DIR = r"D:\binance_ats_clone\obaidur-binance-ats-main"
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.utils import TradeStats

def main():
    s = TradeStats()
    changed = s.sync_pending_with_orders()
    print('sync_pending_with_orders changed:', changed)
    print('pending count:', len(s.pending))

if __name__ == '__main__':
    main()
