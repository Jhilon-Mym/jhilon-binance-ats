import sys, traceback, tempfile, pathlib, os
sys.path.insert(0, r'D:\binance_ats_clone\jhilon-binance-ats-1')
from src.utils import get_profit_tracker, get_netmode
try:
    from src.config import Config
except Exception:
    Config = None

def main():
    try:
        out_arg = None
        if len(sys.argv) > 1:
            out_arg = sys.argv[1]

        tracker = get_profit_tracker()
        use_testnet = get_netmode()
        net = 'testnet' if use_testnet else 'mainnet'
        symbol = getattr(Config, 'SYMBOL', 'BTCUSDT') if Config else 'BTCUSDT'
        if out_arg:
            tmp_path = pathlib.Path(out_arg)
        else:
            tmp_path = pathlib.Path(tempfile.gettempdir()) / f"profit_report_{net}_{symbol}.csv"
        print('Tracker type=', type(tracker))
        print('Has attr profit_tracker=', hasattr(tracker, 'profit_tracker'))
        print('Will write to', tmp_path)
        tracker.report_csv(str(tmp_path))
        print('Wrote file, exists=', tmp_path.exists(), 'size=', tmp_path.stat().st_size if tmp_path.exists() else 'N/A')
    except Exception:
        traceback.print_exc()

if __name__ == '__main__':
    main()
