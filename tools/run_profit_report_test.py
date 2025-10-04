import sys, traceback, tempfile, os
sys.path.insert(0, r'D:\binance_ats_clone\jhilon-binance-ats-1')
from src.utils import get_profit_tracker

def main():
    try:
        t = get_profit_tracker()
        print('Tracker type=', type(t))
        print('has profit_tracker=', hasattr(t, 'profit_tracker'))
        print('has symbol=', hasattr(t, 'symbol'))
        tempdir = tempfile.gettempdir()
        p = os.path.join(tempdir, 'profit_test_direct.csv')
        print('reporting to', p)
        t.report_csv(p)
        exists = os.path.exists(p)
        print('file exists=', exists)
        if exists:
            print('size=', os.path.getsize(p))
        print('done')
    except Exception as e:
        traceback.print_exc()

if __name__ == '__main__':
    main()
