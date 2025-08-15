import argparse
import pandas as pd
from src.strategy import TradingStrategy

def run_backtest(csv_path: str, symbol: str, timeframe: str, **kwargs):
    df = pd.read_csv(csv_path)
    # normalizza colonne richieste
    lower = {c.lower(): c for c in df.columns}
    required = ['time','open','high','low','close','volume']
    for col in required:
        if col not in [c.lower() for c in df.columns]:
            raise ValueError(f"CSV deve contenere la colonna {col}")
    df.columns = [c.lower() for c in df.columns]
    strat = TradingStrategy(symbol, timeframe, df, params=kwargs or {})
    strat.generate_signals()
    res = strat.backtest(initial_balance=10000.0)
    return res

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, help='Percorso CSV OHLCV')
    ap.add_argument('--symbol', required=True)
    ap.add_argument('--timeframe', required=True)
    args = ap.parse_args()
    res = run_backtest(args.csv, args.symbol, args.timeframe)
    import json
    print(json.dumps(res, indent=2))
