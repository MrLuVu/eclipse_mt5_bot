import time
import pandas as pd
from loguru import logger
from typing import Optional
from src.broker.mt5_client import MT5Client, TIMEFRAMES
from src.strategy import TradingStrategy
from src.utils import load_config, rr_size_from_sl

def run_live(config_path: str = 'config/config.json'):
    cfg = load_config(config_path)
    mode = cfg['broker']['mode']
    server = cfg['broker']['server_demo'] if mode == 'demo' else cfg['broker']['server_real']

    client = MT5Client(
        login=int(cfg['broker']['login']),
        password=str(cfg['broker']['password']),
        server=server,
        path_terminal=cfg['broker'].get('path_terminal','')
    )
    assert client.connect(), "MT5 connection failed"

    symbol = cfg['trading']['symbol']
    timeframe = cfg['trading']['timeframe']
    risk_pct = float(cfg['trading']['risk_per_trade_pct'])
    deviation = int(cfg['trading']['max_slippage_points'])
    magic = int(cfg['trading']['magic_number'])
    poll_seconds = int(cfg['trading']['poll_seconds'])
    max_open = int(cfg['trading']['max_open_trades_per_symbol'])

    assert client.ensure_symbol(symbol), f"Symbol {symbol} not available"

    df = client.fetch_ohlc(symbol, timeframe, bars=600)
    strat = TradingStrategy(symbol, timeframe, df, cfg['strategy_params'])
    strat.generate_signals()

    while True:
        time.sleep(poll_seconds)
        df = client.fetch_ohlc(symbol, timeframe, bars=600)
        strat.data = df.reset_index(drop=True)
        signals = strat.generate_signals()
        last = signals.iloc[-1]
        price = float(df['close'].iloc[-1])

        # limite posizioni aperte
        if len(client.positions_by_symbol(symbol)) >= max_open:
            logger.info("Max open trades reached, skip")
            continue

        if last['signal'] in ('BUY','SELL'):
            info = __import__('MetaTrader5').MetaTrader5.symbol_info(symbol)
            size = rr_size_from_sl(info, balance=10000.0, entry=price, stop=float(last['stop_loss']), risk_pct=risk_pct)
            if size <= 0:
                logger.warning("Computed size <= 0, skip order")
                continue
            order_type = __import__('MetaTrader5').MetaTrader5.ORDER_TYPE_BUY if last['signal'] == 'BUY' else __import__('MetaTrader5').MetaTrader5.ORDER_TYPE_SELL
            client.send_order(
                symbol=symbol,
                order_type=order_type,
                volume=size,
                price=price,
                sl=float(last['stop_loss']),
                tp=float(last['take_profit']),
                deviation=deviation,
                magic=magic,
                comment=f"Eclipse-{last['signal']}"
            )
        else:
            logger.info("No signal, HOLD")

if __name__ == "__main__":
    run_live()
