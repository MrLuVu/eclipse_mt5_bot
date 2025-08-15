from typing import Optional, List
from loguru import logger
import MetaTrader5 as MT5
import pandas as pd
from datetime import datetime, timezone

TIMEFRAMES = {
    "M1": MT5.TIMEFRAME_M1,
    "M5": MT5.TIMEFRAME_M5,
    "M15": MT5.TIMEFRAME_M15,
    "M30": MT5.TIMEFRAME_M30,
    "H1": MT5.TIMEFRAME_H1,
    "H4": MT5.TIMEFRAME_H4,
    "D1": MT5.TIMEFRAME_D1,
}

class MT5Client:
    def __init__(self, login: int, password: str, server: str, path_terminal: str = ""):
        self.login = login
        self.password = password
        self.server = server
        self.path_terminal = path_terminal
        self.connected = False

    def connect(self) -> bool:
        if not MT5.initialize(path=self.path_terminal or None):
            logger.error(f"MT5 initialize failed: {MT5.last_error()}" )
            return False
        authorized = MT5.login(self.login, password=self.password, server=self.server)
        if not authorized:
            logger.error(f"MT5 login failed: {MT5.last_error()}" )
            return False
        self.connected = True
        logger.info("Connected to MT5")
        return True

    def shutdown(self):
        MT5.shutdown()
        self.connected = False

    def ensure_symbol(self, symbol: str) -> bool:
        info = MT5.symbol_info(symbol)
        if info is None:
            logger.error(f"Symbol not found: {symbol}")
            return False
        if not info.visible:
            MT5.symbol_select(symbol, True)
        return True

    def fetch_ohlc(self, symbol: str, timeframe: str, bars: int = 500) -> pd.DataFrame:
        tf = TIMEFRAMES[timeframe]
        rates = MT5.copy_rates_from_pos(symbol, tf, 0, bars)
        if rates is None:
            raise RuntimeError("copy_rates_from_pos returned None")
        df = pd.DataFrame(rates)
        df.rename(columns={'real_volume':'volume'}, inplace=True)
        df['time'] = pd.to_datetime(df['time'], unit='s').dt.tz_localize('UTC').dt.tz_convert('Europe/Rome')
        return df[['time','open','high','low','close','volume']]

    def positions_by_symbol(self, symbol: str):
        return [p for p in MT5.positions_get(symbol=symbol) or []]

    def pending_orders_by_symbol(self, symbol: str):
        return [o for o in MT5.orders_get(symbol=symbol) or []]

    def send_order(self, symbol: str, order_type, volume: float, price: float, sl: float, tp: float, deviation: int, magic: int, comment: str = ""):
        request = {
            "action": MT5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": deviation,
            "magic": magic,
            "comment": comment,
            "type_filling": MT5.ORDER_FILLING_IOC
        }
        result = MT5.order_send(request)
        if result is None:
            logger.error("order_send returned None")
            return None
        if result.retcode != MT5.TRADE_RETCODE_DONE:
            logger.error(f"ORDER FAILED: {result.retcode} | {result.comment}")
        else:
            logger.info(f"ORDER OK: ticket={result.order} deal={result.deal}")
        return result
