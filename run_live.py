import time
import json
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta
from src.strategy import TradingStrategy

# -----------------------------
# Leggi config
# -----------------------------
with open("config.json", "r") as f:
    config = json.load(f)

broker = config["broker"]
trading = config["trading"]
strategy_params = config.get("strategy_params", {})

ACCOUNT = broker["login"]
PASSWORD = broker["password"]
SERVER = broker["server_demo"] if broker["mode"] == "demo" else broker["server_real"]
SYMBOL = trading["symbol"]
TIMEFRAME_MAP = {"M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15}
TIMEFRAME = TIMEFRAME_MAP.get(trading["timeframe"], mt5.TIMEFRAME_M5)
LOT_SIZE = 0.1  # puoi calcolare da risk_per_trade_pct se vuoi
DELAY = trading.get("poll_seconds", 10)

# -----------------------------
# Inizializza MetaTrader5
# -----------------------------
if not mt5.initialize(login=ACCOUNT, password=PASSWORD, server=SERVER):
    print("Errore inizializzazione MT5:", mt5.last_error())
    exit()

print(f"MT5 inizializzato ({broker['mode']})!")

# -----------------------------
# Funzione per leggere OHLCV live
# -----------------------------
def get_ohlcv(symbol, timeframe, n=100):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    if rates is None:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

# -----------------------------
# Funzione per inviare ordine
# -----------------------------
def send_order(symbol, action, lot):
    if action not in ["BUY", "SELL"]:
        return False

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print("Errore: impossibile ottenere tick")
        return False

    order_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL
    price = tick.ask if action == "BUY" else tick.bid

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "deviation": trading["max_slippage_points"],
        "magic": trading["magic_number"],
        "comment": "Python bot",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print("Errore ordine:", result)
        return False
    print(f"Ordine eseguito: {action} {lot} {symbol}")
    return True

# -----------------------------
# Funzione countdown candela
# -----------------------------
def seconds_to_next_candle(timeframe: str):
    now = datetime.utcnow()
    if timeframe.startswith('M'):
        minutes = int(timeframe[1:])
        delta = timedelta(minutes=minutes)
        candle_start = now.replace(second=0, microsecond=0) - timedelta(minutes=now.minute % minutes)
    elif timeframe.startswith('H'):
        hours = int(timeframe[1:])
        delta = timedelta(hours=hours)
        candle_start = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=now.hour % hours)
    else:
        raise ValueError("Timeframe non supportato")
    candle_end = candle_start + delta
    remaining = candle_end - now
    return remaining

# -----------------------------
# Loop principale live
# -----------------------------
while True:
    df = get_ohlcv(SYMBOL, TIMEFRAME)
    if df is None or df.empty:
        print("Errore dati OHLCV")
        time.sleep(DELAY)
        continue

    strat = TradingStrategy(SYMBOL, TIMEFRAME, df, params=strategy_params)
    strat.generate_signals()

    signal = strat.signals.iloc[-1]['signal'] if not strat.signals.empty else None

    if signal in ["BUY", "SELL"]:
        send_order(SYMBOL, signal, LOT_SIZE)
    else:
        remaining = seconds_to_next_candle(trading["timeframe"])
        mins, secs = divmod(int(remaining.total_seconds()), 60)
        print(f"Nessun segnale. Candela chiuderà tra {mins}m {secs}s.")

    time.sleep(DELAY)
