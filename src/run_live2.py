import time
import json
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta, timezone
from src.strategy2 import TradingStrategy

# -----------------------------
# Leggi config
# -----------------------------
with open("config/config.json", "r") as f:
    config = json.load(f)

broker = config["broker"]
trading = config["trading"]
strategy_params = config.get("strategy_params", {})

ACCOUNT = broker["login"]
PASSWORD = broker["password"]
SERVER = broker["server_demo"] if broker["mode"] == "demo" else broker["server_real"]
SYMBOL = trading["symbol"]
TIMEFRAME_MAP = {"M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15, "H4": mt5.TIMEFRAME_H4}
TIMEFRAME = TIMEFRAME_MAP.get(trading["timeframe"], mt5.TIMEFRAME_M1)

# Calcolo del lot size basato sul rischio per trade in percentuale
# Questo è un esempio semplificato, in un bot reale servirebbe:
# 1. Recuperare il saldo del conto (mt5.account_info().balance)
# 2. Calcolare il valore di 1 pip per il simbolo (mt5.symbol_info(SYMBOL)._point * mt5.symbol_info(SYMBOL).trade_tick_value)
# 3. Determinare lo stop loss in pips dalla strategia
# 4. Calcolare il lot size effettivo
# Per ora, manteniamo un lot size fisso o usiamo un placeholder
LOT_SIZE = 0.1 # Placeholder, da calcolare dinamicamente in base a risk_per_trade_pct e SL

DELAY = trading.get("poll_seconds", 10)

# -----------------------------
# Inizializza MetaTrader5
# -----------------------------
if not mt5.initialize(login=ACCOUNT, password=PASSWORD, server=SERVER):
    print("Errore inizializzazione MT5:", mt5.last_error())
    exit()

print(f"MT5 inizializzato ({broker["mode"]})!")

# -----------------------------
# Funzione per leggere OHLCV live
# -----------------------------
def get_ohlcv(symbol, timeframe, n=200):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    if rates is None:
        print(f"Errore nel recupero dati OHLCV per {symbol} {timeframe}: {mt5.last_error()}")
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit=\'s\')
    return df

# -----------------------------
# Funzione per inviare ordine
# -----------------------------
def send_order(symbol, action, lot, stop_loss=0.0, take_profit=0.0):
    if action not in ["BUY", "SELL"]:
        print("Azione non valida.")
        return False

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print("Errore: impossibile ottenere tick per il simbolo.")
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

    if stop_loss > 0:
        request["sl"] = stop_loss
    if take_profit > 0:
        request["tp"] = take_profit

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Errore ordine: {result.comment} ({result.retcode})")
        return False
    print(f"Ordine eseguito: {action} {lot} {symbol} @ {price} (SL: {stop_loss}, TP: {take_profit})")
    return True

# -----------------------------
# Funzione countdown candela
# -----------------------------
def seconds_to_next_candle(timeframe: str):
    now = datetime.now(timezone.utc)
    if timeframe.startswith(\'M\'):
        minutes = int(timeframe[1:])
        delta = timedelta(minutes=minutes)
        # Calcola l'inizio della candela corrente
        candle_start = now.replace(second=0, microsecond=0) - timedelta(minutes=now.minute % minutes)
    elif timeframe.startswith(\'H\'):
        hours = int(timeframe[1:])
        delta = timedelta(hours=hours)
        # Calcola l'inizio della candela corrente
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
    try:
        df = get_ohlcv(SYMBOL, TIMEFRAME)
        if df is None or df.empty:
            print("Errore dati OHLCV o DataFrame vuoto. Riprovo...")
            time.sleep(DELAY)
            continue

        # Inizializza la strategia con i parametri e i dati OHLCV
        strat = TradingStrategy(SYMBOL, TIMEFRAME, df, params=strategy_params)
        strat.generate_signals()
        
        # Recupera il segnale dall'ultima riga del DataFrame dei segnali
        signal_data = strat.signals.iloc[-1] if not strat.signals.empty else None
        signal = signal_data["signal"] if signal_data is not None else None
        stop_loss = signal_data["stop_loss"] if signal_data is not None and "stop_loss" in signal_data else 0.0
        take_profit = signal_data["take_profit"] if signal_data is not None and "take_profit" in signal_data else 0.0

        if signal in ["BUY", "SELL"]:
            send_order(SYMBOL, signal, LOT_SIZE, stop_loss, take_profit)
        else:
            remaining = seconds_to_next_candle(trading["timeframe"])
            mins, secs = divmod(int(remaining.total_seconds()), 60)
            print(f"Nessun segnale. Candela chiuderà tra {mins}m {secs}s.")

    except Exception as e:
        print(f"Errore nel loop principale: {e}")

    time.sleep(DELAY)
