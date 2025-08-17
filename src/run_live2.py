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

# Mappa dei timeframe supportati da MetaTrader5
TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1
}

# Timeframe principale per l\'esecuzione del bot
MAIN_TIMEFRAME_STR = trading["timeframe"] # Usiamo la stringa direttamente
MAIN_TIMEFRAME_MT5 = TIMEFRAME_MAP.get(MAIN_TIMEFRAME_STR, mt5.TIMEFRAME_M1)

# Timeframe aggiuntivi per l\'analisi multi-timeframe (HTF e LTF)
# Questi dovrebbero essere configurabili nel config.json
# Per ora, li hardcodiamo per dimostrazione
ADDITIONAL_TIMEFRAMES_STR = [
    "H4", # Esempio di Higher Timeframe
    "M5"  # Esempio di Lower Timeframe
]

# Filtra i timeframe validi e rimuovi duplicati
ALL_TIMEFRAMES_STR = list(set([MAIN_TIMEFRAME_STR] + [tf for tf in ADDITIONAL_TIMEFRAMES_STR if tf is not None]))

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
# Funzione per leggere OHLCV live per più timeframe
# -----------------------------
def get_ohlcv_multi_timeframe(symbol, timeframes_str: list, n=200):
    data = {}
    for tf_name in timeframes_str:
        tf_mt5 = TIMEFRAME_MAP.get(tf_name)
        if tf_mt5 is None:
            print(f"Errore: Timeframe {tf_name} non riconosciuto.")
            continue

        rates = mt5.copy_rates_from_pos(symbol, tf_mt5, 0, n)
        if rates is None:
            print(f"Errore nel recupero dati OHLCV per {symbol} {tf_name}: {mt5.last_error()}")
            data[tf_name] = pd.DataFrame() # Ritorna un DataFrame vuoto in caso di errore
            continue
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit='s')
        data[tf_name] = df
    return data

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
    if timeframe.startswith('M'):
        minutes = int(timeframe[1:])
        delta = timedelta(minutes=minutes)
        # Calcola l\'inizio della candela corrente
        candle_start = now.replace(second=0, microsecond=0) - timedelta(minutes=now.minute % minutes)
    elif timeframe.startswith('H'):
        hours = int(timeframe[1:])
        delta = timedelta(hours=hours)
        # Calcola l\'inizio della candela corrente
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
        # Recupera i dati OHLCV per tutti i timeframe necessari
        ohlcv_data = get_ohlcv_multi_timeframe(SYMBOL, ALL_TIMEFRAMES_STR)

        # Assicurati che il timeframe principale abbia dati validi
        if MAIN_TIMEFRAME_STR not in ohlcv_data or ohlcv_data[MAIN_TIMEFRAME_STR].empty:
            print(f"Errore: Dati OHLCV non disponibili per il timeframe principale {MAIN_TIMEFRAME_STR}. Riprovo...")
            time.sleep(DELAY)
            continue

        # Inizializza la strategia con i parametri e i dati OHLCV per tutti i timeframe
        strat = TradingStrategy(SYMBOL, MAIN_TIMEFRAME_STR, ohlcv_data, params=strategy_params)
        strat.generate_signals()
        
        # Recupera il segnale dall\'ultima riga del DataFrame dei segnali
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

