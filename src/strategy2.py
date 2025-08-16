# =============================================================================
# STRATEGY2.PY - Strategia Eclipse per Eclipse MT5 Bot
# =============================================================================

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime


# -----------------------------------------------------------------------------
# CLASSI DATI
# -----------------------------------------------------------------------------
@dataclass
class Candela:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float


@dataclass
class POI:
    tipo: str
    direzione: str
    candela_di_riferimento: Candela
    prezzo_di_attivazione_top: float
    prezzo_di_attivazione_bottom: float
    key_level_ohlc: dict
    e_mitigato: bool = False
    timeframe: str = ""


@dataclass
class RangeMercato:
    strong_high: Candela
    strong_low: Candela
    weak_highs: List[Candela]
    weak_lows: List[Candela]
    liquidita_esterna_buy_side: float
    liquidita_esterna_sell_side: float
    liquidita_interna: List[POI]
    timeframe: str


@dataclass
class Trade:
    id: str
    coppia: str
    tipo: str  # BUY o SELL
    prezzo_entrata: float
    stop_loss: float
    take_profit_finale: float
    lottaggio: float
    stato: str  # Aperto, Chiuso, Breakeven
    parziali_presi: int = 0


# -----------------------------------------------------------------------------
# FUNZIONI DI SUPPORTO
# -----------------------------------------------------------------------------
def identifica_swing_points(candele: List[Candela]):
    swing_highs, swing_lows = [], []
    for i in range(1, len(candele) - 1):
        prev, curr, nxt = candele[i-1], candele[i], candele[i+1]
        if curr.high > prev.high and curr.high > nxt.high:
            swing_highs.append(curr)
        if curr.low < prev.low and curr.low < nxt.low:
            swing_lows.append(curr)
    print(swing_highs, swing_lows)
    return swing_highs, swing_lows


def analizza_struttura_e_bos(swing_highs, swing_lows):
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return "Indefinito", None, None

    ultimo_high, penultimo_high = swing_highs[-1], swing_highs[-2]
    ultimo_low, penultimo_low = swing_lows[-1], swing_lows[-2]

    trend = "Indefinito"
    ultimo_bos_high, ultimo_bos_low = None, None

    if ultimo_high.high > penultimo_high.high and ultimo_low.low > penultimo_low.low:
        trend = "Bullish"
        ultimo_bos_high = ultimo_high
    elif ultimo_high.high < penultimo_high.high and ultimo_low.low < penultimo_low.low:
        trend = "Bearish"
        ultimo_bos_low = ultimo_low

    return trend, ultimo_bos_high, ultimo_bos_low


def trova_ultimo_low_prima_di(ts, lows):
    candidates = [c for c in lows if c.timestamp < ts]
    return max(candidates, key=lambda c: c.timestamp) if candidates else None


def trova_primo_low_dopo(ts, lows):
    candidates = [c for c in lows if c.timestamp > ts]
    return min(candidates, key=lambda c: c.timestamp) if candidates else None


def trova_ultimo_high_prima_di(ts, highs):
    candidates = [c for c in highs if c.timestamp < ts]
    return max(candidates, key=lambda c: c.timestamp) if candidates else None


def trova_primo_high_dopo(ts, highs):
    candidates = [c for c in highs if c.timestamp > ts]
    return min(candidates, key=lambda c: c.timestamp) if candidates else None


def definisci_range_da_quasimodo(candele: List[Candela], timeframe: str) -> Optional[RangeMercato]:
    swing_highs, swing_lows = identifica_swing_points(candele)
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return None

    strong_high, strong_low = None, None

    # Trova Strong High
    for i in range(len(swing_highs) - 1, 0, -1):
        curr, prev = swing_highs[i], swing_highs[i-1]
        if curr.high > prev.high:
            low_pre = trova_ultimo_low_prima_di(curr.timestamp, swing_lows)
            low_post = trova_primo_low_dopo(curr.timestamp, swing_lows)
            if low_pre and low_post and low_post.low < low_pre.low:
                strong_high = curr
                break

    if not strong_high:
        return None

    # Trova Strong Low (prima dello strong high)
    lows_pre = [l for l in swing_lows if l.timestamp < strong_high.timestamp]
    for i in range(len(lows_pre) - 1, 0, -1):
        curr, prev = lows_pre[i], lows_pre[i-1]
        if curr.low < prev.low:
            high_pre = trova_ultimo_high_prima_di(curr.timestamp, swing_highs)
            high_post = trova_primo_high_dopo(curr.timestamp, swing_highs)
            if high_pre and high_post and high_post.high > high_pre.high:
                strong_low = curr
                break

    if not strong_low:
        return None

    weak_highs = [h for h in swing_highs if strong_low.timestamp < h.timestamp < strong_high.timestamp]
    weak_lows = [l for l in swing_lows if strong_low.timestamp < l.timestamp < strong_high.timestamp]

    return RangeMercato(
        strong_high=strong_high,
        strong_low=strong_low,
        weak_highs=weak_highs,
        weak_lows=weak_lows,
        liquidita_esterna_buy_side=strong_high.high,
        liquidita_esterna_sell_side=strong_low.low,
        liquidita_interna=[],
        timeframe=timeframe
    )


# -----------------------------------------------------------------------------
# STRATEGIA TRADING
# -----------------------------------------------------------------------------
class TradingStrategy:
    def __init__(self, symbol: str, timeframe: str, data: pd.DataFrame, params: dict = {}):
        self.symbol = symbol
        self.timeframe = timeframe
        self.data = data
        self.params = params
        self.signals = pd.DataFrame()

        self.candele = [
            Candela(
                timestamp=row["time"],
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
            )
            for _, row in data.iterrows()
        ]

    def generate_signals(self):
        if len(self.candele) < 50:
            return pd.DataFrame()

        # 1. Identifica swing points e struttura
        swing_highs, swing_lows = identifica_swing_points(self.candele)
        trend, bos_high, bos_low = analizza_struttura_e_bos(swing_highs, swing_lows)
        print(f"[DEBUG] Trend rilevato: {trend}")

        # 2. Definisci range
        range_htf = definisci_range_da_quasimodo(self.candele, self.timeframe)
        if not range_htf:
            print("[DEBUG] Nessun range valido")
            return pd.DataFrame()

        print(f"[DEBUG] Range definito: Low={range_htf.strong_low.low}, High={range_htf.strong_high.high}")

        # 3. Regola di segnale base
        last_price = self.candele[-1].close
        signal = "HOLD"
        if trend == "Bullish" and last_price > range_htf.strong_low.low:
            signal = "BUY"
        elif trend == "Bearish" and last_price < range_htf.strong_high.high:
            signal = "SELL"

        print(f"[DEBUG] Prezzo attuale={last_price}, Segnale={signal}")

        self.signals = pd.DataFrame([{
            "time": self.candele[-1].timestamp,
            "price": last_price,
            "trend": trend,
            "signal": signal
        }])
        return self.signals
