# =============================================================================
# STRATEGY2.PY - Strategia Eclipse per Eclipse MT5 Bot
# =============================================================================
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime, timedelta


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
# FUNZIONI DI SUPPORTO (DAL PSEUDOCODICE)
# -----------------------------------------------------------------------------
def identifica_swing_points(candele: List[Candela]):
    swing_highs, swing_lows = [], []
    for i in range(1, len(candele) - 1):
        prev, curr, nxt = candele[i-1], candele[i], candele[i+1]
        if curr.high > prev.high and curr.high > nxt.high:
            swing_highs.append(curr)
        if curr.low < prev.low and curr.low < nxt.low:
            swing_lows.append(curr)
    return swing_highs, swing_lows


def analizza_struttura_e_bos(swing_highs: List[Candela], swing_lows: List[Candela], trend_precedente: str = "Indefinito"):
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return "Indefinito", None, None

    # Ordina gli swing points per data
    swing_highs.sort(key=lambda c: c.timestamp)
    swing_lows.sort(key=lambda c: c.timestamp)

    ultimo_high = swing_highs[-1]
    penultimo_high = swing_highs[-2]
    ultimo_low = swing_lows[-1]
    penultimo_low = swing_lows[-2]

    trend = "Indefinito"
    ultimo_bos_high = None
    ultimo_bos_low = None

    # Uptrend: Higher Highs (HH) e Higher Lows (HL)
    if ultimo_high.high > penultimo_high.high and ultimo_low.low > penultimo_low.low:
        trend = "Bullish"
        ultimo_bos_high = ultimo_high
    # Downtrend: Lower Highs (LH) e Lower Lows (LL)
    elif ultimo_high.high < penultimo_high.high and ultimo_low.low < penultimo_low.low:
        trend = "Bearish"
        ultimo_bos_low = ultimo_low

    # Inversione da Bullish a Bearish (Change of Character - CHOCH)
    if trend_precedente == "Bullish" and ultimo_low.low < penultimo_low.low:
        trend = "Bearish"
    # Inversione da Bearish a Bullish (Change of Character - CHOCH)
    elif trend_precedente == "Bearish" and ultimo_high.high > penultimo_high.high:
        trend = "Bullish"

    print("Trend, ultimo_bos_high, ultimo_bos_low", trend, ultimo_bos_high, ultimo_bos_low )
    return trend, ultimo_bos_high, ultimo_bos_low


def trova_ultimo_low_prima_di(timestamp, lista_lows):
    candidates = [c for c in lista_lows if c.timestamp < timestamp]
    return max(candidates, key=lambda c: c.timestamp) if candidates else None


def trova_primo_low_dopo(timestamp, lista_lows):
    candidates = [c for c in lista_lows if c.timestamp > timestamp]
    return min(candidates, key=lambda c: c.timestamp) if candidates else None


def trova_ultimo_high_prima_di(timestamp, lista_highs):
    candidates = [c for c in lista_highs if c.timestamp < timestamp]
    return max(candidates, key=lambda c: c.timestamp) if candidates else None


def trova_primo_high_dopo(timestamp, lista_highs):
    candidates = [c for c in lista_highs if c.timestamp > timestamp]
    return min(candidates, key=lambda c: c.timestamp) if candidates else None


def definisci_range_da_quasimodo(candele: List[Candela], timeframe: str) -> Optional[RangeMercato]:
    print(f"Avvio identificazione Range su timeframe {timeframe}...")
    tutti_swing_highs, tutti_swing_lows = identifica_swing_points(candele)

    if len(tutti_swing_highs) < 2 or len(tutti_swing_lows) < 2:
        print("Dati insufficienti (swing points) per definire un range.")
        return None

    strong_high_del_range = None
    for i in range(len(tutti_swing_highs) - 1, 0, -1):
        high_attuale = tutti_swing_highs[i]
        high_precedente = tutti_swing_highs[i-1]

        stop_hunt_eseguito = high_attuale.high > high_precedente.high

        bos_al_ribasso_confermato = False
        low_precedente_allo_stophunt = trova_ultimo_low_prima_di(high_attuale.timestamp, tutti_swing_lows)
        if low_precedente_allo_stophunt:
            low_successivo_allo_stophunt = trova_primo_low_dopo(high_attuale.timestamp, tutti_swing_lows)
            if low_successivo_allo_stophunt and (low_successivo_allo_stophunt.low < low_precedente_allo_stophunt.low):
                bos_al_ribasso_confermato = True

        if stop_hunt_eseguito and bos_al_ribasso_confermato:
            strong_high_del_range = high_attuale
            print(f"QM Ribassista (Strong High) trovato a: {strong_high_del_range.high} in data {strong_high_del_range.timestamp}")
            break

    if strong_high_del_range is None:
        print("Nessun QM Ribassista (Strong High) valido trovato.")
        return None

    strong_low_del_range = None
    swing_lows_precedenti = [low for low in tutti_swing_lows if low.timestamp < strong_high_del_range.timestamp]
    for i in range(len(swing_lows_precedenti) - 1, 0, -1):
        low_attuale = swing_lows_precedenti[i]
        low_precedente = swing_lows_precedenti[i-1]

        stop_hunt_eseguito = low_attuale.low < low_precedente.low

        bos_al_rialzo_confermato = False
        high_precedente_allo_stophunt = trova_ultimo_high_prima_di(low_attuale.timestamp, tutti_swing_highs)
        if high_precedente_allo_stophunt:
            high_successivo_allo_stophunt = trova_primo_high_dopo(low_attuale.timestamp, tutti_swing_highs)
            if high_successivo_allo_stophunt and (high_successivo_allo_stophunt.high > high_precedente_allo_stophunt.high):
                bos_al_rialzo_confermato = True

        if stop_hunt_eseguito and bos_al_rialzo_confermato:
            strong_low_del_range = low_attuale
            print(f"QM Rialzista (Strong Low) trovato a: {strong_low_del_range.low} in data {strong_low_del_range.timestamp}")
            break

    if strong_low_del_range is None:
        print("Nessun QM Rialzista (Strong Low) valido trovato prima dello Strong High.")
        return None

    nuovo_range = RangeMercato(
        strong_high=strong_high_del_range,
        strong_low=strong_low_del_range,
        weak_highs=[],  # Verranno popolati dopo
        weak_lows=[],   # Verranno popolati dopo
        liquidita_esterna_buy_side=strong_high_del_range.high,
        liquidita_esterna_sell_side=strong_low_del_range.low,
        liquidita_interna=[],
        timeframe=timeframe
    )

    nuovo_range.weak_highs = [
        h for h in tutti_swing_highs
        if strong_low_del_range.timestamp < h.timestamp < strong_high_del_range.timestamp
    ]
    nuovo_range.weak_lows = [
        l for l in tutti_swing_lows
        if strong_low_del_range.timestamp < l.timestamp < strong_high_del_range.timestamp
    ]

    print(f"Range valido definito su {timeframe}: Low a {nuovo_range.strong_low.low}, High a {nuovo_range.strong_high.high}")
    return nuovo_range


"""def identifica_tutti_poi(candele: List[Candela], timeframe: str) -> List[POI]:
    print("Identifica_Poi timeframe: ", timeframe, "\nCandele: ", len(candele))
    lista_poi = []
    for i in range(1, len(candele) - 1):
        candela_prec = candele[i-1]
        candela_curr = candele[i]
        candela_succ = candele[i+1]

        # 1. Order Block Ribassista
        if candela_prec.close > candela_prec.open and \
           candela_curr.close < candela_curr.open and \
           candela_curr.low < candela_prec.low:
            # Definire prezzo_di_attivazione_top e bottom per OB
            poi_top = candela_prec.high
            poi_bottom = candela_prec.low
            lista_poi.append(POI(tipo="Orderblock", direzione="Bearish", candela_di_riferimento=candela_prec,
                                 prezzo_di_attivazione_top=poi_top, prezzo_di_attivazione_bottom=poi_bottom,
                                 key_level_ohlc={'open': candela_prec.open, 'high': candela_prec.high, 'low': candela_prec.low, 'close': candela_prec.close},
                                 timeframe=timeframe))

        # 2. Order Block Rialzista
        if candela_prec.close < candela_prec.open and \
           candela_curr.close > candela_curr.open and \
           candela_curr.high > candela_prec.high:
            # Definire prezzo_di_attivazione_top e bottom per OB
            poi_top = candela_prec.high
            poi_bottom = candela_prec.low
            lista_poi.append(POI(tipo="Orderblock", direzione="Bullish", candela_di_riferimento=candela_prec,
                                 prezzo_di_attivazione_top=poi_top, prezzo_di_attivazione_bottom=poi_bottom,
                                 key_level_ohlc={'open': candela_prec.open, 'high': candela_prec.high, 'low': candela_prec.low, 'close': candela_prec.close},
                                 timeframe=timeframe))

        # 3. Inefficienza / Fair Value Gap (FVG) Rialzista
        if candela_prec.high < candela_succ.low:
            poi_top = candela_succ.low
            poi_bottom = candela_prec.high
            lista_poi.append(POI(tipo="Inefficiency", direzione="Bullish", candela_di_riferimento=candela_curr,
                                 prezzo_di_attivazione_top=poi_top, prezzo_di_attivazione_bottom=poi_bottom,
                                 key_level_ohlc={}, timeframe=timeframe))

        # 4. Inefficienza / Fair Value Gap (FVG) Ribassista
        if candela_prec.low > candela_succ.high:
            poi_top = candela_prec.low
            poi_bottom = candela_succ.high
            lista_poi.append(POI(tipo="Inefficiency", direzione="Bearish", candela_di_riferimento=candela_curr,
                                 prezzo_di_attivazione_top=poi_top, prezzo_di_attivazione_bottom=poi_bottom,
                                 key_level_ohlc={}, timeframe=timeframe))

        # TODO: Implementare logica per Breaker, Hidden Base, Wick
    return lista_poi"""
def identifica_tutti_poi(candele: List[Candela], timeframe: str) -> List[POI]:
    print("Identifica_Poi timeframe:", timeframe, " | Candele:", len(candele))
    lista_poi = []

    for i in range(1, len(candele) - 1):
        candela_prec = candele[i-1]   # candela precedente
        candela_curr = candele[i]     # candela centrale (candela di riferimento)
        candela_succ = candele[i+1]   # candela successiva

        # -------------------------------
        # 1. ORDER BLOCK RIBASSISTA (ultima candela verde prima del dump rosso)
        # -------------------------------
        if candela_prec.close > candela_prec.open and candela_curr.close < candela_curr.open:
            # Controllo se la candela successiva conferma la spinta ribassista
            if candela_succ.low < candela_prec.low:
                poi_top = candela_prec.open   # meglio usare corpo
                poi_bottom = candela_prec.low
                lista_poi.append(POI(
                    tipo="Orderblock",
                    direzione="Bearish",
                    candela_di_riferimento=candela_prec,
                    prezzo_di_attivazione_top=poi_top,
                    prezzo_di_attivazione_bottom=poi_bottom,
                    key_level_ohlc={'open': candela_prec.open, 'high': candela_prec.high,
                                    'low': candela_prec.low, 'close': candela_prec.close},
                    timeframe=timeframe
                ))

        # -------------------------------
        # 2. ORDER BLOCK RIALZISTA (ultima candela rossa prima del pump verde)
        # -------------------------------
        if candela_prec.close < candela_prec.open and candela_curr.close > candela_curr.open:
            # Controllo se la candela successiva conferma la spinta rialzista
            if candela_succ.high > candela_prec.high:
                poi_top = candela_prec.high
                poi_bottom = candela_prec.open   # corpo
                lista_poi.append(POI(
                    tipo="Orderblock",
                    direzione="Bullish",
                    candela_di_riferimento=candela_prec,
                    prezzo_di_attivazione_top=poi_top,
                    prezzo_di_attivazione_bottom=poi_bottom,
                    key_level_ohlc={'open': candela_prec.open, 'high': candela_prec.high,
                                    'low': candela_prec.low, 'close': candela_prec.close},
                    timeframe=timeframe
                ))

        # -------------------------------
        # 3. FAIR VALUE GAP (FVG) RIALZISTA
        # condizione: candela_prec.high < candela_succ.low
        # e la candela centrale non ha coperto quel gap
        # -------------------------------
        if candela_prec.high < candela_succ.low and candela_curr.low > candela_prec.high:
            poi_top = candela_succ.low
            poi_bottom = candela_prec.high
            lista_poi.append(POI(
                tipo="Inefficiency",
                direzione="Bullish",
                candela_di_riferimento=candela_curr,
                prezzo_di_attivazione_top=poi_top,
                prezzo_di_attivazione_bottom=poi_bottom,
                key_level_ohlc={},
                timeframe=timeframe
            ))

        # -------------------------------
        # 4. FAIR VALUE GAP (FVG) RIBASSISTA
        # condizione: candela_prec.low > candela_succ.high
        # e la candela centrale non ha coperto il gap
        # -------------------------------
        if candela_prec.low > candela_succ.high and candela_curr.high < candela_prec.low:
            poi_top = candela_prec.low
            poi_bottom = candela_succ.high
            lista_poi.append(POI(
                tipo="Inefficiency",
                direzione="Bearish",
                candela_di_riferimento=candela_curr,
                prezzo_di_attivazione_top=poi_top,
                prezzo_di_attivazione_bottom=poi_bottom,
                key_level_ohlc={},
                timeframe=timeframe
            ))

    # Debug finale
    print(f"[DEBUG] Identificati {len(lista_poi)} POI su timeframe {timeframe}")
    for poi in lista_poi[:5]:
        print(f"  {poi.tipo} {poi.direzione} | Top={poi.prezzo_di_attivazione_top} | Bottom={poi.prezzo_di_attivazione_bottom} | Ref={poi.candela_di_riferimento.timestamp}")

    return lista_poi


"""def filtra_poi_validi(lista_poi: List[POI], swing_points_high: List[Candela], swing_points_low: List[Candela], candele: List[Candela]) -> List[POI]:
    poi_validi = []
    for poi in lista_poi:
        has_taken_liquidity = False
        is_mitigated = False
        # REGOLA 2: Deve aver "preso" liquidità.
        if poi.direzione == "Bearish": # OB ribassista deve aver rotto un massimo precedente
            for swing_high in swing_points_high:
                if poi.candela_di_riferimento.high > swing_high.high and poi.candela_di_riferimento.timestamp > swing_high.timestamp:
                    has_taken_liquidity = True
                    break
        elif poi.direzione == "Bullish": # OB rialzista deve aver rotto un minimo precedente
            for swing_low in swing_points_low:
                if poi.candela_di_riferimento.low < swing_low.low and poi.candela_di_riferimento.timestamp > swing_low.timestamp:
                    has_taken_liquidity = True
                    break

        # REGOLA 3: Non deve essere mitigato.
        # Controlla se il prezzo è già tornato in quella zona DOPO la formazione del POI.
        candele_successive = [c for c in candele if c.timestamp > poi.candela_di_riferimento.timestamp]
        for c_succ in candele_successive:
            if poi.direzione == "Bearish": # Per POI ribassisti, il prezzo è tornato sopra il bottom
                if c_succ.high >= poi.prezzo_di_attivazione_bottom and c_succ.low <= poi.prezzo_di_attivazione_top:
                    is_mitigated = True
                    break
            elif poi.direzione == "Bullish": # Per POI rialzisti, il prezzo è tornato sotto il top
                if c_succ.low <= poi.prezzo_di_attivazione_top and c_succ.high >= poi.prezzo_di_attivazione_bottom:
                    is_mitigated = True
                    break

        if has_taken_liquidity and not is_mitigated:
            poi_validi.append(poi)

    print(len(poi_validi))
    return poi_validi"""
def filtra_poi_validi(lista_poi: List[POI], swing_points_high: List[Candela], swing_points_low: List[Candela], candele: List[Candela]) -> List[POI]:
    poi_validi = []

    for poi in lista_poi:
        has_taken_liquidity = False
        is_mitigated = False

        # -------------------------------
        # (a) Deve aver preso liquidità
        # -------------------------------
        if poi.direzione == "Bearish":  # OB ribassista deve rompere un massimo precedente
            for swing_high in swing_points_high:
                future_candele = [c for c in candele if c.timestamp >= poi.candela_di_riferimento.timestamp]
                if any(c.high > swing_high.high for c in future_candele):
                    has_taken_liquidity = True
                    break

        elif poi.direzione == "Bullish":  # OB rialzista deve rompere un minimo precedente
            for swing_low in swing_points_low:
                future_candele = [c for c in candele if c.timestamp >= poi.candela_di_riferimento.timestamp]
                if any(c.low < swing_low.low for c in future_candele):
                    has_taken_liquidity = True
                    break

        # -------------------------------
        # (b) Non deve essere mitigato
        # -------------------------------
        candele_successive = [c for c in candele if c.timestamp > poi.candela_di_riferimento.timestamp]
        for c_succ in candele_successive:
            if poi.direzione == "Bearish":
                # Consideriamo chiusura dentro la zona come mitigazione
                if c_succ.close >= poi.prezzo_di_attivazione_bottom and c_succ.close <= poi.prezzo_di_attivazione_top:
                    is_mitigated = True
                    break

            elif poi.direzione == "Bullish":
                if c_succ.close <= poi.prezzo_di_attivazione_top and c_succ.close >= poi.prezzo_di_attivazione_bottom:
                    is_mitigated = True
                    break

        # -------------------------------
        # (c) Regole finali + debug
        # -------------------------------
        if has_taken_liquidity and not is_mitigated:
            poi_validi.append(poi)
            print(f"[VALIDO] {poi.direzione} {poi.candela_di_riferimento.timestamp} POI a {poi.prezzo_di_attivazione_top:.5f} | Liquidity: OK | Mitigazione: NO")
        else:
            print(f"[SCARTATO] {poi.direzione} POI a {poi.prezzo_di_attivazione_top:.5f} | Liquidity: {has_taken_liquidity} | Mitigazione: {is_mitigated}")

    print(f"[DEBUG] POI validi trovati: {len(poi_validi)}")
    return poi_validi



"""# Funzioni placeholder per l\'interfaccia con il broker
def get_candele(coppia: str, timeframe: str, numero_candele: int) -> List[Candela]:
    # Questa è una funzione placeholder. In un bot reale, qui si interfaccierebbe con l\'API del broker.
    # Per ora, restituiamo candele fittizie per test.
    print(f"[MOCK] Richiesta {numero_candele} candele per {coppia} su {timeframe}")
    # Genera dati di esempio per simulare le candele
    candele_mock = []
    base_time = datetime.now() - timedelta(minutes=numero_candele * 15) # Esempio per 15M
    for i in range(numero_candele):
        timestamp = base_time + timedelta(minutes=i * 15)
        open_price = 1.0 + i * 0.0001 + np.random.rand() * 0.0005
        close_price = open_price + (np.random.rand() - 0.5) * 0.001
        high_price = max(open_price, close_price) + np.random.rand() * 0.0002
        low_price = min(open_price, close_price) - np.random.rand() * 0.0002
        candele_mock.append(Candela(timestamp, open_price, high_price, low_price, close_price))
    return candele_mock"""
def get_candele(coppia: str, timeframe, numero_candele: int) -> List[Candela]:
    """Ottiene candele reali da MetaTrader5 e le restituisce come lista di oggetti Candela"""
    # Recupero dati reali da MT5
    map = {
        "M1" : mt5.TIMEFRAME_M1,
        "H1" : mt5.TIMEFRAME_H1,
        "4H" : mt5.TIMEFRAME_H4,
    }
    for k, v in map.items():
        print(k, timeframe)
        if k == timeframe:
            timeframe = v
            print(timeframe)
            break
    rates = mt5.copy_rates_from_pos(coppia, timeframe, 0, numero_candele)
    if rates is None:
        print(f"[ERRORE] Nessun dato ricevuto da MT5 per {coppia} su {timeframe}")
        return []

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")

    # Converte in lista di oggetti Candela
    candele = []
    for _, row in df.iterrows():
        c = Candela(
            timestamp=row["time"],
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"]
        )
        candele.append(c)

    return candele

def place_order(trade_details: dict) -> str:
    print(f"[MOCK] Piazzamento ordine: {trade_details}")
    return "ORDER_ID_MOCK_123"

def modify_sl(order_id: str, new_sl: float):
    print(f"[MOCK] Modifica SL per {order_id} a {new_sl}")

def close_partial_position(order_id: str, percentage_to_close: float):
    print(f"[MOCK] Chiusura parziale {percentage_to_close}% per {order_id}")

def get_prezzo_corrente(coppia: str) -> float:
    print(f"[MOCK] Richiesta prezzo corrente per {coppia}")
    return 1.0500 + np.random.rand() * 0.001 # Prezzo fittizio

def trova_poi_corrispondente_su_ltf(poi_htf: POI, candele_entry_tf: List[Candela]) -> Optional[POI]:
    # Logica per trovare un POI su un timeframe inferiore che corrisponda al POI HTF
    # Per semplicità, qui restituiamo un POI fittizio se il timestamp è vicino
    for c in candele_entry_tf:
        if abs((c.timestamp - poi_htf.candela_di_riferimento.timestamp).total_seconds()) < 3600: # Entro 1 ora
            return POI(tipo="Orderblock", direzione=poi_htf.direzione, candela_di_riferimento=c,
                       prezzo_di_attivazione_top=poi_htf.prezzo_di_attivazione_top * 0.99,
                       prezzo_di_attivazione_bottom=poi_htf.prezzo_di_attivazione_bottom * 1.01,
                       key_level_ohlc={}, timeframe="1M")
    return None

def trova_target_logico(poi_htf: POI, trend_htf: str) -> float:
    # Logica per definire un take profit logico
    if trend_htf == "Bullish":
        return poi_htf.prezzo_di_attivazione_top * 1.02 # Esempio
    else:
        return poi_htf.prezzo_di_attivazione_bottom * 0.98 # Esempio

def trova_poi_del_bos_ltf(range_ltf: RangeMercato, candele_ltf: List[Candela]) -> Optional[POI]:
    # Questa funzione dovrebbe identificare il POI che ha causato il BOS sul LTF
    # Per ora, restituiamo un POI fittizio basato sullo strong_high/low del range LTF
    if range_ltf.strong_high:
        return POI(tipo="Orderblock", direzione="Bearish", candela_di_riferimento=range_ltf.strong_high,
                   prezzo_di_attivazione_top=range_ltf.strong_high.high, prezzo_di_attivazione_bottom=range_ltf.strong_high.low,
                   key_level_ohlc={}, timeframe=range_ltf.timeframe)
    elif range_ltf.strong_low:
        return POI(tipo="Orderblock", direzione="Bullish", candela_di_riferimento=range_ltf.strong_low,
                   prezzo_di_attivazione_top=range_ltf.strong_low.high, prezzo_di_attivazione_bottom=range_ltf.strong_low.low,
                   key_level_ohlc={}, timeframe=range_ltf.timeframe)
    return None


# -----------------------------------------------------------------------------
# CONFIGURAZIONE GLOBALE (dal pseudocodice)
# -----------------------------------------------------------------------------
COPPIA_DI_VALUTE = "BTC/USD"
TIMEFRAME_HTF = "4H"
TIMEFRAME_LTF = "15M"
TIMEFRAME_ENTRY = "1M"
RISCHIO_PER_TRADE_PERCENTUALE = 1.0
CAPITALE_CONTO = 10000.0
SPREAD = 0.0001 # Esempio di spread


# -----------------------------------------------------------------------------
# STRATEGIA TRADING (Classe principale)
# -----------------------------------------------------------------------------
class TradingStrategy:
    def __init__(self, symbol: str, timeframe: str, data: pd.DataFrame, params: dict = {}):
        self.symbol = symbol
        self.timeframe = timeframe
        self.data = data
        self.params = params
        self.signals = pd.DataFrame()
        self.active_trade: Optional[Trade] = None # Per gestire un solo trade alla volta
        self.previous_trend_htf = "Indefinito"

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

    def calculate_lot_size(self, stop_loss_pips: float) -> float:
        # Calcola la dimensione del lotto in base al rischio per trade
        # 1 pip = 0.0001 per EUR/USD
        valore_pip = 10.0 # Per EUR/USD, 1 pip per lotto standard (100,000 unità) vale 10 USD
        rischio_denaro = CAPITALE_CONTO * (RISCHIO_PER_TRADE_PERCENTUALE / 100)
        lottaggio = rischio_denaro / (stop_loss_pips * valore_pip)
        return lottaggio

    def generate_signals(self):
        if len(self.candele) < 100: # Necessarie abbastanza candele per l\'analisi
            print("[DEBUG] Dati insufficienti per generare segnali.")
            return pd.DataFrame()

        # 1. Analisi HTF (High Timeframe)
        print("Ciao", TIMEFRAME_HTF)
        candele_htf = get_candele(self.symbol, TIMEFRAME_HTF, 200) # Ottieni più candele per analisi HTF
        swing_highs_htf, swing_lows_htf = identifica_swing_points(candele_htf)
        trend_htf, bos_high_htf, bos_low_htf = analizza_struttura_e_bos(swing_highs_htf, swing_lows_htf, self.previous_trend_htf)
        self.previous_trend_htf = trend_htf # Aggiorna il trend precedente

        print(f"[DEBUG] Trend HTF rilevato: {trend_htf}")

        range_htf = definisci_range_da_quasimodo(candele_htf, TIMEFRAME_HTF)
        if not range_htf:
            print("[DEBUG] Nessun range HTF valido.")
            return pd.DataFrame()
        print(f"[DEBUG] Range HTF definito: Low={range_htf.strong_low.low}, High={range_htf.strong_high.high}")

        # 2. Identificazione e Filtraggio POI HTF
        all_poi_htf = identifica_tutti_poi(candele_htf, TIMEFRAME_HTF)
        valid_poi_htf = filtra_poi_validi(all_poi_htf, swing_highs_htf, swing_lows_htf, candele_htf)

        print(f"[DEBUG] POI HTF validi trovati: {len(valid_poi_htf)}")

        # Logica di entrata a mercato
        current_price = self.candele[-1].close
        signal = "HOLD"
        trade_details = None

        if self.active_trade: # Se c\'è un trade attivo, gestiscilo
            print("trand attivo")
            self.gestisci_posizione_attiva(self.active_trade, range_htf)
            if self.active_trade.stato == "Chiuso":
                print(f"[DEBUG] Trade {self.active_trade.id} chiuso.")
                self.active_trade = None
            signal = "ACTIVE_TRADE"
        else:
            # Cerca opportunità di entrata
            for poi in valid_poi_htf:
                # Condizione di entrata: prezzo entra nel POI e direzione del POI è allineata al trend HTF
                print(poi.prezzo_di_attivazione_bottom, current_price, poi.prezzo_di_attivazione_top)
                print(poi.prezzo_di_attivazione_bottom <= current_price <= poi.prezzo_di_attivazione_top)
                if poi.prezzo_di_attivazione_bottom <= current_price <= poi.prezzo_di_attivazione_top:
                    print(trend_htf, poi.direzione)
                    if (trend_htf == "Bullish" and poi.direzione == "Bullish") or \
                       (trend_htf == "Bearish" and poi.direzione == "Bearish"):
                        print(f"[DEBUG] Prezzo nel POI HTF {poi.tipo} {poi.direzione}. Cerco conferma su LTF...")
                        # Gestione entrata a mercato (conferma o diretta)
                        new_trade = self.gestisci_entrata_a_mercato(poi, trend_htf)
                        if new_trade:
                            self.active_trade = new_trade
                            signal = new_trade.tipo # BUY o SELL
                            trade_details = {
                                "id": new_trade.id,
                                "coppia": new_trade.coppia,
                                "tipo": new_trade.tipo,
                                "prezzo_entrata": new_trade.prezzo_entrata,
                                "stop_loss": new_trade.stop_loss,
                                "take_profit_finale": new_trade.take_profit_finale,
                                "lottaggio": new_trade.lottaggio
                            }
                            break # Trovata un\'opportunità, esci dal ciclo POI

        print(signal)
        self.signals = pd.DataFrame([{
            "time": self.candele[-1].timestamp,
            "price": current_price,
            "trend_htf": trend_htf,
            "signal": signal,
            "trade_details": trade_details
        }])
        return self.signals

    def gestisci_entrata_a_mercato(self, poi_htf: POI, trend_htf: str) -> Optional[Trade]:
        # Metodo 2: Entrata con Conferma (più sicura)
        # Aspetta un cambio di struttura (CHOCH) sul LTFrelaxing 
        candele_ltf = get_candele(self.symbol, TIMEFRAME_LTF, 200) # Ottieni candele LTF
        range_ltf = definisci_range_da_quasimodo(candele_ltf, TIMEFRAME_LTF)

        if range_ltf:
            # Esempio per un trade SELL: Trend HTF è Bearish, prezzo in POI HTF Bearish.
            # Aspettiamo un QM Ribassista sul LTF per confermare.
            if trend_htf == "Bearish" and poi_htf.direzione == "Bearish":
                # Qui dovremmo verificare se un QM Ribassista si è formato sul LTF
                # Per semplicità, simuliamo una conferma se il prezzo è vicino allo strong_high del range LTF
                if abs(candele_ltf[-1].high - range_ltf.strong_high.high) < SPREAD * 5: # Prezzo vicino al QM high
                    print("Conferma LTF ricevuta! QM Ribassista formato (simulato).")
                    # Entra sul ritracciamento verso il POI del LTF che ha causato il BOS
                    poi_entrata_ltf = trova_poi_del_bos_ltf(range_ltf, candele_ltf)
                    if poi_entrata_ltf:
                        entry_price = poi_entrata_ltf.prezzo_di_attivazione_bottom # Per SELL
                        stop_loss = range_ltf.strong_high.high + SPREAD
                        take_profit = trova_target_logico(poi_htf, trend_htf)
                        stop_loss_pips = (stop_loss - entry_price) / 0.0001 # Calcola SL in pips
                        lottaggio = self.calculate_lot_size(stop_loss_pips)

                        trade_details = {
                            "coppia": self.symbol,
                            "tipo": "SELL",
                            "prezzo_entrata": entry_price,
                            "stop_loss": stop_loss,
                            "take_profit_finale": take_profit,
                            "lottaggio": lottaggio
                        }
                        order_id = place_order(trade_details)
                        return Trade(id=order_id, coppia=self.symbol, tipo="SELL",
                                    prezzo_entrata=entry_price, stop_loss=stop_loss,
                                    take_profit_finale=take_profit, lottaggio=lottaggio, stato="Aperto")

            # Esempio per un trade BUY: Trend HTF è Bullish, prezzo in POI HTF Bullish.
            # Aspettiamo un QM Rialzista sul LTF per confermare.
            elif trend_htf == "Bullish" and poi_htf.direzione == "Bullish":
                # Qui dovremmo verificare se un QM Rialzista si è formato sul LTF
                # Per semplicità, simuliamo una conferma se il prezzo è vicino allo strong_low del range LTF
                if abs(candele_ltf[-1].low - range_ltf.strong_low.low) < SPREAD * 5: # Prezzo vicino al QM low
                    print("Conferma LTF ricevuta! QM Rialzista formato (simulato).")
                    # Entra sul ritracciamento verso il POI del LTF che ha causato il BOS
                    poi_entrata_ltf = trova_poi_del_bos_ltf(range_ltf, candele_ltf)
                    if poi_entrata_ltf:
                        entry_price = poi_entrata_ltf.prezzo_di_attivazione_top # Per BUY
                        stop_loss = range_ltf.strong_low.low - SPREAD
                        take_profit = trova_target_logico(poi_htf, trend_htf)
                        stop_loss_pips = (entry_price - stop_loss) / 0.0001 # Calcola SL in pips
                        lottaggio = self.calculate_lot_size(stop_loss_pips)

                        trade_details = {
                            "coppia": self.symbol,
                            "tipo": "BUY",
                            "prezzo_entrata": entry_price,
                            "stop_loss": stop_loss,
                            "take_profit_finale": take_profit,
                            "lottaggio": lottaggio
                        }
                        order_id = place_order(trade_details)
                        return Trade(id=order_id, coppia=self.symbol, tipo="BUY",
                                     prezzo_entrata=entry_price, stop_loss=stop_loss,
                                     take_profit_finale=take_profit, lottaggio=lottaggio, stato="Aperto")

        # Metodo 1: Entrata Diretta (più rischiosa) - Fallback se non c\'è conferma LTF
        # Questo è un esempio semplificato, in un caso reale si raffinerebbe il POI HTF
        print("Nessuna conferma LTF, tentativo di entrata diretta (più rischiosa).")
        """entry_price = entry_price
        if poi_htf.direzione == "Bearish":
            stop_loss = poi_htf.prezzo_di_attivazione_top + SPREAD
            take_profit = trova_target_logico(poi_htf, trend_htf)
            stop_loss_pips = (stop_loss - entry_price) / 0.0001
            lottaggio = self.calculate_lot_size(stop_loss_pips)
            trade_details = {
                "coppia": self.symbol,
                "tipo": "SELL",
                "prezzo_entrata": entry_price,
                "stop_loss": stop_loss,
                "take_profit_finale": take_profit,
                "lottaggio": lottaggio
            }
            order_id = place_order(trade_details)
            return Trade(id=order_id, coppia=self.symbol, tipo="SELL",
                         prezzo_entrata=entry_price, stop_loss=stop_loss,
                         take_profit_finale=take_profit, lottaggio=lottaggio, stato="Aperto")
        elif poi_htf.direzione == "Bullish":
            stop_loss = poi_htf.prezzo_di_attivazione_bottom - SPREAD
            take_profit = trova_target_logico(poi_htf, trend_htf)
            stop_loss_pips = (entry_price - stop_loss) / 0.0001
            lottaggio = self.calculate_lot_size(stop_loss_pips)
            trade_details = {
                "coppia": self.symbol,
                "tipo": "BUY",
                "prezzo_entrata": entry_price,
                "stop_loss": stop_loss,
                "take_profit_finale": take_profit,
                "lottaggio": lottaggio
            }
            order_id = place_order(trade_details)
            return Trade(id=order_id, coppia=self.symbol, tipo="BUY",
                        prezzo_entrata=entry_price, stop_loss=stop_loss,
                        take_profit_finale=take_profit, lottaggio=lottaggio, stato="Aperto")"""
        return None

    def gestisci_posizione_attiva(self, trade: Trade, range_htf: RangeMercato):
        print(f"Gestione del trade {trade.id} in corso...")
        current_price = get_prezzo_corrente(trade.coppia)

        # 1. Logica di Break Even (Lezione 81)
        # Sposta SL a BE quando il prezzo prende la liquidità esterna di un range *inferiore*
        # o dopo un BOS a nostro favore sul LTF.
        # Per semplicità, qui simuliamo lo spostamento a BE dopo un certo profitto (es. 1R)
        if trade.stato == "Aperto":
            profit_pips = 0
            if trade.tipo == "BUY":
                profit_pips = (current_price - trade.prezzo_entrata) / 0.0001
            elif trade.tipo == "SELL":
                profit_pips = (trade.prezzo_entrata - current_price) / 0.0001

            sl_pips = abs(trade.prezzo_entrata - trade.stop_loss) / 0.0001

            if profit_pips >= sl_pips and trade.stato != "Breakeven": # Se il profitto è almeno 1R
                modify_sl(trade.id, trade.prezzo_entrata)
                trade.stato = "Breakeven"
                print(f"Trade {trade.id} spostato a Breakeven.")

        # 2. Logica dei Parziali (Lezione 84)
        # Primo parziale sul "Safe Target" (es. un minimo/massimo debole nel range HTF)
        if trade.parziali_presi == 0:
            safe_target = None
            if trade.tipo == "BUY" and range_htf.weak_highs:
                safe_target = sorted(range_htf.weak_highs, key=lambda c: c.high)[0].high # Primo weak high
            elif trade.tipo == "SELL" and range_htf.weak_lows:
                safe_target = sorted(range_htf.weak_lows, key=lambda c: c.low, reverse=True)[0].low # Primo weak low

            if safe_target:
                if (trade.tipo == "BUY" and current_price >= safe_target) or \
                   (trade.tipo == "SELL" and current_price <= safe_target):
                    close_partial_position(trade.id, 25) # Chiudi il 25%
                    trade.parziali_presi = 1
                    print(f"Preso primo parziale per il trade {trade.id}.")

        # Controllo se SL o TP sono stati colpiti
        if (trade.tipo == "BUY" and current_price <= trade.stop_loss) or \
        (trade.tipo == "SELL" and current_price >= trade.stop_loss):
            trade.stato = "Chiuso"
            print(f"Trade {trade.id} chiuso per Stop Loss.")
        elif (trade.tipo == "BUY" and current_price >= trade.take_profit_finale) or \
            (trade.tipo == "SELL" and current_price <= trade.take_profit_finale):
            trade.stato = "Chiuso"
            print(f"Trade {trade.id} chiuso per Take Profit.")


# Esempio di utilizzo (per test)
if __name__ == "__main__":
    # Dati di esempio per il DataFrame (simulazione)
    data_points = 200
    dates = [datetime.now() - timedelta(minutes=i*15) for i in range(data_points)][::-1]
    mock_data = {
        'time': dates,
        'open': np.random.uniform(1.04, 1.06, data_points),
        'high': np.random.uniform(1.05, 1.07, data_points),
        'low': np.random.uniform(1.03, 1.05, data_points),
        'close': np.random.uniform(1.04, 1.06, data_points),
    }
    df = pd.DataFrame(mock_data)

    strategy = TradingStrategy(symbol="EUR/USD", timeframe="15M", data=df)
    signals = strategy.generate_signals()
    print("\nSegnali generati:")
    print(signals)

    # Simulazione di un trade attivo per testare la gestione della posizione
    if strategy.active_trade:
        print(f"\nTrade attivo simulato: {strategy.active_trade}")
        # Simula il passare del tempo e il movimento del prezzo
        for _ in range(5):
            print("\nSimulazione tick...")
            # Aggiorna il prezzo corrente (mock)
            current_mock_price = get_prezzo_corrente(COPPIA_DI_VALUTE)
            # In un bot reale, qui si riceverebbero gli aggiornamenti di prezzo
            strategy.gestisci_posizione_attiva(strategy.active_trade, strategy.range_htf) # Necessario passare range_htf
            if strategy.active_trade.stato == "Chiuso":
                break
        print(f"Stato finale trade simulato: {strategy.active_trade.stato}")

