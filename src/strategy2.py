import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict
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
class SwingPoint:
    candela: Candela
    tipo: str  # "HIGH" o "LOW"
    timeframe: str
    confermato: bool = False
    liquidita_presa: bool = False


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
    swing_point_origine: Optional[SwingPoint] = None


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
# FUNZIONI DI SUPPORTO MIGLIORATE PER LA FRATTALITÀ
# -----------------------------------------------------------------------------
def identifica_swing_points_frattali(candele: List[Candela], period: int = 3, timeframe: str = "") -> List[SwingPoint]:
    """
    Implementazione migliorata della Formazione delle 3 Candele secondo la strategia Eclipse.
    
    Regole:
    - La seconda candela deve essere sempre la più alta o la più bassa delle 3
    - I colori delle candele non sono importanti, solo la formazione
    - Per un massimo: la candela centrale deve avere il high più alto delle 3
    - Per un minimo: la candela centrale deve avere il low più basso delle 3
    """
    swing_points = []
    
    # Assicuriamoci di avere abbastanza candele per l'analisi
    if len(candele) < period:
        print(f"[DEBUG] Dati insufficienti per swing points su {timeframe}: {len(candele)} candele (minimo {period})")
        return swing_points
    
    # Itera attraverso le candele per identificare i pattern di 3 candele
    for i in range(period // 2, len(candele) - period // 2):
        # Prendi le candele nel range del period
        window = candele[i - period // 2:i + period // 2 + 1]
        center_candle = candele[i]
        
        # Verifica se è un Swing High (massimo)
        is_swing_high = True
        for candle in window:
            if candle.high > center_candle.high:
                is_swing_high = False
                break
        
        # Verifica se è un Swing Low (minimo)
        is_swing_low = True
        for candle in window:
            if candle.low < center_candle.low:
                is_swing_low = False
                break
        
        # Aggiungi il swing point se identificato
        if is_swing_high and not is_swing_low:
            swing_points.append(SwingPoint(
                candela=center_candle,
                tipo="HIGH",
                timeframe=timeframe,
                confermato=False
            ))
        elif is_swing_low and not is_swing_high:
            swing_points.append(SwingPoint(
                candela=center_candle,
                tipo="LOW",
                timeframe=timeframe,
                confermato=False
            ))
    
    print(f"[DEBUG] Identificati {len(swing_points)} swing points su {timeframe}")
    return swing_points

"""
def identifica_swing_points(candele: List[Candela]):
    swing_highs, swing_lows = [], []
    for i in range(1, len(candele) - 1):
        prev, curr, nxt = candele[i-1], candele[i], candele[i+1]
        if curr.high > prev.high and curr.high > nxt.high:
            swing_highs.append(curr)
        if curr.low < prev.low and curr.low < nxt.low:
            swing_lows.append(curr)
    return swing_highs, swing_lows"""

"""
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
    return trend, ultimo_bos_high, ultimo_bos_low"""

def analizza_struttura_e_bos_frattale(swing_points: List[SwingPoint], trend_precedente: str = "Indefinito"):
    """
    Analisi della struttura di mercato migliorata con logica BOS/CHOCH secondo Eclipse.
    
    Concetti chiave:
    - HH (Higher High): Massimo più alto
    - HL (Higher Low): Minimo più alto  
    - LL (Lower Low): Minimo più basso
    - LH (Lower High): Massimo più basso
    - BOS (Break of Structure): Rottura strutturale
    - CHOCH (Change of Character): Cambio di carattere
    """
    swing_highs = [sp for sp in swing_points if sp.tipo == "HIGH"]
    swing_lows = [sp for sp in swing_points if sp.tipo == "LOW"]
    
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return "Indefinito", None, None, swing_points

    # Ordina gli swing points per data
    swing_highs.sort(key=lambda sp: sp.candela.timestamp)
    swing_lows.sort(key=lambda sp: sp.candela.timestamp)

    ultimo_high = swing_highs[-1]
    penultimo_high = swing_highs[-2]
    ultimo_low = swing_lows[-1]
    penultimo_low = swing_lows[-2]

    trend = "Indefinito"
    ultimo_bos_high = None
    ultimo_bos_low = None

    # Logica di conferma degli swing points secondo Eclipse
    # "In una situazione rialzista, fino a quando un nuovo HH è stato creato non possiamo confermare il nostro HL"
    # "In una situazione ribassista non possiamo confermare il nostro LH fino a quando non si è creato un LL"
    
    # Uptrend: Higher Highs (HH) e Higher Lows (HL)
    if ultimo_high.candela.high > penultimo_high.candela.high:
        # Abbiamo un HH, ora possiamo confermare gli HL precedenti
        for sp in swing_lows:
            if sp.candela.timestamp < ultimo_high.candela.timestamp:
                sp.confermato = True
        
        if ultimo_low.candela.low > penultimo_low.candela.low:
            trend = "Bullish"
            ultimo_bos_high = ultimo_high
            ultimo_high.confermato = True
    
    # Downtrend: Lower Highs (LH) e Lower Lows (LL)
    elif ultimo_low.candela.low < penultimo_low.candela.low:
        # Abbiamo un LL, ora possiamo confermare gli LH precedenti
        for sp in swing_highs:
            if sp.candela.timestamp < ultimo_low.candela.timestamp:
                sp.confermato = True
        
        if ultimo_high.candela.high < penultimo_high.candela.high:
            trend = "Bearish"
            ultimo_bos_low = ultimo_low
            ultimo_low.confermato = True

    # Inversione da Bullish a Bearish (CHOCH)
    if trend_precedente == "Bullish" and ultimo_low.candela.low < penultimo_low.candela.low:
        trend = "Bearish"
        print(f"[DEBUG] CHOCH rilevato: da Bullish a Bearish")
    
    # Inversione da Bearish a Bullish (CHOCH)
    elif trend_precedente == "Bearish" and ultimo_high.candela.high > penultimo_high.candela.high:
        trend = "Bullish"
        print(f"[DEBUG] CHOCH rilevato: da Bearish a Bullish")

    print(f"[DEBUG] Trend: {trend}, BOS High: {ultimo_bos_high}, BOS Low: {ultimo_bos_low}")
    return trend, ultimo_bos_high, ultimo_bos_low, swing_points
def identifica_liquidita_swing_points(swing_points: List[SwingPoint]) -> Dict[str, List[float]]:
    """
    Identifica le zone di liquidità basate sui swing points secondo la strategia Eclipse.
    
    Concetti:
    - Buy Side Liquidity: al di sopra di ogni Swing High (Buy Stops)
    - Sell Side Liquidity: al di sotto di ogni Swing Low (Sell Stops)
    """
    liquidita = {
        "buy_side": [],  # Livelli di liquidità buy (sopra swing highs)
        "sell_side": []  # Livelli di liquidità sell (sotto swing lows)
    }
    
    for sp in swing_points:
        if sp.tipo == "HIGH" and not sp.liquidita_presa:
            # Buy Side Liquidity sopra il swing high
            liquidita["buy_side"].append(sp.candela.high)
        elif sp.tipo == "LOW" and not sp.liquidita_presa:
            # Sell Side Liquidity sotto il swing low
            liquidita["sell_side"].append(sp.candela.low)
    
    print(f"[DEBUG] Liquidità identificata - Buy Side: {len(liquidita["buy_side"])}, Sell Side: {len(liquidita["sell_side"])})")
    return liquidita
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

"""
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
    return nuovo_range"""

def definisci_range_da_quasimodo(candele: List[Candela], timeframe: str) -> Optional[RangeMercato]:
    print(f"Avvio identificazione Range su timeframe {timeframe}...")
    swing_points = identifica_swing_points_frattali(candele, timeframe=timeframe)
    swing_highs = [sp.candela for sp in swing_points if sp.tipo == "HIGH"]
    swing_lows = [sp.candela for sp in swing_points if sp.tipo == "LOW"]

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        print("Dati insufficienti (swing points) per definire un range.")
        return None

    strong_high_del_range = None
    for i in range(len(swing_highs) - 1, 0, -1):
        high_attuale = swing_highs[i]
        high_precedente = swing_highs[i-1]

        stop_hunt_eseguito = high_attuale.high > high_precedente.high

        bos_al_ribasso_confermato = False
        low_precedente_allo_stophunt = trova_ultimo_low_prima_di(high_attuale.timestamp, swing_lows)
        if low_precedente_allo_stophunt:
            low_successivo_allo_stophunt = trova_primo_low_dopo(high_attuale.timestamp, swing_lows)
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
    swing_lows_precedenti = [low for low in swing_lows if low.timestamp < strong_high_del_range.timestamp]
    for i in range(len(swing_lows_precedenti) - 1, 0, -1):
        low_attuale = swing_lows_precedenti[i]
        low_precedente = swing_lows_precedenti[i-1]

        stop_hunt_eseguito = low_attuale.low < low_precedente.low

        bos_al_rialzo_confermato = False
        high_precedente_allo_stophunt = trova_ultimo_high_prima_di(low_attuale.timestamp, swing_highs)
        if high_precedente_allo_stophunt:
            high_successivo_allo_stophunt = trova_primo_high_dopo(low_attuale.timestamp, swing_highs)
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
        h for h in swing_highs
        if strong_low_del_range.timestamp < h.timestamp < strong_high_del_range.timestamp
    ]
    nuovo_range.weak_lows = [
        l for l in swing_lows
        if strong_low_del_range.timestamp < l.timestamp < strong_high_del_range.timestamp
    ]

    print(f"Range valido definito su {timeframe}: Low a {nuovo_range.strong_low.low}, High a {nuovo_range.strong_high.high}")
    return nuovo_range

"""
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
                    key_level_ohlc={\'open\': candela_prec.open, \'high\': candela_prec.high,
                                    \'low\': candela_prec.low, \'close\': candela_prec.close},
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
                    key_level_ohlc={\'open\': candela_prec.open, \'high\': candela_prec.high,
                                    \'low\': candela_prec.low, \'close\': candela_prec.close},
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


def filtra_poi_validi(lista_poi: List[POI], swing_points_high: List[Candela], swing_points_low: List[Candela], candele: List[Candela]) -> List[POI]:
    poi_validi = []
    for poi in lista_poi:
        has_taken_liquidity = False
        is_mitigated = False

        # REGOLA 2: Deve aver "preso" liquidità.
        # Per un OB ribassista, deve aver rotto un minimo precedente (non un massimo come scritto prima)
        if poi.direzione == "Bearish":
            for swing_low in swing_points_low:
                if poi.candela_di_riferimento.low < swing_low.low and poi.candela_di_riferimento.timestamp > swing_low.timestamp:
                    has_taken_liquidity = True
                    break
        # Per un OB rialzista, deve aver rotto un massimo precedente
        elif poi.direzione == "Bullish":
            for swing_high in swing_points_high:
                if poi.candela_di_riferimento.high > swing_high.high and poi.candela_di_riferimento.timestamp > swing_high.timestamp:
                    has_taken_liquidity = True
                    break

        # REGOLA 3: Non deve essere mitigato.
        # Controlla se il prezzo è già tornato in quella zona
        # Per un POI ribassista (vendita), il prezzo non deve aver superato il top del POI
        if poi.direzione == "Bearish":
            for candela in candele:
                if candela.high >= poi.prezzo_di_attivazione_top and candela.timestamp > poi.candela_di_riferimento.timestamp:
                    is_mitigated = True
                    break
        # Per un POI rialzista (acquisto), il prezzo non deve aver superato il bottom del POI
        elif poi.direzione == "Bullish":
            for candela in candele:
                if candela.low <= poi.prezzo_di_attivazione_bottom and candela.timestamp > poi.candela_di_riferimento.timestamp:
                    is_mitigated = True
                    break

        if has_taken_liquidity and not is_mitigated:
            poi_validi.append(poi)

    print(f"[DEBUG] Identificati {len(poi_validi)} POI validi dopo il filtraggio.")
    return poi_validi"""

def identifica_tutti_poi(candele: List[Candela], swing_points: List[SwingPoint], timeframe: str) -> List[POI]:
    """
    Identifica tutti i POI basandosi sui swing points e sulle formazioni di candele.
    Versione migliorata che considera la relazione con gli swing points.
    """
    print("Identifica_Poi timeframe:", timeframe, " | Candele:", len(candele), " | Swing Points:", len(swing_points))
    lista_poi = []

    for i in range(1, len(candele) - 1):
        candela_prec = candele[i-1]   # candela precedente
        candela_curr = candele[i]     # candela centrale (candela di riferimento)
        candela_succ = candele[i+1]   # candela successiva

        # Trova il swing point associato a questa candela (se esiste)
        swing_point_associato = None
        for sp in swing_points:
            if sp.candela.timestamp == candela_curr.timestamp:
                swing_point_associato = sp
                break

        # -------------------------------
        # 1. ORDER BLOCK RIBASSISTA (ultima candela verde prima del dump rosso)
        # -------------------------------
        if candela_prec.close > candela_prec.open and candela_curr.close < candela_curr.open:
            # Controllo se la candela successiva conferma la spinta ribassista
            if candela_succ.low < candela_prec.low:
                poi_top = candela_prec.close   # Usa il close della candela verde
                poi_bottom = candela_prec.low
                lista_poi.append(POI(
                    tipo="Orderblock",
                    direzione="Bearish",
                    candela_di_riferimento=candela_prec,
                    prezzo_di_attivazione_top=poi_top,
                    prezzo_di_attivazione_bottom=poi_bottom,
                    key_level_ohlc={\'open\': candela_prec.open, \'high\': candela_prec.high,
                                    \'low\': candela_prec.low, \'close\': candela_prec.close},
                    timeframe=timeframe,
                    swing_point_origine=swing_point_associato
                ))

        # -------------------------------
        # 2. ORDER BLOCK RIALZISTA (ultima candela rossa prima del pump verde)
        # -------------------------------
        if candela_prec.close < candela_prec.open and candela_curr.close > candela_curr.open:
            # Controllo se la candela successiva conferma la spinta rialzista
            if candela_succ.high > candela_prec.high:
                poi_top = candela_prec.high
                poi_bottom = candela_prec.close   # Usa il close della candela rossa
                lista_poi.append(POI(
                    tipo="Orderblock",
                    direzione="Bullish",
                    candela_di_riferimento=candela_prec,
                    prezzo_di_attivazione_top=poi_top,
                    prezzo_di_attivazione_bottom=poi_bottom,
                    key_level_ohlc={\'open\': candela_prec.open, \'high\': candela_prec.high,
                                    \'low\': candela_prec.low, \'close\': candela_prec.close},
                    timeframe=timeframe,
                    swing_point_origine=swing_point_associato
                ))

        # -------------------------------
        # 3. FAIR VALUE GAP (FVG) RIALZISTA
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
                timeframe=timeframe,
                swing_point_origine=swing_point_associato
            ))

        # -------------------------------
        # 4. FAIR VALUE GAP (FVG) RIBASSISTA
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
                timeframe=timeframe,
                swing_point_origine=swing_point_associato
            ))

    # Debug finale
    print(f"[DEBUG] Identificati {len(lista_poi)} POI su timeframe {timeframe}")
    for poi in lista_poi[:3]:  # Mostra solo i primi 3 per non intasare i log
        print(f"  {poi.tipo} {poi.direzione} | Top={poi.prezzo_di_attivazione_top:.5f} | Bottom={poi.prezzo_di_attivazione_bottom:.5f} | Ref={poi.candela_di_riferimento.timestamp}")

    return lista_poi


def filtra_poi_validi(lista_poi: List[POI], swing_points: List[SwingPoint], candele: List[Candela]) -> List[POI]:
    """
    Filtra i POI secondo le regole della strategia Eclipse.
    
    REGOLA 2: Deve aver "preso" liquidità
    REGOLA 3: Non deve essere mitigato
    """
    poi_validi = []
    
    for poi in lista_poi:
        has_taken_liquidity = False
        is_mitigated = False

        # REGOLA 2: Deve aver "preso" liquidità
        # Verifica se il POI è associato a un movimento che ha rotto liquidità
        if poi.swing_point_origine:
            # Se il POI è associato a uno swing point, verifica se ha preso liquidità
            if poi.direzione == "Bearish":
                # Per un POI bearish, verifica se ha rotto un low precedente
                for sp in swing_points:
                    if (sp.tipo == "LOW" and 
                        sp.candela.timestamp < poi.candela_di_riferimento.timestamp and
                        poi.candela_di_riferimento.low < sp.candela.low):
                        has_taken_liquidity = True
                        sp.liquidita_presa = True
                        break
            elif poi.direzione == "Bullish":
                # Per un POI bullish, verifica se ha rotto un high precedente
                for sp in swing_points:
                    if (sp.tipo == "HIGH" and 
                        sp.candela.timestamp < poi.candela_di_riferimento.timestamp and
                        poi.candela_di_riferimento.high > sp.candela.high):
                        has_taken_liquidity = True
                        sp.liquidita_presa = True
                        break
        else:
            # Se non è associato a uno swing point, usa la logica precedente
            if poi.direzione == "Bearish":
                swing_lows = [sp.candela for sp in swing_points if sp.tipo == "LOW"]
                for swing_low in swing_lows:
                    if poi.candela_di_riferimento.low < swing_low.low and poi.candela_di_riferimento.timestamp > swing_low.timestamp:
                        has_taken_liquidity = True
                        break
            elif poi.direzione == "Bullish":
                swing_highs = [sp.candela for sp in swing_points if sp.tipo == "HIGH"]
                for swing_high in swing_highs:
                    if poi.candela_di_riferimento.high > swing_high.high and poi.candela_di_riferimento.timestamp > swing_high.timestamp:
                        has_taken_liquidity = True
                        break

        # REGOLA 3: Non deve essere mitigato
        # Verifica se il prezzo è tornato nella zona del POI dopo la sua formazione
        for candela in candele:
            if candela.timestamp <= poi.candela_di_riferimento.timestamp:
                continue
                
            if poi.direzione == "Bearish":
                # Per un POI bearish, è mitigato se il prezzo torna sopra il top
                if candela.high >= poi.prezzo_di_attivazione_top:
                    is_mitigated = True
                    poi.e_mitigato = True
                    break
            elif poi.direzione == "Bullish":
                # Per un POI bullish, è mitigato se il prezzo torna sotto il bottom
                if candela.low <= poi.prezzo_di_attivazione_bottom:
                    is_mitigated = True
                    poi.e_mitigato = True
                    break

        # Aggiungi il POI solo se ha preso liquidità e non è mitigato
        if has_taken_liquidity and not is_mitigated:
            poi_validi.append(poi)

    print(f"[DEBUG] POI validi dopo filtraggio: {len(poi_validi)}/{len(lista_poi)}")
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
    def __init__(self, symbol: str, main_timeframe: str, ohlcv_data: Dict[str, pd.DataFrame], params: dict):
        self.symbol = symbol
        self.main_timeframe = main_timeframe
        self.ohlcv_data = ohlcv_data
        self.params = params
        self.signals = pd.DataFrame(columns=["time", "signal", "stop_loss", "take_profit"])
        
        # Converti tutti i DataFrame in liste di oggetti Candela
        self.candele_multi_timeframe = {
            tf: self._df_to_candele(df) 
            for tf, df in ohlcv_data.items()
        }
        
        # Stato della strategia
        self.trend = "Indefinito"
        self.ultimo_bos_high = None
        self.ultimo_bos_low = None
        self.range_mercato = None
        self.swing_points_multi_timeframe = {}
        self.liquidita_multi_timeframe = {}

    def _df_to_candele(self, df: pd.DataFrame) -> List[Candela]:
        candele = []
        for _, row in df.iterrows():
            candele.append(Candela(
                timestamp=row["time"],
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"]
            ))
        return candele

    def calculate_lot_size(self, stop_loss_pips: float) -> float:
        # Calcola la dimensione del lotto in base al rischio per trade
        # 1 pip = 0.0001 per EUR/USD
        valore_pip = 10.0 # Per EUR/USD, 1 pip per lotto standard (100,000 unità) vale 10 USD
        rischio_denaro = CAPITALE_CONTO * (RISCHIO_PER_TRADE_PERCENTUALE / 100)
        lottaggio = rischio_denaro / (stop_loss_pips * valore_pip)
        return lottaggio

    def generate_signals(self):
        # Ottieni le candele per il timeframe principale
        main_candele = self.candele_multi_timeframe.get(self.main_timeframe)
        if not main_candele:
            print(f"[ERRORE] Dati non disponibili per il timeframe principale: {self.main_timeframe}")
            return

        print(f"[DEBUG] Analisi su timeframe principale: {self.main_timeframe} con {len(main_candele)} candele")

        # 1. Identificazione Swing Points su tutti i timeframe
        for tf_name, candele_tf in self.candele_multi_timeframe.items():
            if candele_tf:
                swing_period = self.params.get("swing_period", 3)
                swing_points = identifica_swing_points_frattali(candele_tf, swing_period, tf_name)
                self.swing_points_multi_timeframe[tf_name] = swing_points
                
                # Identifica liquidità per questo timeframe
                liquidita = identifica_liquidita_swing_points(swing_points)
                self.liquidita_multi_timeframe[tf_name] = liquidita

        # 2. Analisi Struttura e BOS sul timeframe principale
        main_swing_points = self.swing_points_multi_timeframe.get(self.main_timeframe, [])
        if main_swing_points:
            self.trend, self.ultimo_bos_high, self.ultimo_bos_low, updated_swing_points = analizza_struttura_e_bos_frattale(
                main_swing_points, self.trend
            )
            self.swing_points_multi_timeframe[self.main_timeframe] = updated_swing_points
            print(f"[DEBUG] Trend attuale ({self.main_timeframe}): {self.trend}")

        # 3. Identificazione Range di Mercato (Quasimodo) sul timeframe principale
        self.range_mercato = definisci_range_da_quasimodo(main_candele, self.main_timeframe)
        if self.range_mercato:
            print(f"[DEBUG] Range di Mercato ({self.main_timeframe}): Strong High={self.range_mercato.strong_high.high:.5f}, Strong Low={self.range_mercato.strong_low.low:.5f}")

        # 4. Identificazione POI su tutti i timeframe disponibili
        all_poi = []
        for tf_name, candele_tf in self.candele_multi_timeframe.items():
            if candele_tf and tf_name in self.swing_points_multi_timeframe:
                swing_points_tf = self.swing_points_multi_timeframe[tf_name]
                current_tf_poi = identifica_tutti_poi(candele_tf, swing_points_tf, tf_name)
                valid_tf_poi = filtra_poi_validi(current_tf_poi, swing_points_tf, candele_tf)
                all_poi.extend(valid_tf_poi)
        
        print(f"[DEBUG] Totale POI validi identificati su tutti i timeframe: {len(all_poi)}")

        # 5. Logica di Trading migliorata con considerazione frattale
        signal = None
        stop_loss = 0.0
        take_profit = 0.0
        last_candle_main_tf = main_candele[-1]

        # Prioritizza i POI per rilevanza (timeframe più alti hanno priorità)
        timeframe_priority = {"MN1": 7, "W1": 6, "D1": 5, "H4": 4, "H1": 3, "M30": 2, "M15": 1, "M5": 0, "M1": -1}
        all_poi.sort(key=lambda poi: timeframe_priority.get(poi.timeframe, -1), reverse=True)

        for poi in all_poi:
            # Verifica validità temporale del POI
            # La validità del POI dovrebbe essere relativa al timeframe del POI stesso
            # e non solo al timeframe principale
            poi_validity_bars = self.params.get("poi_validity_bars", 50)
            # Converti poi_validity_bars in minuti per confronto
            # Assumiamo che poi_validity_bars sia in barre del timeframe del POI
            # Questo è un placeholder, la conversione esatta dipende dalla granularità del timeframe
            # Per ora, usiamo una stima approssimativa o un valore fisso per M1
            
            # Per una logica più precisa, dovremmo avere una mappa dei minuti per timeframe
            timeframe_to_minutes = {
                "M1": 1, "M5": 5, "M15": 15, "M30": 30,
                "H1": 60, "H4": 240, "D1": 1440, "W1": 10080, "MN1": 43200
            }
            poi_timeframe_minutes = timeframe_to_minutes.get(poi.timeframe, 1) # Default a M1 se non trovato
            max_validity_minutes = poi_validity_bars * poi_timeframe_minutes

            time_diff_minutes = (last_candle_main_tf.timestamp - poi.candela_di_riferimento.timestamp).total_seconds() / 60
            
            if time_diff_minutes > max_validity_minutes:
                print(f"[DEBUG] POI scartato per anzianità: {poi.tipo} {poi.direzione} su {poi.timeframe} (vecchio di {time_diff_minutes:.2f} min, max {max_validity_minutes} min)")
                continue

            # Verifica se il prezzo corrente è nella zona del POI
            current_price = last_candle_main_tf.close
            
            # Logica di ingresso: Manipolazione e Reazione al POI
            # Questa è la parte cruciale per la frattalità
            # Dobbiamo cercare una 


            # Logica di ingresso: Manipolazione e Reazione al POI
            # Questa è la parte cruciale per la frattalità
            # Dobbiamo cercare una manipolazione del POI seguita da una reazione
            
            # Verifica se siamo in una zona di POI valida
            in_poi_zone = False
            if poi.direzione == "Bullish":
                # Per POI bullish, il prezzo deve essere nella zona del POI
                if poi.prezzo_di_attivazione_bottom <= current_price <= poi.prezzo_di_attivazione_top:
                    in_poi_zone = True
            elif poi.direzione == "Bearish":
                # Per POI bearish, il prezzo deve essere nella zona del POI
                if poi.prezzo_di_attivazione_bottom <= current_price <= poi.prezzo_di_attivazione_top:
                    in_poi_zone = True
            
            if not in_poi_zone:
                continue
            
            # Verifica la confluenza con il trend del timeframe principale
            # Questo è un aspetto chiave della frattalità: allineamento multi-timeframe
            trend_alignment = False
            if poi.direzione == "Bullish" and self.trend in ["Bullish", "Indefinito"]:
                trend_alignment = True
            elif poi.direzione == "Bearish" and self.trend in ["Bearish", "Indefinito"]:
                trend_alignment = True
            
            if not trend_alignment:
                print(f"[DEBUG] POI scartato per disallineamento trend: POI {poi.direzione} vs Trend {self.trend}")
                continue
            
            # Verifica la presenza di liquidità nella direzione del trade
            # Questo è fondamentale per la strategia Eclipse
            has_liquidity_target = False
            if poi.direzione == "Bullish":
                # Per un trade BUY, cerchiamo liquidità buy-side (sopra swing highs)
                tf_liquidita = self.liquidita_multi_timeframe.get(poi.timeframe, {})
                if tf_liquidita.get("buy_side", []):
                    # Verifica se c'è liquidità sopra il prezzo corrente
                    for liq_level in tf_liquidita["buy_side"]:
                        if liq_level > current_price:
                            has_liquidity_target = True
                            break
            elif poi.direzione == "Bearish":
                # Per un trade SELL, cerchiamo liquidità sell-side (sotto swing lows)
                tf_liquidita = self.liquidita_multi_timeframe.get(poi.timeframe, {})
                if tf_liquidita.get("sell_side", []):
                    # Verifica se c'è liquidità sotto il prezzo corrente
                    for liq_level in tf_liquidita["sell_side"]:
                        if liq_level < current_price:
                            has_liquidity_target = True
                            break
            
            if not has_liquidity_target:
                print(f"[DEBUG] POI scartato per mancanza di liquidità target: {poi.tipo} {poi.direzione}")
                continue
            
            # Se arriviamo qui, abbiamo un setup valido
            # Generiamo il segnale con logica di ingresso/uscita migliorata
            
            if poi.direzione == "Bullish":
                signal = "BUY"
                
                # Calcolo Stop Loss migliorato
                # SL sotto il POI con un buffer basato sulla volatilità
                buffer_percentage = 0.0005  # 0.05% buffer
                stop_loss = poi.prezzo_di_attivazione_bottom * (1 - buffer_percentage)
                
                # Calcolo Take Profit basato su liquidità target
                risk = current_price - stop_loss
                
                # Trova il livello di liquidità più vicino come primo target
                tf_liquidita = self.liquidita_multi_timeframe.get(poi.timeframe, {})
                nearest_liquidity = None
                for liq_level in tf_liquidita.get("buy_side", []):
                    if liq_level > current_price:
                        if nearest_liquidity is None or liq_level < nearest_liquidity:
                            nearest_liquidity = liq_level
                
                if nearest_liquidity:
                    # TP basato sulla liquidità più vicina
                    take_profit = nearest_liquidity * 0.999  # Leggermente sotto la liquidità
                else:
                    # TP basato sul risk/reward ratio se non c'è liquidità vicina
                    take_profit = current_price + (risk * self.params.get("risk_reward_ratio", 2.0))
                
                print(f"[DEBUG] Segnale BUY generato da POI {poi.tipo} Bullish ({poi.timeframe}) a {current_price:.5f}")
                print(f"[DEBUG] SL: {stop_loss:.5f}, TP: {take_profit:.5f}, Risk: {risk:.5f}, RR: {(take_profit-current_price)/risk:.2f}")
                break
                
            elif poi.direzione == "Bearish":
                signal = "SELL"
                
                # Calcolo Stop Loss migliorato
                # SL sopra il POI con un buffer basato sulla volatilità
                buffer_percentage = 0.0005  # 0.05% buffer
                stop_loss = poi.prezzo_di_attivazione_top * (1 + buffer_percentage)
                
                # Calcolo Take Profit basato su liquidità target
                risk = stop_loss - current_price
                
                # Trova il livello di liquidità più vicino come primo target
                tf_liquidita = self.liquidita_multi_timeframe.get(poi.timeframe, {})
                nearest_liquidity = None
                for liq_level in tf_liquidita.get("sell_side", []):
                    if liq_level < current_price:
                        if nearest_liquidity is None or liq_level > nearest_liquidity:
                            nearest_liquidity = liq_level
                
                if nearest_liquidity:
                    # TP basato sulla liquidità più vicina
                    take_profit = nearest_liquidity * 1.001  # Leggermente sopra la liquidità
                else:
                    # TP basato sul risk/reward ratio se non c'è liquidità vicina
                    take_profit = current_price - (risk * self.params.get("risk_reward_ratio", 2.0))
                
                print(f"[DEBUG] Segnale SELL generato da POI {poi.tipo} Bearish ({poi.timeframe}) a {current_price:.5f}")
                print(f"[DEBUG] SL: {stop_loss:.5f}, TP: {take_profit:.5f}, Risk: {risk:.5f}, RR: {(current_price-take_profit)/risk:.2f}")
                break

        if signal:
            new_signal = pd.DataFrame([{
                "time": last_candle_main_tf.timestamp, 
                "signal": signal, 
                "stop_loss": stop_loss, 
                "take_profit": take_profit
            }])
            self.signals = pd.concat([self.signals, new_signal], ignore_index=True)
        else:
            print("[DEBUG] Nessun segnale generato in questo ciclo.")

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

