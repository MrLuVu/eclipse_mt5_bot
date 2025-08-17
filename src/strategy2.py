import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import MetaTrader5 as mt5


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

    def __eq__(self, other):
        if not isinstance(other, Candela):
            return NotImplemented
        return self.timestamp == other.timestamp and \
               self.open == other.open and \
               self.high == other.high and \
               self.low == other.low and \
               self.close == other.close


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
    
    # Assicuriamoci di avere abbastanza candele per l\"analisi
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

def identifica_liquidita_swing_points(candele: List[Candela]) -> Dict[str, List[float]]:
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
    
    # Per questa funzione, assumiamo che gli swing points siano già stati identificati
    # e che le candele siano ordinate temporalmente.
    
    # Identifica i massimi e minimi significativi per la liquidità
    # Questo è un placeholder, la logica reale dovrebbe essere più sofisticata
    # e basarsi su swing points confermati o livelli chiave.
    
    # Esempio semplificato: tutti i massimi e minimi delle candele sono potenziali liquidità
    for candela in candele:
        liquidita["buy_side"].append(candela.high) # Potenziale liquidità sopra i massimi
        liquidita["sell_side"].append(candela.low)  # Potenziale liquidità sotto i minimi

    # Rimuovi duplicati e ordina
    liquidita["buy_side"] = sorted(list(set(liquidita["buy_side"])), reverse=True)
    liquidita["sell_side"] = sorted(list(set(liquidita["sell_side"])))

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

def identifica_equal_highs_lows(candele: List[Candela], tolerance: float = 0.0001, timeframe: str = "") -> Dict[str, List[float]]:
    """
    Identifica Massimi Uguali (Equal Highs) e Minimi Uguali (Equal Lows).
    Tolerance è una percentuale del prezzo per considerare i massimi/minimi "uguali".
    """
    equal_levels = {"equal_highs": [], "equal_lows": []}
    
    # Equal Highs
    for i in range(len(candele) - 1):
        for j in range(i + 1, len(candele)):
            if abs(candele[i].high - candele[j].high) <= candele[i].high * tolerance:
                # Evita duplicati e aggiungi solo se non già presente
                if not any(abs(level - candele[i].high) <= candele[i].high * tolerance for level in equal_levels["equal_highs"]):
                    equal_levels["equal_highs"].append(candele[i].high)
    
    # Equal Lows
    for i in range(len(candele) - 1):
        for j in range(i + 1, len(candele)):
            if abs(candele[i].low - candele[j].low) <= candele[i].low * tolerance:
                # Evita duplicati e aggiungi solo se non già presente
                if not any(abs(level - candele[i].low) <= candele[i].low * tolerance for level in equal_levels["equal_lows"]):
                    equal_levels["equal_lows"].append(candele[i].low)
                    
    print(f"[DEBUG] Identificati {len(equal_levels["equal_highs"])} Equal Highs e {len(equal_levels["equal_lows"])} Equal Lows su {timeframe}")
    return equal_levels

def identifica_trendline_liquidity(candele: List[Candela], timeframe: str) -> Dict[str, List[float]]:
    """
    Identifica la liquidità delle trendline.
    Questo è un approccio semplificato e può essere migliorato con algoritmi più complessi.
    """
    trendline_liquidity = {"buy_side": [], "sell_side": []}
    
    # Per identificare una trendline rialzista (liquidità sell-side sotto di essa),
    # cerchiamo almeno 3 minimi crescenti.
    if len(candele) >= 3:
        for i in range(len(candele) - 2):
            # Semplice identificazione di 3 minimi crescenti
            if candele[i].low < candele[i+1].low < candele[i+2].low:
                # Consideriamo il low della candela più recente come un potenziale livello di liquidità
                trendline_liquidity["sell_side"].append(candele[i+2].low)
            
            # Semplice identificazione di 3 massimi decrescenti
            if candele[i].high > candele[i+1].high > candele[i+2].high:
                # Consideriamo l\"high della candela più recente come un potenziale livello di liquidità
                trendline_liquidity["buy_side"].append(candele[i+2].high)
                
    print(f"[DEBUG] Identificati {len(trendline_liquidity["buy_side"])} Trendline Buy Side e {len(trendline_liquidity["sell_side"])} Trendline Sell Side su {timeframe}")
    return trendline_liquidity

def identifica_asian_session_liquidity(candele: List[Candela], session_asian_start: int, session_asian_end: int, timeframe: str) -> Dict[str, float]:
    """
    Identifica i massimi e minimi della sessione asiatica.
    Assume che le candele siano ordinate per timestamp.
    """
    # Inizializza con valori float validi, non con np.inf o -np.inf che possono causare problemi di tipo
    asian_session_high = -1.0  # Un valore basso iniziale per il massimo
    asian_session_low = float("inf") # Un valore alto iniziale per il minimo
    
    # Trova le candele che rientrano nella sessione asiatica
    # La sessione asiatica può estendersi a cavallo della mezzanotte
    asian_candele = []
    for candela in candele:
        hour = candela.timestamp.hour
        if session_asian_start < session_asian_end:
            # Sessione che non attraversa la mezzanotte (es. 23-6)
            if session_asian_start <= hour < session_asian_end:
                asian_candele.append(candela)
        else:
            # Sessione che attraversa la mezzanotte (es. 23-6, dove 23 è il giorno prima)
            if hour >= session_asian_start or hour < session_asian_end:
                asian_candele.append(candela)

    if not asian_candele:
        print(f"[DEBUG] Nessuna candela trovata per la sessione asiatica su {timeframe}")
        return {"high": 0.0, "low": 0.0}

    for candela in asian_candele:
        if candela.high > asian_session_high:
            asian_session_high = candela.high
        if candela.low < asian_session_low:
            asian_session_low = candela.low
            
    print(f"[DEBUG] Liquidità Sessione Asiatica su {timeframe}: High={asian_session_high}, Low={asian_session_low}")
    return {"high": asian_session_high, "low": asian_session_low}

def identifica_order_block(candele: List[Candela], index: int, direzione: str) -> Optional[POI]:
    """
    Identifica un Order Block (OB) rialzista o ribassista.
    Un OB rialzista è l\"ultima candela ribassista prima di un movimento impulsivo rialzista.
    Un OB ribassista è l\"ultima candela rialzista prima di un movimento impulsivo ribassista.
    """
    if index < 1 or index >= len(candele):
        return None

    candela_attuale = candele[index]
    candela_precedente = candele[index - 1]

    if direzione == "Bullish":
        # OB rialzista: ultima candela ribassista prima di un movimento rialzista
        if candela_precedente.close < candela_precedente.open:  # Candela precedente ribassista
            # Movimento impulsivo rialzista: candela attuale chiude sopra la precedente
            if candela_attuale.close > candela_attuale.open and candela_attuale.close > candela_precedente.high:
                return POI(
                    tipo="Orderblock",
                    direzione="Bullish",
                    candela_di_riferimento=candela_precedente,
                    prezzo_di_attivazione_top=candela_precedente.open,  # Open della candela ribassista
                    prezzo_di_attivazione_bottom=candela_precedente.low, # Low della candela ribassista
                    key_level_ohlc={"open": candela_precedente.open, "high": candela_precedente.high, "low": candela_precedente.low, "close": candela_precedente.close}
                )
    elif direzione == "Bearish":
        # OB ribassista: ultima candela rialzista prima di un movimento ribassista
        if candela_precedente.close > candela_precedente.open:  # Candela precedente rialzista
            # Movimento impulsivo ribassista: candela attuale chiude sotto la precedente
            if candela_attuale.close < candela_attuale.open and candela_attuale.close < candela_precedente.low:
                return POI(
                    tipo="Orderblock",
                    direzione="Bearish",
                    candela_di_riferimento=candela_precedente,
                    prezzo_di_attivazione_top=candela_precedente.high, # High della candela rialzista
                    prezzo_di_attivazione_bottom=candela_precedente.open, # Open della candela rialzista
                    key_level_ohlc={"open": candela_precedente.open, "high": candela_precedente.high, "low": candela_precedente.low, "close": candela_precedente.close}
                )
    return None

def identifica_fair_value_gap(candele: List[Candela], index: int, direzione: str) -> Optional[POI]:
    """
    Identifica un Fair Value Gap (FVG) rialzista o ribassista.
    FVG rialzista: low della candela 1 > high della candela 3 (con candela 2 tra 1 e 3).
    FVG ribassista: high della candela 1 < low della candela 3.
    """
    if index < 2 or index >= len(candele):
        return None

    candela_1 = candele[index - 2]
    candela_2 = candele[index - 1] # Candela di riferimento
    candela_3 = candele[index]

    if direzione == "Bullish":
        # FVG rialzista: low di candela_1 > high di candela_3
        if candela_1.low > candela_3.high:
            return POI(
                tipo="Inefficiency",
                direzione="Bullish",
                candela_di_riferimento=candela_2,
                prezzo_di_attivazione_top=candela_1.low,  # Top del gap
                prezzo_di_attivazione_bottom=candela_3.high, # Bottom del gap
                key_level_ohlc={}
            )
    elif direzione == "Bearish":
        # FVG ribassista: high di candela_1 < low di candela_3
        if candela_1.high < candela_3.low:
            return POI(
                tipo="Inefficiency",
                direzione="Bearish",
                candela_di_riferimento=candela_2,
                prezzo_di_attivazione_top=candela_3.low,  # Top del gap
                prezzo_di_attivazione_bottom=candela_1.high, # Bottom del gap
                key_level_ohlc={}
            )
    return None

def identifica_breaker_block(candele: List[Candela], index: int, direzione: str) -> Optional[POI]:
    """
    Identifica un Breaker Block (BB).
    Un BB rialzista si forma dopo un movimento ribassista che rompe una struttura rialzista, 
    ed è l\"ultima candela ribassista prima del movimento che ha rotto la struttura.
    """
    # Logica semplificata per Breaker Block
    # Richiede un\"analisi più profonda della struttura per essere accurata
    return None

def identifica_mitigation_block(candele: List[Candela], index: int, direzione: str) -> Optional[POI]:
    """
    Identifica un Mitigation Block (MB).
    Un MB rialzista si forma dopo un movimento ribassista che non riesce a rompere una struttura rialzista, 
    ed è l\"ultima candela ribassista prima del movimento che ha testato il low.
    """
    # Logica semplificata per Mitigation Block
    # Richiede un\"analisi più profonda della struttura per essere accurata
    return None

def identifica_wick_non_mitigata(candele: List[Candela], index: int, direzione: str) -> Optional[POI]:
    """
    Identifica una Wick Non Mitigata (Unmitigated Wick).
    """
    # Logica semplificata per Wick Non Mitigata
    # Richiede un\"analisi più profonda per essere accurata
    return None

def identifica_tutti_poi(candele: List[Candela], timeframe: str) -> List[POI]:
    all_poi = []
    for i in range(len(candele)):
        # Order Blocks
        ob_bullish = identifica_order_block(candele, i, "Bullish")
        if ob_bullish: all_poi.append(ob_bullish)
        ob_bearish = identifica_order_block(candele, i, "Bearish")
        if ob_bearish: all_poi.append(ob_bearish)

        # Fair Value Gaps
        fvg_bullish = identifica_fair_value_gap(candele, i, "Bullish")
        if fvg_bullish: all_poi.append(fvg_bullish)
        fvg_bearish = identifica_fair_value_gap(candele, i, "Bearish")
        if fvg_bearish: all_poi.append(fvg_bearish)

        # Breaker Blocks (semplificato)
        bb_bullish = identifica_breaker_block(candele, i, "Bullish")
        if bb_bullish: all_poi.append(bb_bullish)
        bb_bearish = identifica_breaker_block(candele, i, "Bearish")
        if bb_bearish: all_poi.append(bb_bearish)

        # Mitigation Blocks (semplificato)
        mb_bullish = identifica_mitigation_block(candele, i, "Bullish")
        if mb_bullish: all_poi.append(mb_bullish)
        mb_bearish = identifica_mitigation_block(candele, i, "Bearish")
        if mb_bearish: all_poi.append(mb_bearish)

        # Unmitigated Wicks (semplificato)
        uw_bullish = identifica_wick_non_mitigata(candele, i, "Bullish")
        if uw_bullish: all_poi.append(uw_bullish)
        uw_bearish = identifica_wick_non_mitigata(candele, i, "Bearish")
        if uw_bearish: all_poi.append(uw_bearish)

    print(f"[DEBUG] Identificati {len(all_poi)} POI su timeframe {timeframe}")
    return all_poi

def is_price_in_poi_zone(price: float, poi: POI) -> bool:
    return poi.prezzo_di_attivazione_bottom <= price <= poi.prezzo_di_attivazione_top

def has_taken_liquidity(poi: POI, candele: List[Candela], lookback_candles: int = 10) -> bool:
    """
    Verifica se il POI ha preso liquidità prima o durante la sua formazione.
    Regola 2: Il POI deve aver preso liquidità.
    """
    poi_index = candele.index(poi.candela_di_riferimento)
    start_index = max(0, poi_index - lookback_candles)
    end_index = min(len(candele), poi_index + 1) # Includi la candela del POI

    relevant_candele = candele[start_index:end_index]

    if poi.direzione == "Bullish": # POI rialzista, cerca liquidità sell-side (sotto i lows)
        # Cerca un low che sia stato rotto prima o durante la formazione del POI
        for i in range(len(relevant_candele) - 1):
            if relevant_candele[i].low < poi.candela_di_riferimento.low: # Se c\"è un low rotto
                return True
    elif poi.direzione == "Bearish": # POI ribassista, cerca liquidità buy-side (sopra gli highs)
        # Cerca un high che sia stato rotto prima o durante la formazione del POI
            if relevant_candele[i].high > poi.candela_di_riferimento.high: # Se c\"è un high rotto
                return True
    return False

def is_mitigated(poi: POI, candele: List[Candela], current_candle_index: int, tolerance_pct: float = 0.002) -> bool:
    """
    Verifica se il POI è stato mitigato.
    Regola 3: Il POI non deve essere mitigato.
    Un POI è mitigato se il prezzo lo ha attraversato completamente dopo la sua formazione.
    Ho rilassato la condizione di mitigazione: il prezzo deve chiudere *significativamente* oltre la zona del POI.
    """
    poi_index = candele.index(poi.candela_di_riferimento)
    
    # Controlla solo le candele *dopo* la candela di riferimento del POI fino alla candela corrente
    for i in range(poi_index + 1, current_candle_index + 1):
        candela = candele[i]
        
        if poi.direzione == "Bullish": # POI rialzista (zona bassa)
            # Mitigato se il prezzo chiude significativamente sotto il bottom del POI
            if candela.close <= poi.prezzo_di_attivazione_bottom * (1 - tolerance_pct):
                return True
            # O se il prezzo ha attraversato completamente il POI verso l\"alto e poi è tornato indietro
            if candela.high > poi.prezzo_di_attivazione_top and candela.close < poi.prezzo_di_attivazione_bottom:
                return True

        elif poi.direzione == "Bearish": # POI ribassista (zona alta)
            # Mitigato se il prezzo chiude significativamente sopra il top del POI
            if candela.close >= poi.prezzo_di_attivazione_top * (1 + tolerance_pct):
                return True
            # O se il prezzo ha attraversato completamente il POI verso il basso e poi è tornato indietro
            if candela.low < poi.prezzo_di_attivazione_bottom and candela.close > poi.prezzo_di_attivazione_top:
                return True
                
    return False

def filtra_poi_validi(all_poi: List[POI], candele_main_tf: List[Candela], current_candle_index: int, poi_validity_bars: int = 50) -> List[POI]:
    valid_poi = []
    current_timestamp = candele_main_tf[current_candle_index].timestamp
    
    for poi in all_poi:
        # Regola 1: Validità temporale (anzianità)
        age_minutes = (current_timestamp - poi.candela_di_riferimento.timestamp).total_seconds() / 60
        # Converti poi_validity_bars in minuti basandoti sul timeframe del POI
        # Assumiamo che il timeframe del POI sia in minuti (es. M5 = 5 minuti)
        # Questa logica andrebbe affinata per H1, H4, D1, ecc.
        if poi.timeframe.startswith("M"):
            tf_minutes = int(poi.timeframe[1:])
        elif poi.timeframe.startswith("H"):
            tf_minutes = int(poi.timeframe[1:]) * 60
        elif poi.timeframe.startswith("D"):
            tf_minutes = 24 * 60
        else:
            tf_minutes = 1 # Fallback per timeframe non riconosciuti

        max_age_minutes = poi_validity_bars * tf_minutes

        if age_minutes > max_age_minutes:
            print(f"[DEBUG] POI scartato per anzianità: {poi.tipo} {poi.direzione} su {poi.timeframe} (vecchio di {age_minutes:.2f} min, max {max_age_minutes} min) - Ref={poi.candela_di_riferimento.timestamp}")
            continue

        # Regola 2: Liquidità presa
        # Questa funzione necessita di un elenco di candele per il timeframe del POI
        # Per ora, usiamo le candele del timeframe principale per semplicità, ma andrebbe migliorato
        # per usare le candele del timeframe specifico del POI.
        # if not has_taken_liquidity(poi, candele_main_tf): # Usare candele del TF del POI
        #     print(f"[DEBUG] POI scartato: {poi.tipo} {poi.direzione} su {poi.timeframe} (Motivo: non ha preso liquidità) - Ref={poi.candela_di_riferimento.timestamp}")
        #     continue

        # Regola 3: Non mitigato
        if is_mitigated(poi, candele_main_tf, current_candle_index): # Usare candele del TF del POI
            print(f"[DEBUG] POI scartato: {poi.tipo} {poi.direzione} su {poi.timeframe} (Motivo: mitigato) - Ref={poi.candela_di_riferimento.timestamp}")
            continue

        # Regola 4: Prezzo attuale nella zona del POI (o molto vicino)
        # Questo è un filtro per l\"ingresso, non per la validità del POI in sé.
        # Spostato nella logica di generazione del segnale.

        valid_poi.append(poi)

    print(f"[DEBUG] POI validi dopo filtraggio: {len(valid_poi)}/{len(all_poi)}")
    return valid_poi


# -----------------------------------------------------------------------------
# CLASSE STRATEGIA DI TRADING
# -----------------------------------------------------------------------------
class TradingStrategy:
    def __init__(self, symbol: str, main_timeframe: str, ohlcv_data: Dict[str, pd.DataFrame], params: dict):
        self.symbol = symbol
        self.main_timeframe = main_timeframe
        self.ohlcv_data = ohlcv_data
        self.params = params
        self.signals = pd.DataFrame(columns=["timestamp", "signal", "price", "stop_loss", "take_profit", "risk_reward_ratio", "lot_size"])
        self.open_trades = [] # Lista per tenere traccia dei trade aperti

        self.candele_multi_timeframe = {}
        for tf_name, df in ohlcv_data.items():
            if not df.empty:
                self.candele_multi_timeframe[tf_name] = [
                    Candela(row["time"], row["open"], row["high"], row["low"], row["close"])
                    for index, row in df.iterrows()
                ]
            else:
                self.candele_multi_timeframe[tf_name] = []
        
        self.main_candele = self.candele_multi_timeframe.get(self.main_timeframe, [])
        if not self.main_candele:
            print(f"[ERRORE] Dati non disponibili per il timeframe principale: {self.main_timeframe}")
            return

        self.current_candle_index = len(self.main_candele) - 1
        self.current_price = self.main_candele[-1].close

        # Parametri della strategia
        self.swing_period = params.get("swing_period", 3)
        self.structure_lookback = params.get("structure_lookback", 20)
        self.liquidity_threshold = params.get("liquidity_threshold", 0.001)
        self.poi_validity_bars = params.get("poi_validity_bars", 50)
        self.range_min_size = params.get("range_min_size", 0.005)
        self.risk_reward_ratio = params.get("risk_reward_ratio", 2.0)
        self.max_drawdown_pct = params.get("max_drawdown_pct", 5.0)
        self.session_asian_start = params.get("session_asian_start", 23)
        self.session_asian_end = params.get("session_asian_end", 6)
        self.session_london_start = params.get("session_london_start", 7)
        self.session_london_end = params.get("session_london_end", 10)
        self.session_ny_start = params.get("session_ny_start", 14)
        self.session_ny_end = params.get("session_ny_end", 17)
        self.max_open_trades_per_symbol = params.get("max_open_trades_per_symbol", 1) # Nuovo parametro

    def calculate_lot_size(self, risk_per_trade_pct: float, stop_loss_pips: float) -> float:
        """
        Calcola il lot size in base al rischio per trade e alla distanza dello Stop Loss.
        """
        if stop_loss_pips == 0: # Evita divisione per zero
            print("[DEBUG] Stop Loss in pips è zero, impossibile calcolare lot size.")
            return 0.0

        account_info = mt5.account_info()
        if account_info is None:
            print("[ERRORE] Impossibile recuperare informazioni account per calcolo lot size.")
            return 0.0

        balance = account_info.balance
        risk_amount = balance * (risk_per_trade_pct / 100)

        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            print(f"[ERRORE] Impossibile recuperare informazioni simbolo {self.symbol} per calcolo lot size.")
            return 0.0

        # Per calcolare il valore di un pip per il simbolo specifico
        # Usiamo mt5.symbol_info_tick per ottenere il prezzo corrente e calcolare il valore del punto
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            print(f"[ERRORE] Impossibile ottenere tick per {self.symbol} per calcolo lot size.")
            return 0.0

        # Calcola il valore di un punto (tick_size) in valuta del conto per 1 lotto
        # Questo è un calcolo più accurato rispetto a un valore fisso
        point_value = mt5.symbol_info(self.symbol).trade_tick_value
        point_size = mt5.symbol_info(self.symbol).trade_tick_size

        # Se lo SL è in punti, convertiamo in valuta del conto
        cost_per_lot_at_sl = stop_loss_pips * (point_value / point_size) # Costo per lotto per la distanza SL

        if cost_per_lot_at_sl == 0:
            print("[DEBUG] Costo per lotto a SL è zero, impossibile calcolare lot size.")
            return 0.0

        lot = risk_amount / cost_per_lot_at_sl

        # Assicurati che il lot size sia valido per il broker (min, max, step)
        min_lot = symbol_info.volume_min
        max_lot = symbol_info.volume_max
        step_lot = symbol_info.volume_step

        lot = max(min_lot, round(lot / step_lot) * step_lot) # Arrotonda al passo più vicino
        lot = min(lot, max_lot)
        
        print(f"[DEBUG] Calcolo Lot Size: Balance={balance}, RiskAmount={risk_amount}, SL_Pips={stop_loss_pips}, CostPerLotAtSL={cost_per_lot_at_sl}, CalcolatoLot={lot}")

        return lot

    def generate_signals(self):
        # Ottieni le candele per il timeframe principale
        main_candele = self.candele_multi_timeframe.get(self.main_timeframe)
        if not main_candele:
            print(f"[ERRORE] Dati non disponibili per il timeframe principale: {self.main_timeframe}")
            return

        # Aggiorna il prezzo corrente con l\"ultima candela
        self.current_price = main_candele[-1].close

        # 1. Identificazione Swing Points e Struttura
        all_swing_points = {}
        for tf_name, candele_list in self.candele_multi_timeframe.items():
            if candele_list:
                all_swing_points[tf_name] = identifica_swing_points_frattali(candele_list, period=self.swing_period, timeframe=tf_name)

        # Analisi della struttura sul timeframe principale
        main_tf_swing_points = all_swing_points.get(self.main_timeframe, [])
        trend, bos_high, bos_low, _ = analizza_struttura_e_bos_frattale(main_tf_swing_points)
        print(f"[DEBUG] Trend su {self.main_timeframe}: {trend}")

        # 2. Identificazione POI (Order Blocks, FVG, Breaker, Mitigation, Wick)
        all_poi_raw = []
        for tf_name, candele_list in self.candele_multi_timeframe.items():
            if candele_list:
                poi_tf = identifica_tutti_poi(candele_list, tf_name)
                all_poi_raw.extend(poi_tf)

        # 3. Filtraggio POI
        # Filtra i POI validi (anzianità, liquidità presa, non mitigato)
        valid_poi = filtra_poi_validi(all_poi_raw, main_candele, self.current_candle_index, self.poi_validity_bars)
        print(f"[DEBUG] Totale POI validi identificati su tutti i timeframe: {len(valid_poi)}")

        # 4. Logica di Trading (Generazione Segnali)
        signal = None
        stop_loss = 0.0
        take_profit = 0.0
        lot_size = 0.0
        poi_riferimento = None

        # Gestione del numero massimo di posizioni aperte
        open_positions = mt5.positions_get(symbol=self.symbol)
        num_open_positions = len(open_positions) if open_positions else 0

        if num_open_positions >= self.max_open_trades_per_symbol:
            print(f"[DEBUG] Numero massimo di posizioni aperte ({num_open_positions}) raggiunto per {self.symbol}. Nessun nuovo segnale generato.")
            return

        # Ordina i POI validi per rilevanza (es. i più recenti o quelli più vicini al prezzo)
        # Per semplicità, li ordiniamo per timestamp decrescente (dal più recente al più vecchio)
        valid_poi.sort(key=lambda p: p.candela_di_riferimento.timestamp, reverse=True)

        for poi in valid_poi:
            # Regola 4: Prezzo attuale nella zona del POI (o molto vicino)
            # Consideriamo un buffer intorno al POI per l'ingresso
            buffer_pct = 0.0005 # 0.05% di buffer
            poi_top_buffered = poi.prezzo_di_attivazione_top * (1 + buffer_pct)
            poi_bottom_buffered = poi.prezzo_di_attivazione_bottom * (1 - buffer_pct)

            if not (poi_bottom_buffered <= self.current_price <= poi_top_buffered):
                print(f"[DEBUG] POI scartato: {poi.tipo} {poi.direzione} su {poi.timeframe} (Motivo: prezzo {self.current_price:.5f} non nella zona del POI [{poi.prezzo_di_attivazione_bottom:.5f}-{poi.prezzo_di_attivazione_top:.5f}]) - Ref={poi.candela_di_riferimento.timestamp}")
                continue

            # Regola 5: Allineamento con il trend del timeframe principale
            if trend == "Bullish" and poi.direzione == "Bearish":
                print(f"[DEBUG] POI scartato: {poi.tipo} {poi.direzione} su {poi.timeframe} (Motivo: disallineamento trend Bullish) - Ref={poi.candela_di_riferimento.timestamp}")
                continue
            if trend == "Bearish" and poi.direzione == "Bullish":
                print(f"[DEBUG] POI scartato: {poi.tipo} {poi.direzione} su {poi.timeframe} (Motivo: disallineamento trend Bearish) - Ref={poi.candela_di_riferimento.timestamp}")
                continue
            
            # Se arriviamo qui, il POI è valido e allineato con il trend
            poi_riferimento = poi
            
            # Calcolo Stop Loss (SL)
            if poi.direzione == "Bullish": # Per BUY, SL sotto il POI
                stop_loss = poi.prezzo_di_attivazione_bottom * 0.999 # Un po' sotto il bottom del POI
            elif poi.direzione == "Bearish": # Per SELL, SL sopra il POI
                stop_loss = poi.prezzo_di_attivazione_top * 1.001 # Un po' sopra il top del POI

            # Calcolo Take Profit (TP)
            # Prima cerca liquidità target, altrimenti usa RR
            target_liquidity = None
            liquidita_levels = identifica_liquidita_swing_points(main_candele) # Liquidità sul TF principale

            if poi.direzione == "Bullish": # Per BUY, cerca liquidità buy-side sopra il prezzo
                # Cerca il livello di liquidità buy-side più vicino e sopra il prezzo corrente
                potential_tps = [lvl for lvl in liquidita_levels["buy_side"] if lvl > self.current_price]
                if potential_tps:
                    target_liquidity = min(potential_tps) # Il più basso tra quelli sopra
            elif poi.direzione == "Bearish": # Per SELL, cerca liquidità sell-side sotto il prezzo
                # Cerca il livello di liquidità sell-side più vicino e sotto il prezzo corrente
                potential_tps = [lvl for lvl in liquidita_levels["sell_side"] if lvl < self.current_price]
                if potential_tps:
                    target_liquidity = max(potential_tps) # Il più alto tra quelli sotto

            if target_liquidity is not None:
                take_profit = target_liquidity
                print(f"[DEBUG] TP basato su liquidità target: {take_profit}")
            else:
                # Se non trova liquidità target, calcola TP basato su RR
                risk_pips = abs(self.current_price - stop_loss)
                if risk_pips > 0:
                    if poi.direzione == "Bullish": # BUY
                        take_profit = self.current_price + (risk_pips * self.risk_reward_ratio)
                    elif poi.direzione == "Bearish": # SELL
                        take_profit = self.current_price - (risk_pips * self.risk_reward_ratio)
                    print(f"[DEBUG] TP basato su RR ({self.risk_reward_ratio}): {take_profit}")
                else:
                    print("[DEBUG] Risk Pips è zero, impossibile calcolare TP basato su RR.")
                    continue # Salta questo POI se SL è troppo vicino

            # Calcolo RR effettivo
            if poi.direzione == "Bullish": # BUY
                if (take_profit - self.current_price) > 0 and (self.current_price - stop_loss) > 0:
                    risk_reward = (take_profit - self.current_price) / (self.current_price - stop_loss)
                else:
                    risk_reward = -999 # Valore per indicare RR non valido
            elif poi.direzione == "Bearish": # SELL
                if (self.current_price - take_profit) > 0 and (stop_loss - self.current_price) > 0:
                    risk_reward = (self.current_price - take_profit) / (stop_loss - self.current_price)
                else:
                    risk_reward = -999 # Valore per indicare RR non valido
            
            print(f"[DEBUG] Calcolo SL/TP: Prezzo={self.current_price:.5f}, SL={stop_loss:.5f}, TP={take_profit:.5f}, Risk={abs(self.current_price - stop_loss):.5f}, RR={risk_reward:.2f}")

            # Validazione finale SL/TP per evitare errori 10016 (Invalid stops)
            tick_size = mt5.symbol_info(self.symbol).trade_tick_size
            min_distance_sl_tp = 10 * tick_size # Esempio: 10 tick di distanza minima

            is_sl_valid = False
            is_tp_valid = False

            if poi.direzione == "Bullish": # BUY
                if stop_loss < self.current_price - min_distance_sl_tp: # SL deve essere sotto il prezzo e a distanza minima
                    is_sl_valid = True
                if take_profit > self.current_price + min_distance_sl_tp: # TP deve essere sopra il prezzo e a distanza minima
                    is_tp_valid = True
            elif poi.direzione == "Bearish": # SELL
                if stop_loss > self.current_price + min_distance_sl_tp: # SL deve essere sopra il prezzo e a distanza minima
                    is_sl_valid = True
                if take_profit < self.current_price - min_distance_sl_tp: # TP deve essere sotto il prezzo e a distanza minima
                    is_tp_valid = True
            
            if not is_sl_valid or not is_tp_valid:
                print(f"[DEBUG] POI scartato: {poi.tipo} {poi.direzione} su {poi.timeframe} (Motivo: SL/TP non validi o troppo vicini al prezzo) - SL Valid: {is_sl_valid}, TP Valid: {is_tp_valid}")
                continue

            # Se il RR è inferiore a quello desiderato, scarta il segnale
            if risk_reward < self.risk_reward_ratio:
                print(f"[DEBUG] POI scartato: {poi.tipo} {poi.direzione} su {poi.timeframe} (Motivo: RR {risk_reward:.2f} < {self.risk_reward_ratio}) - Ref={poi.candela_di_riferimento.timestamp}")
                continue

            # Calcola Lot Size Dinamico
            # Converti SL in pips per il calcolo del lot size
            # Per BTCUSD, 1 punto = 1 tick_size
            sl_pips = abs(self.current_price - stop_loss) / tick_size
            lot_size = self.calculate_lot_size(self.params.get("risk_per_trade_pct", 1.0), sl_pips)
            
            if lot_size == 0.0:
                print(f"[DEBUG] POI scartato: {poi.tipo} {poi.direzione} su {poi.timeframe} (Motivo: Lot Size calcolato è zero) - Ref={poi.candela_di_riferimento.timestamp}")
                continue

            # Se tutte le condizioni sono soddisfatte, genera il segnale
            signal = poi.direzione.upper() # 

