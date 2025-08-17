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
    
    print(f"[DEBUG] Liquidità identificata - Buy Side: {len(liquidita["buy_side"])}, Sell Side: {len(liquidita["sell_side"])})
")
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
            # Sessione che attraversa la mezzanotte (es. 23-6)
            if hour >= session_asian_start or hour < session_asian_end:
                asian_candele.append(candela)
                
    if not asian_candele:
        print(f"[DEBUG] Nessuna candela trovata per la sessione asiatica su {timeframe}")
        # Restituisci valori che indicano l'assenza di dati validi
        return {"high": 0.0, "low": 0.0}

    # Aggiorna i valori solo se ci sono candele nella sessione asiatica
    asian_session_high = max(c.high for c in asian_candele)
    asian_session_low = min(c.low for c in asian_candele)
        
    print(f"[DEBUG] Identificata liquidità sessione asiatica su {timeframe}: High={asian_session_high:.5f}, Low={asian_session_low:.5f}")
    return {"high": asian_session_high, "low": asian_session_low}

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
                    key_level_ohlc={"open": candela_prec.open, "high": candela_prec.high,
                                    "low": candela_prec.low, "close": candela_prec.close},
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
                    key_level_ohlc={"open": candela_prec.open, "high": candela_prec.high,
                                    "low": candela_prec.low, "close": candela_prec.close},
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

def filtra_poi_validi(lista_poi: List[POI], swing_points: List[SwingPoint], candele: List[Candela], all_liquidita: Dict[str, List[float]]) -> List[POI]:
    """
    Filtra i POI secondo le regole della strategia Eclipse.
    
    REGOLA 2: Deve aver "preso" liquidità
    REGOLA 3: Non deve essere mitigato
    """
    poi_validi = []
    
    for poi in lista_poi:
        has_taken_liquidity = False
        is_mitigated = False
        reason_discarded = []

        # REGOLA 2: Deve aver "preso" liquidità
        # Verifica se il POI è associato a un movimento che ha rotto liquidità
        # Ora considera tutte le forme di liquidità
        # Aggiungo un lookback per la liquidità presa, non solo la candela di riferimento
        # Estendo il lookback per la liquidità presa per timeframe più alti
        lookback_candles_for_liquidity = 10 # Guarda le ultime 10 candele prima e dopo il POI
        poi_index = candele.index(poi.candela_di_riferimento)
        relevant_candles = candele[max(0, poi_index - lookback_candles_for_liquidity) : min(len(candele), poi_index + lookback_candles_for_liquidity + 1)]

        if poi.direzione == "Bearish": # Per un POI bearish, cerchiamo liquidità buy-side presa
            for liq_level in all_liquidita.get("buy_side", []):
                # Se il prezzo ha superato un livello di liquidità buy-side in una delle candele rilevanti
                if any(c.high > liq_level for c in relevant_candles):
                    has_taken_liquidity = True
                    break
        elif poi.direzione == "Bullish": # Per un POI bullish, cerchiamo liquidità sell-side presa
            for liq_level in all_liquidita.get("sell_side", []):
                # Se il prezzo ha superato un livello di liquidità sell-side in una delle candele rilevanti
                if any(c.low < liq_level for c in relevant_candles):
                    has_taken_liquidity = True
                    break
        
        if not has_taken_liquidity:
            reason_discarded.append("non ha preso liquidità")

        # REGOLA 3: Non deve essere mitigato
        # Verifica se il prezzo è tornato nella zona del POI dopo la sua formazione
        # Rilasso la condizione di mitigazione: un POI è mitigato solo se il prezzo chiude
        # completamente oltre la sua zona, non solo se la tocca.
        
        # Candele successive al POI
        subsequent_candles = candele[candele.index(poi.candela_di_riferimento) + 1:]

        for candela in subsequent_candles:
            if poi.direzione == "Bearish":
                # Per un POI bearish, è mitigato se il prezzo torna sopra il top
                # Consideriamo mitigato se il close della candela successiva è >= top del POI
                # Rilassamento: il prezzo deve chiudere *significativamente* oltre il POI per essere mitigato
                # Usiamo una percentuale maggiore, ad esempio 0.1% o 0.2% oltre il bordo
                mitigation_threshold = poi.prezzo_di_attivazione_top * 1.002 # 0.2% oltre il top
                if candela.close >= mitigation_threshold:
                    is_mitigated = True
                    poi.e_mitigato = True
                    reason_discarded.append("mitigato (prezzo tornato sopra il top)")
                    break
            elif poi.direzione == "Bullish":
                # Per un POI bullish, è mitigato se il prezzo torna sotto il bottom
                # Consideriamo mitigato se il close della candela successiva è <= bottom del POI
                # Rilassamento: il prezzo deve chiudere *significativamente* oltre il POI per essere mitigato
                mitigation_threshold = poi.prezzo_di_attivazione_bottom * 0.998 # 0.2% sotto il bottom
                if candela.close <= mitigation_threshold:
                    is_mitigated = True
                    poi.e_mitigato = True
                    reason_discarded.append("mitigato (prezzo tornato sotto il bottom)")
                    break

        # Aggiungi il POI solo se ha preso liquidità e non è mitigato
        if has_taken_liquidity and not is_mitigated:
            poi_validi.append(poi)
        else:
            print(f"[DEBUG] POI scartato: {poi.tipo} {poi.direzione} su {poi.timeframe} (Motivo: {", ".join(reason_discarded)}) - Ref={poi.candela_di_riferimento.timestamp}")

    print(f"[DEBUG] POI validi dopo filtraggio: {len(poi_validi)}/{len(lista_poi)}")
    return poi_validi


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
        self.equal_highs_lows_multi_timeframe = {}
        self.trendline_liquidity_multi_timeframe = {}
        self.asian_session_liquidity_multi_timeframe = {}

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

    def generate_signals(self):
        # Ottieni le candele per il timeframe principale
        main_candele = self.candele_multi_timeframe.get(self.main_timeframe)
        if not main_candele:
            print(f"[ERRORE] Dati non disponibili per il timeframe principale: {self.main_timeframe}")
            return

        print(f"[DEBUG] Analisi su timeframe principale: {self.main_timeframe} con {len(main_candele)} candele")

        # 1. Identificazione Swing Points e Liquidità su tutti i timeframe
        for tf_name, candele_tf in self.candele_multi_timeframe.items():
            if candele_tf:
                swing_period = self.params.get("swing_period", 3)
                swing_points = identifica_swing_points_frattali(candele_tf, swing_period, tf_name)
                self.swing_points_multi_timeframe[tf_name] = swing_points
                
                # Identifica liquidità per questo timeframe (swing points)
                liquidita_sp = identifica_liquidita_swing_points(swing_points)
                
                # Identifica Massimi/Minimi Uguali
                equal_levels = identifica_equal_highs_lows(candele_tf, timeframe=tf_name)
                self.equal_highs_lows_multi_timeframe[tf_name] = equal_levels
                
                # Identifica Trendline Liquidity
                trendline_liq = identifica_trendline_liquidity(candele_tf, tf_name)
                self.trendline_liquidity_multi_timeframe[tf_name] = trendline_liq

                # Identifica Asian Session Liquidity (solo se i parametri sono disponibili)
                asian_liq = {"high": 0.0, "low": 0.0}
                if "session_asian_start" in self.params and "session_asian_end" in self.params:
                    asian_liq = identifica_asian_session_liquidity(
                        candele_tf, 
                        self.params["session_asian_start"],
                        self.params["session_asian_end"],
                        tf_name
                    )
                self.asian_session_liquidity_multi_timeframe[tf_name] = asian_liq

                # Combina tutte le forme di liquidità identificate per il timeframe
                combined_liquidita = {
                    "buy_side": liquidita_sp["buy_side"] + equal_levels["equal_highs"] + trendline_liq["buy_side"],
                    "sell_side": liquidita_sp["sell_side"] + equal_levels["equal_lows"] + trendline_liq["sell_side"]
                }
                # Aggiungi i massimi/minimi della sessione asiatica come liquidità esterna
                if asian_liq["high"] > 0: # Solo se identificata
                    combined_liquidita["buy_side"].append(asian_liq["high"])
                if asian_liq["low"] > 0: # Solo se identificata
                    combined_liquidita["sell_side"].append(asian_liq["low"])

                self.liquidita_multi_timeframe[tf_name] = combined_liquidita

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
                
                # Passa tutte le liquidità identificate per il timeframe corrente
                all_liquidita_tf = self.liquidita_multi_timeframe.get(tf_name, {})
                valid_tf_poi = filtra_poi_validi(current_tf_poi, swing_points_tf, candele_tf, all_liquidita_tf)
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
            poi_validity_bars = self.params.get("poi_validity_bars", 50)
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
                print(f"[DEBUG] POI scartato: {poi.tipo} {poi.direzione} su {poi.timeframe} (Motivo: Prezzo corrente {current_price:.5f} non nella zona del POI [{poi.prezzo_di_attivazione_bottom:.5f}-{poi.prezzo_di_attivazione_top:.5f}])")
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
                    # Verifica se c\"è liquidità sopra il prezzo corrente
                    for liq_level in tf_liquidita["buy_side"]:
                        if liq_level > current_price:
                            has_liquidity_target = True
                            break
            elif poi.direzione == "Bearish":
                # Per un trade SELL, cerchiamo liquidità sell-side (sotto swing lows)
                tf_liquidita = self.liquidita_multi_timeframe.get(poi.timeframe, {})
                if tf_liquidita.get("sell_side", []): # Corretto da sell_level a sell_side
                    # Verifica se c\"è liquidità sotto il prezzo corrente
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
                # Per un TP BUY, vogliamo un livello di liquidità più alto del prezzo corrente
                # e il più vicino possibile (quindi il più basso tra quelli più alti)
                for liq_level in tf_liquidita.get("buy_side", []):
                    if liq_level > current_price:
                        if nearest_liquidity is None or liq_level < nearest_liquidity:
                            nearest_liquidity = liq_level
                
                if nearest_liquidity:
                    # TP basato sulla liquidità più vicina
                    take_profit = nearest_liquidity # Non sottraggo/aggiungo buffer qui, lo faccio dopo
                else:
                    # TP basato sul risk/reward ratio se non c\"è liquidità vicina
                    take_profit = current_price + (risk * self.params.get("risk_reward_ratio", 2.0))
                
                # Aggiungo un piccolo buffer al TP per BUY per assicurare che sia sopra il prezzo di entrata
                if take_profit <= current_price: # Se il TP calcolato è sotto o uguale al prezzo corrente, lo aggiusto
                    take_profit = current_price + (risk * self.params.get("risk_reward_ratio", 2.0))

                # Validazione finale di SL e TP per BUY
                # SL deve essere minore del prezzo corrente e TP maggiore del prezzo corrente
                # E TP deve essere maggiore di SL
                if not (stop_loss < current_price and take_profit > current_price and take_profit > stop_loss):
                    print(f"[DEBUG] Segnale BUY scartato: SL/TP non validi. SL: {stop_loss:.5f}, TP: {take_profit:.5f}, Current: {current_price:.5f}")
                    signal = None # Annulla il segnale se SL/TP non validi
                    continue

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
                # Per un TP SELL, vogliamo un livello di liquidità più basso del prezzo corrente
                # e il più vicino possibile (quindi il più alto tra quelli più bassi)
                for liq_level in tf_liquidita.get("sell_side", []):
                    if liq_level < current_price:
                        if nearest_liquidity is None or liq_level > nearest_liquidity:
                            nearest_liquidity = liq_level
                
                if nearest_liquidity:
                    # TP basato sulla liquidità più vicina
                    take_profit = nearest_liquidity # Non sottraggo/aggiungo buffer qui, lo faccio dopo
                else:
                    # TP basato sul risk/reward ratio se non c\"è liquidità vicina
                    take_profit = current_price - (risk * self.params.get("risk_reward_ratio", 2.0))
                
                # Aggiungo un piccolo buffer al TP per SELL per assicurare che sia sotto il prezzo di entrata
                if take_profit >= current_price: # Se il TP calcolato è sopra o uguale al prezzo corrente, lo aggiusto
                    take_profit = current_price - (risk * self.params.get("risk_reward_ratio", 2.0))

                # Validazione finale di SL e TP per SELL
                # SL deve essere maggiore del prezzo corrente e TP minore del prezzo corrente
                # E TP deve essere minore di SL
                if not (stop_loss > current_price and take_profit < current_price and take_profit < stop_loss):
                    print(f"[DEBUG] Segnale SELL scartato: SL/TP non validi. SL: {stop_loss:.5f}, TP: {take_profit:.5f}, Current: {current_price:.5f}")
                    signal = None # Annulla il segnale se SL/TP non validi
                    continue

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
