from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import pandas as pd
import numpy as np

# =============================================================================
# ECLIPSE STRATEGY — QUASIMODO / POI / RANGE with rich DEBUG
# Compatible with existing run_backtest.py (expects Strategy.generate_signals())
# and ready for live integration (see BrokerInterface + next_signal()).
# =============================================================================

# -----------------------------
# Data structures
# -----------------------------
@dataclass
class Candle:
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float

    @property
    def range(self) -> float:
        return float(self.high - self.low)

    @property
    def body(self) -> float:
        return float(abs(self.close - self.open))

    @property
    def upper_wick(self) -> float:
        return float(self.high - max(self.open, self.close))

    @property
    def lower_wick(self) -> float:
        return float(min(self.open, self.close) - self.low)


@dataclass
class POI:
    tipo: str  # "Orderblock", "Inefficiency", "Breaker", "HiddenBase", "Wick"
    direzione: str  # "Bullish" | "Bearish"
    candela_di_riferimento: Candle
    prezzo_di_attivazione_top: float
    prezzo_di_attivazione_bottom: float
    key_level_ohlc: Dict[str, float]
    e_mitigato: bool = False
    timeframe: str = "HTF"

    def contains(self, price: float) -> bool:
        return self.prezzo_di_attivazione_bottom <= price <= self.prezzo_di_attivazione_top


@dataclass
class RangeMercato:
    strong_high: Candle
    strong_low: Candle
    weak_highs: List[Candle] = field(default_factory=list)
    weak_lows: List[Candle] = field(default_factory=list)
    liquidita_esterna_buy_side: float = 0.0
    liquidita_esterna_sell_side: float = 0.0
    liquidita_interna: List[POI] = field(default_factory=list)
    timeframe: str = "HTF"


@dataclass
class TradePlan:
    tipo: str  # BUY | SELL
    entry: float
    stop: float
    take_profit: float
    confidence: float
    poi: Optional[POI] = None


# -----------------------------
# Minimal broker interface (for live)
# -----------------------------
class BrokerInterface:
    """Adapter da implementare in run_live per MT5 (o mock nel backtest)."""
    def get_candles(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        raise NotImplementedError

    def get_price(self, symbol: str) -> float:
        raise NotImplementedError


# =============================================================================
# Strategy class
# =============================================================================
class Strategy:
    def __init__(self, data: Optional[pd.DataFrame] = None, params: Optional[dict] = None):
        self.data = data  # singolo TF per backtest
        self.params = params or {}
        self.symbol = self.params.get("symbol", "EURUSD")
        self.timeframe_htf = self.params.get("TIMEFRAME_HTF", "H4")
        self.timeframe_ltf = self.params.get("TIMEFRAME_LTF", "M15")
        self.timeframe_entry = self.params.get("TIMEFRAME_ENTRY", "M1")
        self.prev_trend = None

        # Caches
        self.swing_high_idx: List[int] = []
        self.swing_low_idx: List[int] = []
        self.market_structure: Optional[pd.DataFrame] = None

    # -------------------------
    # Utilities
    # -------------------------
    def _df_to_candles(self, df: pd.DataFrame) -> List[Candle]:
        return [
            Candle(timestamp=idx, open=float(r.open), high=float(r.high), low=float(r.low), close=float(r.close))
            for idx, r in df.iterrows()
        ]

    # =============================================================================
    # SEZIONE 1: SWING POINTS
    # =============================================================================
    def identifica_swing_points(self, candles: List[Candle]) -> Tuple[List[Candle], List[Candle]]:
        swing_highs: List[Candle] = []
        swing_lows: List[Candle] = []
        for i in range(1, len(candles) - 1):
            p, c, n = candles[i - 1], candles[i], candles[i + 1]
            if c.high > p.high and c.high > n.high:
                swing_highs.append(c)
            if c.low < p.low and c.low < n.low:
                swing_lows.append(c)
        return swing_highs, swing_lows

    # =============================================================================
    # SEZIONE 2: STRUTTURA + BOS
    # =============================================================================
    def analizza_struttura_e_bos(
        self, swing_highs: List[Candle], swing_lows: List[Candle]
    ) -> Tuple[str, Optional[Candle], Optional[Candle]]:
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return "SIDEWAYS", None, None
        sh_sorted = sorted(swing_highs, key=lambda c: c.timestamp)
        sl_sorted = sorted(swing_lows, key=lambda c: c.timestamp)
        uh, puh = sh_sorted[-1], sh_sorted[-2]
        ul, pul = sl_sorted[-1], sl_sorted[-2]

        trend = "SIDEWAYS"
        bos_high = None
        bos_low = None
        if uh.high > puh.high and ul.low > pul.low:
            trend = "BULLISH"
            bos_high = uh
        elif uh.high < puh.high and ul.low < pul.low:
            trend = "BEARISH"
            bos_low = ul
        # CHOCH opzionale: se stato precedente bull e rompi ultimo low -> bearish
        if self.prev_trend == "BULLISH" and ul.low < pul.low:
            trend = "BEARISH"
        elif self.prev_trend == "BEARISH" and uh.high > puh.high:
            trend = "BULLISH"
        return trend, bos_high, bos_low

    # ----------------------------------------------------------------------------
    # Helpers per ricerca swing prima/dopo un timestamp
    # ----------------------------------------------------------------------------
    @staticmethod
    def _ultimo_low_prima_di(ts: pd.Timestamp, lows: List[Candle]) -> Optional[Candle]:
        cand = [c for c in lows if c.timestamp < ts]
        return max(cand, key=lambda c: c.timestamp) if cand else None

    @staticmethod
    def _primo_low_dopo(ts: pd.Timestamp, lows: List[Candle]) -> Optional[Candle]:
        cand = [c for c in lows if c.timestamp > ts]
        return min(cand, key=lambda c: c.timestamp) if cand else None

    @staticmethod
    def _ultimo_high_prima_di(ts: pd.Timestamp, highs: List[Candle]) -> Optional[Candle]:
        cand = [c for c in highs if c.timestamp < ts]
        return max(cand, key=lambda c: c.timestamp) if cand else None

    @staticmethod
    def _primo_high_dopo(ts: pd.Timestamp, highs: List[Candle]) -> Optional[Candle]:
        cand = [c for c in highs if c.timestamp > ts]
        return min(cand, key=lambda c: c.timestamp) if cand else None

    # =============================================================================
    # SEZIONE 5: RANGE DA QUASIMODO
    # =============================================================================
    def definisci_range_da_quasimodo(self, candles: List[Candle], timeframe: str = "HTF") -> Optional[RangeMercato]:
        print(f"[RANGE] Avvio identificazione Range su timeframe {timeframe}...")
        swing_highs, swing_lows = self.identifica_swing_points(candles)
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            print("[RANGE] Dati insufficienti (swing points) per definire un range.")
            return None

        strong_high = None
        # Scansiona gli swing high a ritroso
        for i in range(len(swing_highs) - 1, 0, -1):
            h_cur = swing_highs[i]
            h_prev = swing_highs[i - 1]
            stop_hunt = h_cur.high > h_prev.high
            bos_ribasso = False
            low_pre = self._ultimo_low_prima_di(h_cur.timestamp, swing_lows)
            if low_pre is not None:
                low_after = self._primo_low_dopo(h_cur.timestamp, swing_lows)
                if low_after and (low_after.low < low_pre.low):
                    bos_ribasso = True
            if stop_hunt and bos_ribasso:
                strong_high = h_cur
                print(f"[RANGE] QM Ribassista (Strong High) a {strong_high.high} @ {strong_high.timestamp}")
                break
        if strong_high is None:
            print("[RANGE] Nessun QM Ribassista (Strong High) trovato.")
            return None

        # Strong Low precedente allo Strong High
        lows_before = [l for l in swing_lows if l.timestamp < strong_high.timestamp]
        strong_low = None
        for i in range(len(lows_before) - 1, 0, -1):
            l_cur = lows_before[i]
            l_prev = lows_before[i - 1]
            stop_hunt = l_cur.low < l_prev.low
            bos_rialzo = False
            high_pre = self._ultimo_high_prima_di(l_cur.timestamp, swing_highs)
            if high_pre is not None:
                high_after = self._primo_high_dopo(l_cur.timestamp, swing_highs)
                if high_after and (high_after.high > high_pre.high):
                    bos_rialzo = True
            if stop_hunt and bos_rialzo:
                strong_low = l_cur
                print(f"[RANGE] QM Rialzista (Strong Low) a {strong_low.low} @ {strong_low.timestamp}")
                break
        if strong_low is None:
            print("[RANGE] Nessun QM Rialzista (Strong Low) valido trovato prima dello Strong High.")
            return None

        r = RangeMercato(
            strong_high=strong_high,
            strong_low=strong_low,
            liquidita_esterna_buy_side=float(strong_high.high),
            liquidita_esterna_sell_side=float(strong_low.low),
            timeframe=timeframe,
        )
        r.weak_highs = [h for h in swing_highs if strong_low.timestamp < h.timestamp < strong_high.timestamp]
        r.weak_lows = [l for l in swing_lows if strong_low.timestamp < l.timestamp < strong_high.timestamp]
        print(
            f"[RANGE] Range valido {timeframe}: Low={r.strong_low.low:.5f}, High={r.strong_high.high:.5f}, "
            f"WeakHighs={len(r.weak_highs)}, WeakLows={len(r.weak_lows)}"
        )
        return r

    # =============================================================================
    # SEZIONE 3-4: POI
    # =============================================================================
    def identifica_tutti_poi(self, candles: List[Candle], timeframe: str = "HTF") -> List[POI]:
        out: List[POI] = []
        for i in range(1, len(candles)):
            prev = candles[i - 1]
            cur = candles[i]

            # OB Bearish: ultima verde prima di forte rossa (semplificato)
            if prev.close > prev.open and cur.close < cur.open and cur.low < prev.low:
                poi = POI(
                    tipo="Orderblock",
                    direzione="Bearish",
                    candela_di_riferimento=prev,
                    prezzo_di_attivazione_top=float(prev.high),
                    prezzo_di_attivazione_bottom=float(prev.open),
                    key_level_ohlc={"open": prev.open, "high": prev.high, "low": prev.low, "close": prev.close},
                    timeframe=timeframe,
                )
                out.append(poi)

            # OB Bullish: ultima rossa prima di forte verde (semplificato)
            if prev.close < prev.open and cur.close > cur.open and cur.high > prev.high:
                poi = POI(
                    tipo="Orderblock",
                    direzione="Bullish",
                    candela_di_riferimento=prev,
                    prezzo_di_attivazione_top=float(prev.open),
                    prezzo_di_attivazione_bottom=float(prev.low),
                    key_level_ohlc={"open": prev.open, "high": prev.high, "low": prev.low, "close": prev.close},
                    timeframe=timeframe,
                )
                out.append(poi)

            # FVG tra i-1 e i+1 usando i
            if i < len(candles) - 1:
                nxt = candles[i + 1]
                # FVG Bullish: max(prev) < min(nxt)
                if prev.high < nxt.low:
                    mid = (prev.high + nxt.low) / 2.0
                    poi = POI(
                        tipo="Inefficiency",
                        direzione="Bullish",
                        candela_di_riferimento=cur,
                        prezzo_di_attivazione_top=float(nxt.low),
                        prezzo_di_attivazione_bottom=float(prev.high),
                        key_level_ohlc={"open": cur.open, "high": cur.high, "low": cur.low, "close": cur.close},
                        timeframe=timeframe,
                    )
                    out.append(poi)
                # FVG Bearish: min(prev) > max(nxt)
                if prev.low > nxt.high:
                    mid = (prev.low + nxt.high) / 2.0
                    poi = POI(
                        tipo="Inefficiency",
                        direzione="Bearish",
                        candela_di_riferimento=cur,
                        prezzo_di_attivazione_top=float(prev.low),
                        prezzo_di_attivazione_bottom=float(nxt.high),
                        key_level_ohlc={"open": cur.open, "high": cur.high, "low": cur.low, "close": cur.close},
                        timeframe=timeframe,
                    )
                    out.append(poi)

            # Wick importante (placeholder — usa 30% della range)
            if cur.upper_wick > cur.range * 0.3 and cur.close < cur.open:
                poi = POI(
                    tipo="Wick",
                    direzione="Bearish",
                    candela_di_riferimento=cur,
                    prezzo_di_attivazione_top=float(cur.high),
                    prezzo_di_attivazione_bottom=float(max(cur.open, cur.close)),
                    key_level_ohlc={"open": cur.open, "high": cur.high, "low": cur.low, "close": cur.close},
                    timeframe=timeframe,
                )
                out.append(poi)
            if cur.lower_wick > cur.range * 0.3 and cur.close > cur.open:
                poi = POI(
                    tipo="Wick",
                    direzione="Bullish",
                    candela_di_riferimento=cur,
                    prezzo_di_attivazione_top=float(min(cur.open, cur.close)),
                    prezzo_di_attivazione_bottom=float(cur.low),
                    key_level_ohlc={"open": cur.open, "high": cur.high, "low": cur.low, "close": cur.close},
                    timeframe=timeframe,
                )
                out.append(poi)
        print(f"[POI] trovati {len(out)} POI su {timeframe}")
        return out

    def filtra_poi_validi(self, poi_list: List[POI], swing_highs: List[Candle], swing_lows: List[Candle], candles: List[Candle]) -> List[POI]:
        validi: List[POI] = []
        for poi in poi_list:
            # ha preso liquidità?
            ha_preso = False
            if poi.direzione == "Bearish":
                for sh in swing_highs:
                    if poi.candela_di_riferimento.timestamp > sh.timestamp and poi.candela_di_riferimento.high > sh.high:
                        ha_preso = True
                        break
            else:
                for sl in swing_lows:
                    if poi.candela_di_riferimento.timestamp > sl.timestamp and poi.candela_di_riferimento.low < sl.low:
                        ha_preso = True
                        break

            # mitigazione semplice: prezzo è già tornato in zona dopo la candela di rif.
            e_mitigato = False
            for c in candles:
                if c.timestamp <= poi.candela_di_riferimento.timestamp:
                    continue
                # se qualsiasi candela reentra nella zona, è mitigato
                if poi.direzione == "Bearish":
                    if c.high >= poi.prezzo_di_attivazione_bottom:
                        e_mitigato = True
                        break
                else:
                    if c.low <= poi.prezzo_di_attivazione_top:
                        e_mitigato = True
                        break

            if ha_preso and not e_mitigato:
                validi.append(poi)
        print(f"[POI] validi: {len(validi)} / {len(poi_list)}")
        return validi

    # Utility: scelgo POI più logico (più vicino al prezzo corrente, direzione coerente)
    def seleziona_poi_piu_logico(self, poi_validi: List[POI], direzione: str, prezzo_corrente: float) -> Optional[POI]:
        compatibili = [p for p in poi_validi if p.direzione.lower() == direzione.lower()]
        if not compatibili:
            return None
        # distanza media dalla zona
        def dist(p: POI) -> float:
            if prezzo_corrente < p.prezzo_di_attivazione_bottom:
                return p.prezzo_di_attivazione_bottom - prezzo_corrente
            if prezzo_corrente > p.prezzo_di_attivazione_top:
                return prezzo_corrente - p.prezzo_di_attivazione_top
            return 0.0
        sel = min(compatibili, key=dist)
        print(f"[POI] selezionato {sel.tipo} {sel.direzione} @ [{sel.prezzo_di_attivazione_bottom:.5f}, {sel.prezzo_di_attivazione_top:.5f}] dist={dist(sel):.5f}")
        return sel

    # =============================================================================
    # GENERAZIONE SEGNALI (backtest single TF)
    # =============================================================================
    def generate_signals(self) -> pd.DataFrame:
        assert self.data is not None, "Strategy.generate_signals richiede self.data"
        df = self.data.copy()
        n = len(df)
        out = pd.DataFrame(index=df.index, data={
            "signal": np.array(["HOLD"] * n, dtype=object),
            "entry_price": np.full(n, np.nan),
            "stop_loss": np.full(n, np.nan),
            "take_profit": np.full(n, np.nan),
            "confidence": np.zeros(n, dtype=float),
        })

        rr = float(self.params.get("risk_reward_ratio", 2.0))
        spread = float(self.params.get("spread", 0.0))

        candles = self._df_to_candles(df)
        swing_highs, swing_lows = self.identifica_swing_points(candles)
        trend, bos_high, bos_low = self.analizza_struttura_e_bos(swing_highs, swing_lows)
        self.prev_trend = trend
        rng = self.definisci_range_da_quasimodo(candles, timeframe="BACKTEST")

        poi_all = self.identifica_tutti_poi(candles, timeframe="BACKTEST")
        poi_validi = self.filtra_poi_validi(poi_all, swing_highs, swing_lows, candles)

        print(f"[SUMMARY] Trend globale={trend}, BOS_high={getattr(bos_high,'high',None)}, BOS_low={getattr(bos_low,'low',None)}")
        if rng:
            print(
                f"[SUMMARY] Range: StrongLow={rng.strong_low.low:.5f} | StrongHigh={rng.strong_high.high:.5f} | WeakH={len(rng.weak_highs)} | WeakL={len(rng.weak_lows)}"
            )
        
        # loop barre per segnali: entriamo quando il close entra nella zona del POI selezionato
        for i in range(n):
            price = float(df["close"].iat[i])

            # ricalcolo trend vicino (finestra locale sui primi i+1 candele per evitare look-ahead)
            local_candles = candles[: i + 1]
            loc_sh, loc_sl = self.identifica_swing_points(local_candles)
            loc_trend, _, _ = self.analizza_struttura_e_bos(loc_sh, loc_sl)

            poi_dir = "Bullish" if loc_trend == "BULLISH" else ("Bearish" if loc_trend == "BEARISH" else None)
            chosen_poi = None
            if poi_dir is not None:
                chosen_poi = self.seleziona_poi_piu_logico(poi_validi, poi_dir, price)

            print(
                f"[BAR {i}] price={price:.5f} trend={loc_trend} poi_dir={poi_dir} poi_sel={'yes' if chosen_poi else 'no'}"
            )

            if chosen_poi and chosen_poi.contains(price):
                if chosen_poi.direzione == "Bullish":
                    entry = price
                    sl = chosen_poi.prezzo_di_attivazione_bottom - spread
                    risk = entry - sl
                    tp = entry + rr * risk
                    out.at[df.index[i], "signal"] = "BUY"
                    out.at[df.index[i], "entry_price"] = entry
                    out.at[df.index[i], "stop_loss"] = sl
                    out.at[df.index[i], "take_profit"] = tp
                    out.at[df.index[i], "confidence"] = 0.7
                    print(f"  -> BUY @ {entry:.5f} SL={sl:.5f} TP={tp:.5f} (POI {chosen_poi.tipo})")
                elif chosen_poi.direzione == "Bearish":
                    entry = price
                    sl = chosen_poi.prezzo_di_attivazione_top + spread
                    risk = sl - entry
                    tp = entry - rr * risk
                    out.at[df.index[i], "signal"] = "SELL"
                    out.at[df.index[i], "entry_price"] = entry
                    out.at[df.index[i], "stop_loss"] = sl
                    out.at[df.index[i], "take_profit"] = tp
                    out.at[df.index[i], "confidence"] = 0.7
                    print(f"  -> SELL @ {entry:.5f} SL={sl:.5f} TP={tp:.5f} (POI {chosen_poi.tipo})")
            else:
                # HOLD
                pass

        self.signals = out
        return out

    # =============================================================================
    # LIVE: calcolo del prossimo segnale su dati del broker (multitimeframe)
    # =============================================================================
    def next_signal(self, broker: BrokerInterface) -> Optional[TradePlan]:
        # HTF overview
        df_htf = broker.get_candles(self.symbol, self.timeframe_htf, 500)
        candles_htf = self._df_to_candles(df_htf)
        sh, sl = self.identifica_swing_points(candles_htf)
        trend, _, _ = self.analizza_struttura_e_bos(sh, sl)
        rng = self.definisci_range_da_quasimodo(candles_htf, timeframe=self.timeframe_htf)

        poi_all = self.identifica_tutti_poi(candles_htf, timeframe=self.timeframe_htf)
        poi_validi = self.filtra_poi_validi(poi_all, sh, sl, candles_htf)

        price = float(broker.get_price(self.symbol))
        dir_needed = "Bullish" if trend == "BULLISH" else ("Bearish" if trend == "BEARISH" else None)
        if not dir_needed:
            print("[LIVE] trend SIDEWAYS: nessun segnale")
            return None
        poi_sel = self.seleziona_poi_piu_logico(poi_validi, dir_needed, price)
        if not poi_sel:
            print("[LIVE] nessun POI coerente col trend")
            return None
        if not poi_sel.contains(price):
            print(
                f"[LIVE] Prezzo {price:.5f} non nel POI selezionato [{poi_sel.prezzo_di_attivazione_bottom:.5f}, {poi_sel.prezzo_di_attivazione_top:.5f}]"
            )
            return None

        rr = float(self.params.get("risk_reward_ratio", 2.0))
        spread = float(self.params.get("spread", 0.0))
        if dir_needed == "Bullish":
            entry = price
            sl = poi_sel.prezzo_di_attivazione_bottom - spread
            tp = entry + rr * (entry - sl)
            return TradePlan("BUY", entry, sl, tp, 0.7, poi_sel)
        else:
            entry = price
            sl = poi_sel.prezzo_di_attivazione_top + spread
            tp = entry - rr * (sl - entry)
            return TradePlan("SELL", entry, sl, tp, 0.7, poi_sel)


# Convenience export expected by some runners
MyStrategy = Strategy
