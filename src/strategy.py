from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import warnings
warnings.filterwarnings("ignore")


class TradingStrategy:
    """
    Eclipse Trading Strategy - versione più robusta

    Nota:
    - Input: DataFrame OHLCV con colonne [time, open, high, low, close, volume]
    - Tutte le operazioni interne usano posizioni (iloc) per evitare mismatch tra indici datetime.
    """

    def __init__(self, symbol: str, timeframe: str, price_data: pd.DataFrame, params: Dict[str, Any]):
        self.symbol = symbol
        self.timeframe = timeframe
        self.params = self._validate_params(dict(params or {}))

        # normalize and copy data; ensure columns lowercase
        df = price_data.copy()
        df.columns = [c.lower() for c in df.columns]
        # required cols
        for col in ("open", "high", "low", "close"):
            if col not in df.columns:
                raise ValueError(f"DataFrame deve contenere la colonna '{col}'")
        # ensure time column exists
        if "time" not in df.columns:
            # create a synthetic time index if not present
            df = df.reset_index().rename(columns={"index": "time"})
        df = df.sort_values("time").reset_index(drop=True)
        # coerce numeric
        for c in ("open", "high", "low", "close"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=("open", "high", "low", "close")).reset_index(drop=True)

        self.data = df
        # placeholders
        self.signals: pd.DataFrame | None = None
        self.swing_points: pd.DataFrame | None = None
        self.market_structure: pd.DataFrame | None = None
        self.liquidity_zones: pd.DataFrame | None = None
        self.poi_zones: pd.DataFrame | None = None
        self.ranges: pd.DataFrame | None = None

    def _validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        defaults = {
            "swing_period": 3,
            "structure_lookback": 20,
            "liquidity_threshold": 0.001,
            "poi_validity_bars": 50,
            "range_min_size": 0.005,
            "risk_reward_ratio": 2.0,
            "max_drawdown_pct": 5.0,
            "session_asian_start": 23,
            "session_asian_end": 6,
            "session_london_start": 7,
            "session_london_end": 10,
            "session_ny_start": 14,
            "session_ny_end": 17,
        }
        for k, v in defaults.items():
            params.setdefault(k, v)
        return params

    def _prepare_basic_indicators(self) -> None:
        """Calcola indicatori base (hl2, range, wick, body)"""
        d = self.data
        d["hl2"] = (d["high"] + d["low"]) / 2.0
        d["hlc3"] = (d["high"] + d["low"] + d["close"]) / 3.0
        d["range"] = d["high"] - d["low"]
        d["body"] = (d["close"] - d["open"]).abs()
        d["upper_wick"] = d["high"] - d[["open", "close"]].max(axis=1)
        d["lower_wick"] = d[["open", "close"]].min(axis=1) - d["low"]
        # hour if time available
        if "time" in d.columns:
            try:
                d["hour"] = pd.to_datetime(d["time"]).dt.hour
            except Exception:
                pass

    def identify_swing_points(self) -> pd.DataFrame:
        """Trova swing highs/lows usando la formazione 3 candele (candela centrale più alta/bassa)"""
        self._prepare_basic_indicators()
        n = len(self.data)
        period = int(self.params.get("swing_period", 3))
        swing_highs = np.zeros(n, dtype=bool)
        swing_lows = np.zeros(n, dtype=bool)

        for i in range(1, n - 1):
            # Swing High
            if self.data["high"].iat[i] > self.data["high"].iat[i - 1] and                self.data["high"].iat[i] > self.data["high"].iat[i + 1] and                self.data["high"].iat[i] >= self.data["high"].iat[i]:
                swing_highs[i] = True
            # Swing Low
            if self.data["low"].iat[i] < self.data["low"].iat[i - 1] and                self.data["low"].iat[i] < self.data["low"].iat[i + 1] and                self.data["low"].iat[i] <= self.data["low"].iat[i]:
                swing_lows[i] = True

        df = pd.DataFrame(
            {
                "swing_high": swing_highs,
                "swing_low": swing_lows,
                "swing_high_price": np.where(swing_highs, self.data["high"], np.nan),
                "swing_low_price": np.where(swing_lows, self.data["low"], np.nan),
            }
        )
        self.swing_points = df
        return df

    def analyze_market_structure(self) -> pd.DataFrame:
        if self.swing_points is None:
            self.identify_swing_points()

        n = len(self.data)
        market_structure_df = pd.DataFrame(index=self.data.index)
        market_structure_df["trend"] = "SIDEWAYS"
        market_structure_df["HH"] = np.nan
        market_structure_df["HL"] = np.nan
        market_structure_df["LL"] = np.nan
        market_structure_df["LH"] = np.nan
        market_structure_df["BOS_bullish"] = False
        market_structure_df["BOS_bearish"] = False

        last_hh = last_hl = last_ll = last_lh = None

        for i in range(n):
            high = self.data["high"].iat[i]
            low = self.data["low"].iat[i]
            is_swing_high = self.swing_points["swing_high"].iat[i]
            is_swing_low = self.swing_points["swing_low"].iat[i]

            if is_swing_high:
                if last_hh is None or high > last_hh:
                    last_hh = high
                    market_structure_df.at[i, "HH"] = high
                elif last_lh is None or high < last_lh:
                    last_lh = high
                    market_structure_df.at[i, "LH"] = high

            if is_swing_low:
                if last_ll is None or low < last_ll:
                    last_ll = low
                    market_structure_df.at[i, "LL"] = low
                elif last_hl is None or low > last_hl:
                    last_hl = low
                    market_structure_df.at[i, "HL"] = low

            if last_hh is not None and last_hl is not None and last_hh > last_hl:
                market_structure_df.at[i, "trend"] = "BULLISH"
            elif last_ll is not None and last_lh is not None and last_ll < last_lh:
                market_structure_df.at[i, "trend"] = "BEARISH"
            else:
                market_structure_df.at[i, "trend"] = "SIDEWAYS"

            print(f"[DEBUG] i={i} high={high} low={low} "
                  f"swingH={is_swing_high} swingL={is_swing_low} "
                  f"trend={market_structure_df.at[i,'trend']} "
                  f"HH={last_hh} HL={last_hl} LH={last_lh} LL={last_ll} ")
        self.market_structure = market_structure_df
        return market_structure_df

    def identify_liquidity_zones(self) -> pd.DataFrame:
        if self.swing_points is None:
            self.identify_swing_points()

        n = len(self.data)
        liq = pd.DataFrame(index=self.data.index)
        liq["buy_side_liquidity"] = np.nan
        liq["sell_side_liquidity"] = np.nan
        liq["equal_highs"] = False
        liq["equal_lows"] = False
        liq["trendline_liquidity"] = False
        liq["asian_session_liquidity"] = False

        threshold = float(self.params.get("liquidity_threshold", 0.001))
        highs_idx = np.where(self.swing_points["swing_high"].values)[0]
        lows_idx = np.where(self.swing_points["swing_low"].values)[0]

        for idx in highs_idx:
            price = self.data["high"].iat[idx]
            liq.at[idx, "buy_side_liquidity"] = price

        for idx in lows_idx:
            price = self.data["low"].iat[idx]
            liq.at[idx, "sell_side_liquidity"] = price

        for j in range(1, len(highs_idx)):
            i_cur, i_prev = highs_idx[j], highs_idx[j - 1]
            cur_h = self.data["high"].iat[i_cur]
            prev_h = self.data["high"].iat[i_prev]
            if prev_h != 0 and abs(cur_h - prev_h) / prev_h < threshold:
                liq.at[i_cur, "equal_highs"] = True
                liq.at[i_prev, "equal_highs"] = True

        for j in range(1, len(lows_idx)):
            i_cur, i_prev = lows_idx[j], lows_idx[j - 1]
            cur_l = self.data["low"].iat[i_cur]
            prev_l = self.data["low"].iat[i_prev]
            if prev_l != 0 and abs(cur_l - prev_l) / prev_l < threshold:
                liq.at[i_cur, "equal_lows"] = True
                liq.at[i_prev, "equal_lows"] = True

        for i in range(n - 2):
            if (self.data["low"].iat[i] < self.data["low"].iat[i+1] < self.data["low"].iat[i+2]) and                (self.data["low"].iat[i+1] - self.data["low"].iat[i] < self.data["range"].iat[i] * 0.5) and                (self.data["low"].iat[i+2] - self.data["low"].iat[i+1] < self.data["range"].iat[i] * 0.5):
                liq.at[i+2, "trendline_liquidity"] = True
            if (self.data["high"].iat[i] > self.data["high"].iat[i+1] > self.data["high"].iat[i+2]) and                (self.data["high"].iat[i] - self.data["high"].iat[i+1] < self.data["range"].iat[i] * 0.5) and                (self.data["high"].iat[i+1] - self.data["high"].iat[i+2] < self.data["range"].iat[i] * 0.5):
                liq.at[i+2, "trendline_liquidity"] = True

        if "hour" in self.data.columns:
            asian_start = self.params.get("session_asian_start", 23)
            asian_end = self.params.get("session_asian_end", 6)

            asian_session_high = np.nan
            asian_session_low = np.nan
            asian_session_indices = []

            for i in range(n):
                current_hour = self.data["hour"].iat[i]
                if (current_hour >= asian_start) or (current_hour < asian_end):
                    if np.isnan(asian_session_high) or self.data["high"].iat[i] > asian_session_high:
                        asian_session_high = self.data["high"].iat[i]
                    if np.isnan(asian_session_low) or self.data["low"].iat[i] < asian_session_low:
                        asian_session_low = self.data["low"].iat[i]
                    asian_session_indices.append(i)
                else:
                    if len(asian_session_indices) > 0:
                        for idx in asian_session_indices:
                            liq.at[idx, "asian_session_liquidity"] = True
                        asian_session_high = np.nan
                        asian_session_low = np.nan
                        asian_session_indices = []

        self.liquidity_zones = liq.reset_index(drop=True)
        return self.liquidity_zones

    def identify_poi(self) -> pd.DataFrame:
        n = len(self.data)
        poi = pd.DataFrame(
            {
                "order_block_bull": np.zeros(n, dtype=bool),
                "order_block_bear": np.zeros(n, dtype=bool),
                "breaker_block_bull": np.zeros(n, dtype=bool),
                "breaker_block_bear": np.zeros(n, dtype=bool),
                "inefficiency": np.zeros(n, dtype=bool),
                "unmitigated_wick": np.zeros(n, dtype=bool),
                "poi_price": np.full(n, np.nan),
            }
        )

        for i in range(2, n - 1):
            cur = self.data.iloc[i]
            prev = self.data.iloc[i - 1]
            prev2 = self.data.iloc[i - 2]
            nxt = self.data.iloc[i + 1]

            if prev["close"] < prev["open"] and cur["close"] > cur["open"]:
                poi.at[i, "order_block_bull"] = True
                poi.at[i, "poi_price"] = prev["low"]

            if prev["close"] > prev["open"] and cur["close"] < cur["open"]:
                poi.at[i, "order_block_bear"] = True
                poi.at[i, "poi_price"] = prev["high"]

            if i >= 2:
                if prev2["high"] < cur["low"]:
                    poi.at[i-1, "inefficiency"] = True
                    poi.at[i-1, "poi_price"] = (prev2["high"] + cur["low"]) / 2.0
                elif prev2["low"] > cur["high"]:
                    poi.at[i-1, "inefficiency"] = True
                    poi.at[i-1, "poi_price"] = (prev2["low"] + cur["high"]) / 2.0

            if i > 0 and self.market_structure is not None:
                if self.market_structure["BOS_bullish"].iat[i] and prev.get("order_block_bear", False):
                    poi.at[i, "breaker_block_bull"] = True
                    poi.at[i, "poi_price"] = prev["high"]
                elif self.market_structure["BOS_bearish"].iat[i] and prev.get("order_block_bull", False):
                    poi.at[i, "breaker_block_bear"] = True
                    poi.at[i, "poi_price"] = prev["low"]

            if cur["upper_wick"] > (cur["range"] * 0.3) and cur["close"] < cur["open"]:
                poi.at[i, "unmitigated_wick"] = True
                poi.at[i, "poi_price"] = cur["high"]
            elif cur["lower_wick"] > (cur["range"] * 0.3) and cur["close"] > cur["open"]:
                poi.at[i, "unmitigated_wick"] = True
                poi.at[i, "poi_price"] = cur["low"]

        self.poi_zones = poi
        return poi

    def identify_ranges(self) -> pd.DataFrame:
        if self.swing_points is None:
            self.identify_swing_points()
        if self.market_structure is None:
            self.analyze_market_structure()
        if self.liquidity_zones is None:
            self.identify_liquidity_zones()

        n = len(self.data)
        ranges_df = pd.DataFrame(
            {
                "range_high": np.full(n, np.nan),
                "range_low": np.full(n, np.nan),
                "is_quasimodo_bullish": np.zeros(n, dtype=bool),
                "is_quasimodo_bearish": np.zeros(n, dtype=bool),
                "strong_high": np.zeros(n, dtype=bool),
                "strong_low": np.zeros(n, dtype=bool),
                "weak_high": np.zeros(n, dtype=bool),
                "weak_low": np.zeros(n, dtype=bool),
                "internal_liquidity_zone": np.zeros(n, dtype=bool),
                "external_liquidity_zone": np.zeros(n, dtype=bool),
            }
        )

        for i in range(n):
            if i > 0 and self.liquidity_zones["sell_side_liquidity"].iat[i] and self.data["low"].iat[i] < self.data["low"].iat[i-1]:
                if self.market_structure["BOS_bullish"].iat[i]:
                    ranges_df.at[i, "is_quasimodo_bullish"] = True
            if i > 0 and self.liquidity_zones["buy_side_liquidity"].iat[i] and self.data["high"].iat[i] > self.data["high"].iat[i-1]:
                if self.market_structure["BOS_bearish"].iat[i]:
                    ranges_df.at[i, "is_quasimodo_bearish"] = True

        for i in range(n):
            if ranges_df["is_quasimodo_bullish"].iat[i]:
                ranges_df.at[i, "strong_low"] = True
                if i > 0 and self.market_structure["trend"].iat[i-1] == "BULLISH":
                    ranges_df.at[i-1, "weak_high"] = True

            if ranges_df["is_quasimodo_bearish"].iat[i]:
                ranges_df.at[i, "strong_high"] = True
                if i > 0 and self.market_structure["trend"].iat[i-1] == "BEARISH":
                    ranges_df.at[i-1, "weak_low"] = True

        last_strong_low_idx = ranges_df[ranges_df["strong_low"]].index.max()
        last_strong_high_idx = ranges_df[ranges_df["strong_high"]].index.max()

        if not pd.isna(last_strong_low_idx) and not pd.isna(last_strong_high_idx):
            range_start_idx = min(last_strong_low_idx, last_strong_high_idx)
            range_end_idx = max(last_strong_low_idx, last_strong_high_idx)

            range_high_price = self.data["high"].iloc[range_start_idx:range_end_idx+1].max()
            range_low_price = self.data["low"].iloc[range_start_idx:range_end_idx+1].min()

            for i in range(n):
                current_price = self.data["close"].iat[i]
                if i >= range_start_idx and i <= range_end_idx:
                    if current_price > range_low_price and current_price < range_high_price:
                        ranges_df.at[i, "internal_liquidity_zone"] = True
                    else:
                        ranges_df.at[i, "external_liquidity_zone"] = True
                else:
                    ranges_df.at[i, "external_liquidity_zone"] = True

        self.ranges = ranges_df
        return ranges_df

    def update_candle(self, candle):
        if isinstance(candle, dict):
            candle = pd.DataFrame([candle])
        self.data = pd.concat([self.data, candle], ignore_index=True)
        self.analyze_market_structure()
        self.identify_poi()
        self.identify_liquidity_zones()
        self.generate_signals()

    def generate_signals(self) -> pd.DataFrame:
        self.identify_swing_points()
        self.analyze_market_structure()
        self.identify_liquidity_zones()
        self.identify_poi()
        print("Bullish OB totali:", self.poi_zones["order_block_bull"].sum())
        print("Bearish OB totali:", self.poi_zones["order_block_bear"].sum())
        print("Buy Side Liquidity totali:", (~self.liquidity_zones["buy_side_liquidity"].isna()).sum())
        print("Sell Side Liquidity totali:", (~self.liquidity_zones["sell_side_liquidity"].isna()).sum())

        self.identify_ranges()

        n = len(self.data)
        signals = pd.DataFrame({
            "signal": np.array(["HOLD"] * n, dtype=object),
            "entry_price": np.full(n, np.nan),
            "stop_loss": np.full(n, np.nan),
            "take_profit": np.full(n, np.nan),
            "confidence": np.zeros(n, dtype=float),
        })

        rr = float(self.params.get("risk_reward_ratio", 2.0))
        self.data["range"] = self.data["high"] - self.data["low"]

        for i in range(n):
            current_price = self.data["close"].iat[i]
            current_high = self.data["high"].iat[i]
            current_low = self.data["low"].iat[i]
            current_range = self.data["range"].iat[i]

            current_trend = self.market_structure["trend"].iat[i] if self.market_structure is not None else "SIDEWAYS"
            is_bos_bullish = self.market_structure["BOS_bullish"].iat[i] if self.market_structure is not None else False
            is_bos_bearish = self.market_structure["BOS_bearish"].iat[i] if self.market_structure is not None else False

            poi_bull_ob = self.poi_zones["order_block_bull"].iat[i] if self.poi_zones is not None else False
            poi_bear_ob = self.poi_zones["order_block_bear"].iat[i] if self.poi_zones is not None else False
            poi_inefficiency = self.poi_zones["inefficiency"].iat[i] if self.poi_zones is not None else False
            poi_unmitigated_wick = self.poi_zones["unmitigated_wick"].iat[i] if self.poi_zones is not None else False
            poi_breaker_bull = self.poi_zones["breaker_block_bull"].iat[i] if self.poi_zones is not None else False
            poi_breaker_bear = self.poi_zones["breaker_block_bear"].iat[i] if self.poi_zones is not None else False

            liq_bsl = not pd.isna(self.liquidity_zones["buy_side_liquidity"].iat[i]) if self.liquidity_zones is not None else False
            liq_ssl = not pd.isna(self.liquidity_zones["sell_side_liquidity"].iat[i]) if self.liquidity_zones is not None else False
            liq_equal_highs = self.liquidity_zones["equal_highs"].iat[i] if self.liquidity_zones is not None else False
            liq_equal_lows = self.liquidity_zones["equal_lows"].iat[i] if self.liquidity_zones is not None else False
            liq_trendline = self.liquidity_zones["trendline_liquidity"].iat[i] if self.liquidity_zones is not None else False
            liq_asian_session = self.liquidity_zones["asian_session_liquidity"].iat[i] if self.liquidity_zones is not None else False

            range_internal_liq = self.ranges["internal_liquidity_zone"].iat[i] if self.ranges is not None else False
            range_external_liq = self.ranges["external_liquidity_zone"].iat[i] if self.ranges is not None else False

            print(f"\nIndex {i} | Price: {current_price} | Trend: {current_trend}")
            print(f"Bull BOS: {is_bos_bullish} | Bear BOS: {is_bos_bearish}")
            print(f"POI -> Bull OB: {poi_bull_ob}, Bear OB: {poi_bear_ob}, Ineff: {poi_inefficiency}, Wick: {poi_unmitigated_wick}, Breaker Bull: {poi_breaker_bull}, Breaker Bear: {poi_breaker_bear}")
            print(f"Liquidity -> BSL: {liq_bsl}, SSL: {liq_ssl}, Equal Highs: {liq_equal_highs}, Equal Lows: {liq_equal_lows}, Trendline: {liq_trendline}, Asian: {liq_asian_session}")
            print(f"Range Zones -> Internal: {range_internal_liq}, External: {range_external_liq}")

            triggered_signal = False

            if current_trend == "BULLISH":
                if poi_bull_ob and liq_ssl:
                    signal_type = "BUY"
                    confidence = 0.7
                    sl = current_low - current_range * 0.5
                    tp = current_price + rr * (current_price - sl)
                    triggered_signal = True
                elif poi_inefficiency and liq_ssl:
                    signal_type = "BUY"
                    confidence = 0.65
                    sl = current_low - current_range * 0.5
                    tp = current_price + rr * (current_price - sl)
                    triggered_signal = True
                elif poi_unmitigated_wick and liq_ssl:
                    signal_type = "BUY"
                    confidence = 0.6
                    sl = current_low - current_range * 0.5
                    tp = current_price + rr * (current_price - sl)
                    triggered_signal = True
                elif poi_breaker_bull and liq_ssl:
                    signal_type = "BUY"
                    confidence = 0.75
                    sl = current_low - current_range * 0.5
                    tp = current_price + rr * (current_price - sl)
                    triggered_signal = True

            elif current_trend == "BEARISH":
                if poi_bear_ob and liq_bsl:
                    signal_type = "SELL"
                    confidence = 0.7
                    sl = current_high + current_range * 0.5
                    tp = current_price - rr * (sl - current_price)
                    triggered_signal = True
                elif poi_inefficiency and liq_bsl:
                    signal_type = "SELL"
                    confidence = 0.65
                    sl = current_high + current_range * 0.5
                    tp = current_price - rr * (sl - current_price)
                    triggered_signal = True
                elif poi_unmitigated_wick and liq_bsl:
                    signal_type = "SELL"
                    confidence = 0.6
                    sl = current_high + current_range * 0.5
                    tp = current_price - rr * (sl - current_price)
                    triggered_signal = True
                elif poi_breaker_bear and liq_bsl:
                    signal_type = "SELL"
                    confidence = 0.75
                    sl = current_high + current_range * 0.5
                    tp = current_price - rr * (sl - current_price)
                    triggered_signal = True

            if triggered_signal:
                signals.at[i, "signal"] = signal_type
                signals.at[i, "entry_price"] = current_price
                signals.at[i, "stop_loss"] = sl
                signals.at[i, "take_profit"] = tp
                signals.at[i, "confidence"] = confidence
                print(f"Triggered Signal: {signal_type} | SL: {sl} | TP: {tp} | Confidence: {confidence}")
            else:
                signals.at[i, "signal"] = "HOLD"
                signals.at[i, "confidence"] = 0.0
                print("No signal triggered, HOLD")

        self.signals = signals
        return signals

    def backtest(self, initial_balance: float = 10000.0) -> Dict[str, Any]:
        if self.signals is None:
            self.generate_signals()

        data = self.data.reset_index(drop=True)
        signals = self.signals.reset_index(drop=True)
        balance = float(initial_balance)
        position = None
        trades: List[Dict[str, Any]] = []
        equity_curve = [balance]

        for i in range(len(signals)):
            row = signals.iloc[i]
            price = float(data["close"].iat[i])

            if position is not None:
                if position["type"] == "BUY":
                    if price >= position["take_profit"]:
                        pnl = (position["take_profit"] - position["entry_price"]) * position["size"]
                        balance += pnl
                        trades.append({"type": "BUY", "pnl": pnl, "result": "WIN"})
                        position = None
                    elif price <= position["stop_loss"]:
                        pnl = (position["stop_loss"] - position["entry_price"]) * position["size"]
                        balance += pnl
                        trades.append({"type": "BUY", "pnl": pnl, "result": "LOSS"})
                        position = None
                elif position["type"] == "SELL":
                    if price <= position["take_profit"]:
                        pnl = (position["entry_price"] - position["take_profit"]) * position["size"]
                        balance += pnl
                        trades.append({"type": "SELL", "pnl": pnl, "result": "WIN"})
                        position = None
                    elif price >= position["stop_loss"]:
                        pnl = (position["entry_price"] - position["stop_loss"]) * position["size"]
                        balance += pnl
                        trades.append({"type": "SELL", "pnl": pnl, "result": "LOSS"})
                        position = None

            if position is None and row["signal"] in ("BUY", "SELL"):
                if not pd.isna(row["stop_loss"]) and not pd.isna(row["take_profit"]):
                    risk_per_trade = balance * 0.02
                    if row["signal"] == "BUY":
                        risk_per_unit = float(row["entry_price"]) - float(row["stop_loss"])
                    else:
                        risk_per_unit = float(row["stop_loss"]) - float(row["entry_price"])

                    if risk_per_unit > 0:
                        size = risk_per_trade / risk_per_unit
                        position = {
                            "type": row["signal"],
                            "entry_price": float(row["entry_price"]),
                            "stop_loss": float(row["stop_loss"]),
                            "take_profit": float(row["take_profit"]),
                            "size": size,
                        }

            equity_curve.append(balance)

        if len(trades) == 0:
            return {
                "total_return": 0.0,
                "total_trades": 0,
                "win_rate": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "profit_factor": 0.0,
                "final_balance": round(balance, 2),
                "trades": trades,
                "equity_curve": equity_curve,
            }

        total_return = (balance - initial_balance) / initial_balance * 100.0
        total_trades = len(trades)
        wins = [t for t in trades if t["result"] == "WIN"]
        win_rate = len(wins) / total_trades * 100.0 if total_trades > 0 else 0.0

        peak = initial_balance
        max_dd = 0.0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100.0
            if dd > max_dd:
                max_dd = dd

        gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
        gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        arr = np.array(equity_curve, dtype=float)
        if len(arr) > 1:
            rets = np.diff(arr) / arr[:-1]
            sharpe = float(np.mean(rets) / np.std(rets) * np.sqrt(252)) if np.std(rets) > 0 else 0.0
        else:
            sharpe = 0.0

        return {
            "total_return": round(total_return, 2),
            "total_trades": total_trades,
            "win_rate": round(win_rate, 2),
            "max_drawdown": round(max_dd, 2),
            "sharpe_ratio": round(sharpe, 3),
            "profit_factor": round(profit_factor, 2),
            "final_balance": round(balance, 2),
            "trades": trades,
            "equity_curve": equity_curve,
        }

    def optimize(self) -> Dict[str, Any]:
        return {"status": "not_implemented", "message": "Optimize with Optuna (to implement)"}

    def describe(self) -> str:
        txt = (
            f"ECLIPSE TRADING STRATEGY - {self.symbol} {self.timeframe}\n\n"
            "=== LOGICA PRINCIPALE ===\n"
            "1. FORMAZIONE 3 CANDELE: Identifica swing points per struttura mercato\n"
            "2. MARKET STRUCTURE: Analizza trend (HH/HL vs LH/LL) e break of structure\n"
            "3. LIQUIDITÀ: Identifica zone Buy/Sell side liquidity\n"
            "4. POI: Order blocks, breaker blocks, inefficienze\n"
            "5. RANGE TRADING: Cicli liquidità interna/esterna\n\n"
            "=== GESTIONE RISCHIO ===\n"
            f"- Risk/Reward ratio: {self.params.get('risk_reward_ratio')} :1\n"
            f"- Max Drawdown (target): {self.params.get('max_drawdown_pct')}%\n\n"
            "=== PARAMETRI CORRENTI ===\n"
        )
        for k, v in self.params.items():
            txt += f" - {k}: {v}\n"

        if self.signals is not None:
            total = int((self.signals["signal"] != "HOLD").sum())
            buys = int((self.signals["signal"] == "BUY").sum())
            sells = int((self.signals["signal"] == "SELL").sum())
            txt += "\n=== SEGNALI GENERATI ===\n"
            txt += f" - Totale segnali: {total}\n - Buy: {buys}\n - Sell: {sells}\n"

        return txt
