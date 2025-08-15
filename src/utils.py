import json
from loguru import logger
from dataclasses import dataclass
from typing import Any, Dict
import math

def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        cfg = json.load(f)
    return cfg

def rr_size_from_sl(symbol_info, balance: float, entry: float, stop: float, risk_pct: float) -> float:
    """
    Calcola la size in lotti data la distanza di stop e il rischio %.
    Usa tick_value/tick_size dal simbolo per stimare il valore per punto.
    """
    risk = max(0.0, risk_pct) / 100.0 * balance
    distance = abs(entry - stop)
    if distance <= 0:
        return 0.0
    # valore per punto (grezzo)
    tv = getattr(symbol_info, 'trade_tick_value', None) or getattr(symbol_info, 'trade_tick_value_profit', None) or symbol_info.tick_value
    ts = symbol_info.tick_size if symbol_info.tick_size else 0.0001
    value_per_point_per_lot = tv / (1.0 / ts) if ts > 0 else tv
    if value_per_point_per_lot <= 0:
        return 0.0
    points = distance / ts
    loss_per_lot = points * value_per_point_per_lot
    lots = risk / loss_per_lot if loss_per_lot > 0 else 0.0
    # adatta a min/max/step
    lots = max(symbol_info.volume_min, min(lots, symbol_info.volume_max))
    # arrotonda allo step consentito
    step = symbol_info.volume_step if symbol_info.volume_step else 0.01
    lots = math.floor(lots / step) * step
    return max(lots, 0.0)

@dataclass
class Signal:
    type: str
    entry: float
    sl: float
    tp: float
    confidence: float
