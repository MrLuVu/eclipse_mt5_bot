# Eclipse MT5 Bot (ICMarkets)

Progetto Python che integra la tua **Eclipse Trading Strategy** in un bot per **MetaTrader 5** compatibile con **ICMarkets** (conto demo o reale), con `config.json` per configurare tutti i parametri.

## Struttura

```
eclipse_mt5_bot/
├─ config/
│  └─ config.json
├─ src/
│  ├─ strategy.py            # la tua strategia (così come inviata)
│  ├─ bot.py                 # loop live trading MT5
│  ├─ backtest.py            # backtest offline da CSV
│  ├─ utils.py               # helper: load config, logging, sizing
│  └─ broker/
│     └─ mt5_client.py       # connessione e operatività MT5
├─ run_live.py               # esegue trading live
├─ run_backtest.py           # esegue un backtest su CSV
├─ requirements.txt
└─ README.md
```

## Requisiti
- Python 3.10+
- MetaTrader 5 installato e *loggato* sullo stesso PC dove gira Python
- `pip install -r requirements.txt`

## Configurazione
Modifica `config/config.json`:
- **broker**: credenziali ICMarkets (`login`, `password`, `server`) e `mode` = `"demo"` o `"real"`
- **symbol**, **timeframe**
- **risk**: `risk_per_trade_pct`
- **strategy_params**: tutti i parametri della tua strategia

> Non committare credenziali. Usa variabili d'ambiente o un `.env` locale se vuoi.

## Avvio (Live)
```bash
python run_live.py
```

## Backtest da CSV
Metti un CSV OHLCV con colonne: `time,open,high,low,close,volume` e lancia:
```bash
python run_backtest.py --csv path/to/data.csv --symbol EURUSD --timeframe M5
```

**Nota**: il sizing usa info di simbolo MT5 (`tick_value`, `tick_size`) per stimare la size in lotti in funzione dello stop loss corrente e del rischio in percentuale sull'equity.
