# POOS V1 Backtest (Free Data) — GitHub + R2 + Railway

## What this is
A research-grade V1 backtest for the POOS strategy using free daily OHLCV data from Stooq.
It implements:
- Market filter (SPY EMA5 > EMA10)
- Sector strength filter (SectorETF/SPY)
- Stock selection (price < 70, 3M perf > 60%, liquidity, relative strength to sector)
- POOS entry (limit at EMA20, daily fill proxy, gap filter)
- Portfolio rules (2% risk per trade, BE at +1%, "green garden" gate)
- Commission + slippage
- Equity curve + metrics

## Repo setup
1) Create the files/folders exactly as in this repo.
2) Edit `data/tickers.csv` to include the tickers you want to consider.
3) Fill `data/ticker_sector_etf.csv` to map tickers to a sector ETF.
   Example:
   AAPL,XLK
   XOM,XLE

## Local run
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .

# Optional: export env vars or use a .env loader of your own
python -m poos_backtest.main
