from __future__ import annotations
import logging
import os
from datetime import date
import pandas as pd

from .logging_setup import setup_logging
from .config import load_config
from .date_utils import utc_today, parse_ymd, days_ago, ymd
from .data_stooq import StooqClient, clip_date_range
from .data_r2 import from_env as r2_from_env
from .universe import read_tickers_csv, read_sector_etfs_csv, read_ticker_sector_map
from .engine import run_backtest
from .report import save_outputs

log = logging.getLogger(__name__)

def _align_on_dates(dfs: dict[str, pd.DataFrame], dates: list[date]) -> dict[str, pd.DataFrame]:
    out = {}
    idx = pd.Index(dates, name="date")
    for t, df in dfs.items():
        d = df.set_index("date").reindex(idx).reset_index()
        # forward fill close for missing? No — we drop tickers with gaps later.
        out[t] = d
    return out

def _drop_bad(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    good = {}
    for t, df in dfs.items():
        if df[["open","high","low","close","volume"]].isna().any().any():
            continue
        good[t] = df
    return good

def main() -> None:
    setup_logging()
    cfg = load_config()
    r2 = r2_from_env()

    end = parse_ymd(cfg.bt_end_date) if cfg.bt_end_date else utc_today()
    start = days_ago(end, cfg.bt_days)

    run_id = f"run_{ymd(end)}_{cfg.bt_days}d"
    out_dir = os.path.join("out", run_id)

    tickers = read_tickers_csv(cfg.tickers_file)
    ticker_to_sector = read_ticker_sector_map(cfg.ticker_sector_file)
    sector_etfs = read_sector_etfs_csv(cfg.sector_etfs_file)

    if "SPY" not in sector_etfs:
        raise ValueError("sector_etfs.csv must include SPY")

    # Download data
    stooq = StooqClient.create()

    # ETF data (SPY + sectors)
    etf_dfs: dict[str, pd.DataFrame] = {}
    for t in sector_etfs:
        try:
            df = stooq.fetch_daily(t)
            df = clip_date_range(df, start, end)
            etf_dfs[t] = df
            log.info("Loaded ETF %s rows=%d", t, len(df))
        except Exception as e:
            log.warning("ETF %s failed: %s", t, e)

    if "SPY" not in etf_dfs:
        raise RuntimeError("Failed to load SPY from Stooq. Cannot run backtest.")

    # Stock data
    stock_dfs: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            df = stooq.fetch_daily(t)
            df = clip_date_range(df, start, end)
            stock_dfs[t] = df
        except Exception as e:
            log.warning("Stock %s failed: %s", t, e)

    # Build master date index from SPY
    spy = etf_dfs["SPY"].copy()
    master_dates = spy["date"].tolist()

    # Align all series
    aligned_etfs = _align_on_dates(etf_dfs, master_dates)
    aligned_stocks = _align_on_dates(stock_dfs, master_dates)

    # Drop any with gaps
    aligned_etfs = _drop_bad(aligned_etfs)
    aligned_stocks = _drop_bad(aligned_stocks)

    # Extract sector dfs excluding SPY
    sector_dfs = {t: df for t, df in aligned_etfs.items() if t != "SPY"}

    # Backtest
    equity, trades = run_backtest(
        start_cash=cfg.start_cash,
        dates=master_dates,
        spy=aligned_etfs["SPY"],
        sector_dfs=sector_dfs,
        stock_dfs=aligned_stocks,
        ticker_to_sector=ticker_to_sector,
        risk_per_trade=cfg.risk_per_trade,
        max_position_pct=cfg.max_position_pct,
        slippage_bps=cfg.slippage_bps,
        commission_per_share=cfg.commission_per_share,
        commission_min=cfg.commission_min,
        min_dollar_volume=cfg.min_dollar_volume,
        price_max=cfg.price_max,
        perf_3m_min=cfg.perf_3m_min,
    )

    paths = save_outputs(out_dir, equity, trades)

    # Upload to R2
    for name, p in paths.items():
        key = f"{run_id}/{os.path.basename(p)}"
        r2.upload_file(p, key)

    log.info("DONE. Run id: %s", run_id)
    log.info("Local outputs in: %s", out_dir)
    if r2.enabled():
        log.info("Uploaded to R2 bucket=%s prefix=%s", r2.bucket, r2.prefix)

if __name__ == "__main__":
    main()
