from __future__ import annotations
import logging
import os
import json
from datetime import date
import pandas as pd

from .logging_setup import setup_logging
from .config import load_config
from .date_utils import utc_today, parse_ymd, days_ago, ymd
from .data_stooq import StooqClient, clip_date_range
from .data_r2 import from_env as r2_from_env
from .universe import read_tickers_csv, read_sector_etfs_csv, read_ticker_sector_map
from .engine import run_backtest
from .report import save_outputs, compute_metrics

log = logging.getLogger(__name__)

def _align_on_dates(dfs: dict[str, pd.DataFrame], dates: list[date]) -> dict[str, pd.DataFrame]:
    out = {}
    idx = pd.Index(dates, name="date")
    for t, df in dfs.items():
        d = df.set_index("date").reindex(idx).reset_index()
        out[t] = d
    return out

def _drop_bad(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    good = {}
    for t, df in dfs.items():
        if df[["open", "high", "low", "close", "volume"]].isna().any().any():
            continue
        good[t] = df
    return good

def _write_summary(out_dir: str, run_id: str, cfg_dict: dict, metrics: dict, equity: pd.DataFrame, trades_df: pd.DataFrame) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "summary.txt")

    lines = []
    lines.append(f"RUN: {run_id}")
    lines.append("")
    lines.append("CONFIG:")
    lines.append(json.dumps(cfg_dict, indent=2))
    lines.append("")
    lines.append("METRICS:")
    lines.append(json.dumps(metrics, indent=2))
    lines.append("")
    lines.append(f"EQUITY rows: {len(equity)} | start: {equity['equity'].iloc[0]:.2f} | end: {equity['equity'].iloc[-1]:.2f}")
    lines.append(f"TRADES: {len(trades_df)}")
    lines.append("")

    if len(trades_df) > 0:
        lines.append("LAST 10 TRADES:")
        tail = trades_df.tail(10).copy()
        lines.append(tail.to_string(index=False))
        lines.append("")
        lines.append("TOP 10 PNL TRADES:")
        top = trades_df.sort_values("pnl", ascending=False).head(10).copy()
        lines.append(top.to_string(index=False))
        lines.append("")
        lines.append("BOTTOM 10 PNL TRADES:")
        bot = trades_df.sort_values("pnl", ascending=True).head(10).copy()
        lines.append(bot.to_string(index=False))
        lines.append("")
    else:
        lines.append("No trades were taken. Check universe filters / mappings / thresholds.")
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return path

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

    log.info("Universe tickers: %d", len(tickers))
    log.info("Sector map rows: %d", len(ticker_to_sector))
    log.info("Sector ETFs: %s", ",".join(sector_etfs))

    if "SPY" not in sector_etfs:
        raise ValueError("sector_etfs.csv must include SPY")

    stooq = StooqClient.create()

    # ETF data
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

    # Stocks
    stock_dfs: dict[str, pd.DataFrame] = {}
    ok, fail = 0, 0
    for t in tickers:
        try:
            df = stooq.fetch_daily(t)
            df = clip_date_range(df, start, end)
            stock_dfs[t] = df
            ok += 1
        except Exception as e:
            fail += 1
            log.warning("Stock %s failed: %s", t, e)
    log.info("Stocks loaded ok=%d failed=%d", ok, fail)

    # Master date index from SPY
    spy = etf_dfs["SPY"].copy()
    master_dates = spy["date"].tolist()

    aligned_etfs = _align_on_dates(etf_dfs, master_dates)
    aligned_stocks = _align_on_dates(stock_dfs, master_dates)

    aligned_etfs = _drop_bad(aligned_etfs)
    aligned_stocks = _drop_bad(aligned_stocks)

    log.info("After gap-drop: ETFs=%d Stocks=%d", len(aligned_etfs), len(aligned_stocks))

    sector_dfs = {t: df for t, df in aligned_etfs.items() if t != "SPY"}

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

    trades_df = pd.DataFrame([t.__dict__ for t in trades]) if trades else pd.DataFrame(
        columns=["ticker","entry_date","entry_price","exit_date","exit_price","shares","pnl","pnl_pct","reason"]
    )

    metrics = compute_metrics(equity, trades)

    # Log key results into Railway logs
    log.info("RESULTS: start=%.2f end=%.2f cagr=%s maxDD=%.2f%% trades=%d winrate=%s pf=%s expectancy=%s",
             metrics["start_equity"] or float("nan"),
             metrics["end_equity"] or float("nan"),
             f"{metrics['cagr']:.4f}" if metrics.get("cagr") is not None else "None",
             (metrics["max_drawdown"] or 0.0) * 100.0,
             metrics["trade_count"],
             f"{metrics['winrate']:.3f}" if metrics.get("winrate") is not None else "None",
             f"{metrics['profit_factor']:.3f}" if metrics.get("profit_factor") is not None else "None",
             f"{metrics['expectancy_dollars']:.2f}" if metrics.get("expectancy_dollars") is not None else "None")

    # Save standard outputs
    paths = save_outputs(out_dir, equity, trades)

    # Save summary.txt for quick reading
    cfg_dict = cfg.__dict__
    summary_path = _write_summary(out_dir, run_id, cfg_dict, metrics, equity, trades_df)

    # Upload all outputs to R2
    for name, p in paths.items():
        key = f"{run_id}/{os.path.basename(p)}"
        r2.upload_file(p, key)

    r2.upload_file(summary_path, f"{run_id}/summary.txt")

    log.info("DONE. Run id: %s", run_id)
    log.info("Local outputs in: %s", out_dir)
    if r2.enabled():
        log.info("Uploaded to R2 bucket=%s prefix=%s", r2.bucket, r2.prefix)

if __name__ == "__main__":
    main()
