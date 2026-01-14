from __future__ import annotations
import logging
import os
import json
from datetime import date
import pandas as pd
import numpy as np

from .logging_setup import setup_logging
from .config import load_config
from .date_utils import utc_today, parse_ymd, days_ago, ymd
from .data_stooq import StooqClient, clip_date_range
from .data_r2 import from_env as r2_from_env
from .universe import read_tickers_csv, read_sector_etfs_csv, read_ticker_sector_map
from .universe_nasdaq import NasdaqSymbolDirectory
from .engine import run_backtest
from .report import save_outputs, compute_metrics
from .indicators import atr

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

def _auto_assign_sector_by_corr(
    stocks: dict[str, pd.DataFrame],
    sectors: dict[str, pd.DataFrame],
    lookback: int,
) -> dict[str, str]:
    sector_list = sorted(sectors.keys())
    sector_rets = {}
    for s in sector_list:
        close = sectors[s]["close"].astype(float)
        sector_rets[s] = close.pct_change().replace([np.inf, -np.inf], np.nan)

    mapping: dict[str, str] = {}
    for t, df in stocks.items():
        r = df["close"].astype(float).pct_change().replace([np.inf, -np.inf], np.nan)
        if r.dropna().shape[0] < max(60, lookback // 2):
            continue
        r_win = r.tail(lookback)

        best_s = None
        best_corr = -2.0
        for s in sector_list:
            sr = sector_rets[s].tail(lookback)
            joined = pd.concat([r_win, sr], axis=1).dropna()
            if len(joined) < 30:
                continue
            c = float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))
            if np.isnan(c):
                continue
            if c > best_corr:
                best_corr = c
                best_s = s
        if best_s is not None:
            mapping[t] = best_s

    return mapping

def _prefilter_small_mid(
    stocks: dict[str, pd.DataFrame],
    *,
    price_min: float,
    price_max: float,
    adv20_min: float,
    adv20_max: float,
    min_atr_pct: float,
) -> dict[str, pd.DataFrame]:
    """
    Operational small/mid caps filter:
    - last close in [price_min, price_max]
    - ADV20 in [adv20_min, adv20_max]
    - ATR14/close >= min_atr_pct (on last bar)
    """
    out = {}
    for t, df in stocks.items():
        d = df.copy()
        d["dollar_vol"] = d["close"].astype(float) * d["volume"].astype(float)
        d["adv20"] = d["dollar_vol"].rolling(20).mean()
        d["atr14"] = atr(d, 14)
        # last row
        last = d.iloc[-1]
        px = float(last["close"])
        adv = float(last["adv20"]) if not np.isnan(last["adv20"]) else np.nan
        a = float(last["atr14"]) if not np.isnan(last["atr14"]) else np.nan
        if np.isnan(adv) or np.isnan(a) or px <= 0:
            continue
        atr_pct = a / px

        if not (price_min <= px <= price_max):
            continue
        if not (adv20_min <= adv <= adv20_max):
            continue
        if atr_pct < min_atr_pct:
            continue

        out[t] = df
    return out

def _write_summary(out_dir: str, run_id: str, cfg_dict: dict, metrics: dict, universe_stats: dict) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "summary.txt")
    payload = {
        "run_id": run_id,
        "config": cfg_dict,
        "universe_stats": universe_stats,
        "metrics": metrics,
    }
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(payload, indent=2))
        f.write("\n")
    return path

def main() -> None:
    setup_logging()
    cfg = load_config()
    r2 = r2_from_env()

    end = parse_ymd(cfg.bt_end_date) if cfg.bt_end_date else utc_today()
    start = days_ago(end, cfg.bt_days)

    run_id = f"run_{ymd(end)}_{cfg.bt_days}d"
    out_dir = os.path.join("out", run_id)

    sector_etfs = read_sector_etfs_csv(cfg.sector_etfs_file)
    if "SPY" not in sector_etfs:
        raise ValueError("sector_etfs.csv must include SPY")
    log.info("Sector ETFs: %s", ",".join(sector_etfs))

    # Universe tickers
    if cfg.auto_universe:
        nd = NasdaqSymbolDirectory.create()
        tickers = nd.get_common_stock_symbols(limit=cfg.universe_size)
        log.info("AUTO universe enabled. Pulled %d tickers (limit=%d) from NASDAQ Trader Symbol Directory.",
                 len(tickers), cfg.universe_size)
    else:
        tickers = read_tickers_csv(cfg.tickers_file)
        log.info("Universe tickers: %d (from %s)", len(tickers), cfg.tickers_file)

    stooq = StooqClient.create()

    # Load sector ETFs
    etf_dfs: dict[str, pd.DataFrame] = {}
    for t in sector_etfs:
        try:
            df = stooq.fetch_daily(t)
            df = clip_date_range(df, start, end)
            etf_dfs[t] = df
            log.info("Loaded ETF %s rows=%d", t, len(df))
        except Exception as e:
            log.warning("ETF %s failed: %s", t, e)

    # Load market regime ticker (IWM default)
    market_ticker = cfg.market_ticker
    if market_ticker not in etf_dfs:
        # try load it even if not listed in sector_etfs.csv
        try:
            mdf = stooq.fetch_daily(market_ticker)
            mdf = clip_date_range(mdf, start, end)
            etf_dfs[market_ticker] = mdf
            log.info("Loaded Market %s rows=%d", market_ticker, len(mdf))
        except Exception as e:
            raise RuntimeError(f"Failed to load market ticker {market_ticker}: {e}")

    # Master dates based on market ticker (IWM)
    market_df = etf_dfs[market_ticker].copy()
    master_dates = market_df["date"].tolist()

    # Load stocks
    stock_dfs: dict[str, pd.DataFrame] = {}
    ok, fail = 0, 0
    for t in tickers:
        try:
            df = stooq.fetch_daily(t)
            df = clip_date_range(df, start, end)
            if len(df) < 120:
                fail += 1
                continue
            stock_dfs[t] = df
            ok += 1
        except Exception:
            fail += 1
    log.info("Stocks loaded ok=%d failed=%d (requested=%d)", ok, fail, len(tickers))

    # Align all series to market dates
    aligned_etfs = _align_on_dates(etf_dfs, master_dates)
    aligned_stocks = _align_on_dates(stock_dfs, master_dates)

    aligned_etfs = _drop_bad(aligned_etfs)
    aligned_stocks = _drop_bad(aligned_stocks)

    # Sector dfs excluding SPY and excluding market ticker if it is IWM
    sector_dfs = {}
    for t, df in aligned_etfs.items():
        if t in ("SPY", market_ticker):
            continue
        sector_dfs[t] = df

    universe_before = len(aligned_stocks)

    # Prefilter operational small+mid caps
    aligned_stocks = _prefilter_small_mid(
        aligned_stocks,
        price_min=cfg.price_min,
        price_max=cfg.price_max,
        adv20_min=cfg.adv20_min,
        adv20_max=cfg.adv20_max,
        min_atr_pct=cfg.min_atr_pct,
    )

    universe_after = len(aligned_stocks)
    log.info("Small+Mid prefilter: before=%d after=%d (price %.2f-%.2f, ADV20 %.0f-%.0f, ATR%%>=%.2f%%)",
             universe_before, universe_after,
             cfg.price_min, cfg.price_max,
             cfg.adv20_min, cfg.adv20_max,
             cfg.min_atr_pct * 100.0)

    log.info("After gap-drop: ETFs=%d Stocks=%d", len(aligned_etfs), len(aligned_stocks))

    # Sector mapping
    ticker_to_sector: dict[str, str] = {}
    if cfg.auto_sector_assign:
        ticker_to_sector = _auto_assign_sector_by_corr(
            stocks=aligned_stocks,
            sectors=sector_dfs,
            lookback=cfg.sector_assign_lookback,
        )
        log.info("AUTO sector mapping built: %d tickers mapped (out of %d loaded stocks)",
                 len(ticker_to_sector), len(aligned_stocks))
    else:
        ticker_to_sector = read_ticker_sector_map(cfg.ticker_sector_file)
        log.info("Sector map rows: %d (from %s)", len(ticker_to_sector), cfg.ticker_sector_file)

    # IMPORTANT: engine expects `spy` df for risk_on calculation.
    # We pass MARKET df in place of "spy" so that risk_on is based on IWM in this regime.
    equity, trades = run_backtest(
        start_cash=cfg.start_cash,
        dates=master_dates,
        spy=aligned_etfs[market_ticker],
        sector_dfs=sector_dfs,
        stock_dfs=aligned_stocks,
        ticker_to_sector=ticker_to_sector,
        risk_per_trade=cfg.risk_per_trade,
        max_position_pct=cfg.max_position_pct,
        slippage_bps=cfg.slippage_bps,
        commission_per_share=cfg.commission_per_share,
        commission_min=cfg.commission_min,
        min_dollar_volume=cfg.adv20_min,   # keep engine’s min filter aligned to ADV20 min baseline
        price_max=cfg.price_max,
        perf_3m_min=cfg.perf_3m_min,
    )

    metrics = compute_metrics(equity, trades)

    log.info(
        "RESULTS: start=%.2f end=%.2f cagr=%s maxDD=%.2f%% trades=%d winrate=%s pf=%s expectancy=%s",
        metrics["start_equity"] or float("nan"),
        metrics["end_equity"] or float("nan"),
        f"{metrics['cagr']:.4f}" if metrics.get("cagr") is not None else "None",
        (metrics["max_drawdown"] or 0.0) * 100.0,
        metrics["trade_count"],
        f"{metrics['winrate']:.3f}" if metrics.get("winrate") is not None else "None",
        f"{metrics['profit_factor']:.3f}" if metrics.get("profit_factor") is not None else "None",
        f"{metrics['expectancy_dollars']:.2f}" if metrics.get("expectancy_dollars") is not None else "None",
    )

    paths = save_outputs(out_dir, equity, trades)

    universe_stats = {
        "auto_universe_requested": cfg.universe_size,
        "loaded_stocks_before_prefilter": universe_before,
        "stocks_after_small_mid_prefilter": universe_after,
        "mapped_to_sector": len(ticker_to_sector),
        "market_ticker_for_regime": market_ticker,
    }
    summary_path = _write_summary(out_dir, run_id, cfg.__dict__, metrics, universe_stats)

    for _, p in paths.items():
        key = f"{run_id}/{os.path.basename(p)}"
        r2.upload_file(p, key)
    r2.upload_file(summary_path, f"{run_id}/summary.txt")

    log.info("DONE. Run id: %s", run_id)
    log.info("Local outputs in: %s", out_dir)
    if r2.enabled():
        log.info("Uploaded to R2 bucket=%s prefix=%s", r2.bucket, r2.prefix)

if __name__ == "__main__":
    main()
