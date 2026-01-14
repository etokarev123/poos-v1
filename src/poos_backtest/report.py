from __future__ import annotations
import json
from dataclasses import asdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .engine import Trade

def compute_metrics(equity: pd.DataFrame, trades: list[Trade]) -> dict:
    eq = equity["equity"].astype(float)
    returns = eq.pct_change().dropna()

    cagr = None
    if len(eq) > 2 and eq.iloc[0] > 0:
        years = (equity["date"].iloc[-1] - equity["date"].iloc[0]).days / 365.25
        if years > 0:
            cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1

    peak = eq.cummax()
    dd = (eq / peak) - 1
    max_dd = float(dd.min()) if len(dd) else 0.0

    tr = pd.DataFrame([asdict(t) for t in trades]) if trades else pd.DataFrame()
    winrate = None
    pf = None
    expectancy = None
    if not tr.empty:
        wins = tr[tr["pnl"] > 0]["pnl"].sum()
        losses = tr[tr["pnl"] < 0]["pnl"].sum()
        winrate = float((tr["pnl"] > 0).mean())
        pf = float(wins / abs(losses)) if losses != 0 else None
        expectancy = float(tr["pnl"].mean())

    return {
        "start_equity": float(eq.iloc[0]) if len(eq) else None,
        "end_equity": float(eq.iloc[-1]) if len(eq) else None,
        "cagr": float(cagr) if cagr is not None else None,
        "max_drawdown": max_dd,
        "trade_count": int(len(trades)),
        "winrate": winrate,
        "profit_factor": pf,
        "expectancy_dollars": expectancy,
        "daily_vol": float(returns.std()) if len(returns) else None,
    }

def save_outputs(out_dir: str, equity: pd.DataFrame, trades: list[Trade]) -> dict:
    import os
    os.makedirs(out_dir, exist_ok=True)

    equity_path = os.path.join(out_dir, "equity.csv")
    trades_path = os.path.join(out_dir, "trades.csv")
    metrics_path = os.path.join(out_dir, "metrics.json")
    plot_path = os.path.join(out_dir, "equity_curve.png")

    equity.to_csv(equity_path, index=False)

    tr = pd.DataFrame([t.__dict__ for t in trades]) if trades else pd.DataFrame(
        columns=["ticker","entry_date","entry_price","exit_date","exit_price","shares","pnl","pnl_pct","reason"]
    )
    tr.to_csv(trades_path, index=False)

    metrics = compute_metrics(equity, trades)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # plot
    plt.figure()
    plt.plot(pd.to_datetime(equity["date"]), equity["equity"].astype(float))
    plt.title("Equity Curve (POOS V1)")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    return {
        "equity_csv": equity_path,
        "trades_csv": trades_path,
        "metrics_json": metrics_path,
        "equity_png": plot_path,
    }
