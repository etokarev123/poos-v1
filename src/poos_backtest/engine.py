from __future__ import annotations
import logging
from dataclasses import dataclass
from datetime import date
import pandas as pd
import numpy as np

from .indicators import ema, atr, percent_change, safe_div

log = logging.getLogger(__name__)

@dataclass
class Position:
    ticker: str
    entry_date: date
    entry_price: float
    shares: int
    stop_price: float
    breakeven_set: bool

@dataclass
class Trade:
    ticker: str
    entry_date: date
    entry_price: float
    exit_date: date
    exit_price: float
    shares: int
    pnl: float
    pnl_pct: float
    reason: str

def _slip(price: float, bps: float, is_buy: bool) -> float:
    # buy worse (higher), sell worse (lower)
    mult = 1.0 + (bps / 10000.0) if is_buy else 1.0 - (bps / 10000.0)
    return price * mult

def _commission(shares: int, per_share: float, min_fee: float) -> float:
    return max(min_fee, shares * per_share)

def run_backtest(
    start_cash: float,
    dates: list[date],
    spy: pd.DataFrame,
    sector_dfs: dict[str, pd.DataFrame],
    stock_dfs: dict[str, pd.DataFrame],
    ticker_to_sector: dict[str, str],
    *,
    risk_per_trade: float,
    max_position_pct: float,
    slippage_bps: float,
    commission_per_share: float,
    commission_min: float,
    min_dollar_volume: float,
    price_max: float,
    perf_3m_min: float,
) -> tuple[pd.DataFrame, list[Trade]]:
    """
    Daily bar approximation.
    All dfs must have: date, open, high, low, close, volume and be aligned by date.
    """
    # Prepare SPY indicators
    spy = spy.copy()
    spy["ema5"] = ema(spy["close"], 5)
    spy["ema10"] = ema(spy["close"], 10)
    spy["risk_on"] = spy["ema5"] > spy["ema10"]

    # Prepare sector strength vs SPY
    for sec, sdf in sector_dfs.items():
        sdf["rs"] = safe_div(sdf["close"], spy["close"])
        sdf["rs_ema20"] = ema(sdf["rs"], 20)
        sdf["sec_strong"] = sdf["rs_ema20"] > sdf["rs_ema20"].shift(1)

    # Prepare stock indicators
    prepared: dict[str, pd.DataFrame] = {}
    for t, df in stock_dfs.items():
        d = df.copy()
        d["ema20"] = ema(d["close"], 20)
        d["ema21"] = ema(d["close"], 21)
        d["atr14"] = atr(d, 14)
        d["perf_3m"] = percent_change(d["close"], 63)
        d["dollar_vol"] = d["close"] * d["volume"]
        prepared[t] = d

    cash = start_cash
    equity = start_cash
    positions: dict[str, Position] = {}
    trades: list[Trade] = []
    last_trade_unlocked = True  # green garden gate

    equity_rows = []

    for i, day in enumerate(dates):
        # mark-to-market
        mkt_value = 0.0
        for pos in positions.values():
            px = prepared[pos.ticker].loc[i, "close"]
            mkt_value += pos.shares * px
        equity = cash + mkt_value

        # market regime
        risk_on = bool(spy.loc[i, "risk_on"])
        # manage open positions first (stops & BE)
        to_close: list[tuple[str, float, str]] = []
        for t, pos in positions.items():
            row = prepared[t].iloc[i]
            o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])

            # BE move at +1%
            if (not pos.breakeven_set) and (h >= pos.entry_price * 1.01):
                pos.stop_price = pos.entry_price
                pos.breakeven_set = True
                last_trade_unlocked = True  # unlock new trades once any open pos hits BE

            # Stop hit?
            if l <= pos.stop_price <= h:
                exit_px = _slip(pos.stop_price, slippage_bps, is_buy=False)
                to_close.append((t, exit_px, "STOP"))

        for t, exit_px, reason in to_close:
            pos = positions.pop(t)
            gross = (exit_px - pos.entry_price) * pos.shares
            fees = _commission(pos.shares, commission_per_share, commission_min)
            cash += pos.shares * exit_px - fees
            pnl = gross - fees
            pnl_pct = (exit_px - pos.entry_price) / pos.entry_price
            trades.append(
                Trade(
                    ticker=t,
                    entry_date=pos.entry_date,
                    entry_price=pos.entry_price,
                    exit_date=day,
                    exit_price=exit_px,
                    shares=pos.shares,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    reason=reason,
                )
            )

        # update equity after closes
        mkt_value = 0.0
        for pos in positions.values():
            px = prepared[pos.ticker].loc[i, "close"]
            mkt_value += pos.shares * px
        equity = cash + mkt_value

        # Entry logic: only if risk_on, and "green garden" allows
        if risk_on and last_trade_unlocked:
            candidates = []
            for t, df in prepared.items():
                if t in positions:
                    continue
                row = df.iloc[i]
                if np.isnan(row["ema20"]) or np.isnan(row["atr14"]) or np.isnan(row["perf_3m"]):
                    continue

                # Stock filters
                if float(row["close"]) > price_max:
                    continue
                if float(row["dollar_vol"]) < min_dollar_volume:
                    continue
                if float(row["perf_3m"]) < perf_3m_min:
                    continue

                # Sector filters
                sec = ticker_to_sector.get(t)
                if not sec or sec not in sector_dfs:
                    continue
                if not bool(sector_dfs[sec].iloc[i]["sec_strong"]):
                    continue

                # Stock relative strength to sector
                sec_close = float(sector_dfs[sec].iloc[i]["close"])
                rs = float(row["close"]) / sec_close if sec_close > 0 else np.nan
                # RS trend: compare to previous day (simple proxy)
                if i == 0:
                    continue
                prev_rs = float(prepared[t].iloc[i-1]["close"]) / float(sector_dfs[sec].iloc[i-1]["close"])
                if not (rs > prev_rs):
                    continue

                # POOS limit entry
                limit_px = float(row["ema20"])
                gap_ok = float(row["open"]) >= float(row["ema21"])
                if not gap_ok:
                    continue
                if not (float(row["low"]) <= limit_px <= float(row["high"])):
                    continue

                candidates.append((t, limit_px))

            # optional: choose best by perf_3m (strongest "monster")
            if candidates:
                candidates.sort(key=lambda x: float(prepared[x[0]].iloc[i]["perf_3m"]), reverse=True)
                t, limit_px = candidates[0]

                # Position sizing
                a = float(prepared[t].iloc[i]["atr14"])
                stop_px = limit_px - a
                if stop_px <= 0:
                    stop_px = limit_px * 0.9
                risk_per_share = max(0.01, limit_px - stop_px)

                max_risk_dollars = equity * risk_per_trade
                shares_by_risk = int(max_risk_dollars // risk_per_share)

                max_pos_dollars = equity * max_position_pct
                shares_by_size = int(max_pos_dollars // limit_px)

                shares = max(0, min(shares_by_risk, shares_by_size))
                if shares > 0:
                    entry_px = _slip(limit_px, slippage_bps, is_buy=True)
                    fees = _commission(shares, commission_per_share, commission_min)
                    cost = shares * entry_px + fees
                    if cost <= cash:
                        cash -= cost
                        positions[t] = Position(
                            ticker=t,
                            entry_date=day,
                            entry_price=entry_px,
                            shares=shares,
                            stop_price=stop_px,
                            breakeven_set=False,
                        )
                        last_trade_unlocked = False  # lock until BE
                        log.info("BUY %s %d @ %.2f (limit %.2f) stop %.2f", t, shares, entry_px, limit_px, stop_px)

        equity_rows.append(
            {
                "date": day,
                "cash": cash,
                "market_value": mkt_value,
                "equity": equity,
                "positions": len(positions),
                "risk_on": risk_on,
            }
        )

    equity_df = pd.DataFrame(equity_rows)
    return equity_df, trades
