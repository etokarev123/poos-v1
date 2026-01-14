from __future__ import annotations
import numpy as np
import pandas as pd

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()

def percent_change(series: pd.Series, periods: int) -> pd.Series:
    return series.pct_change(periods=periods)

def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    out = a / b
    out = out.replace([np.inf, -np.inf], np.nan)
    return out
