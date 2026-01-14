from __future__ import annotations
import pandas as pd

def read_tickers_csv(path: str) -> list[str]:
    df = pd.read_csv(path)
    if "ticker" not in df.columns:
        raise ValueError(f"{path} must have a 'ticker' column")
    return sorted(set(df["ticker"].astype(str).str.strip().str.upper().tolist()))

def read_sector_etfs_csv(path: str) -> list[str]:
    df = pd.read_csv(path)
    if "ticker" not in df.columns:
        raise ValueError(f"{path} must have a 'ticker' column")
    return sorted(set(df["ticker"].astype(str).str.strip().str.upper().tolist()))

def read_ticker_sector_map(path: str) -> dict[str, str]:
    df = pd.read_csv(path)
    if not {"ticker", "sector_etf"}.issubset(df.columns):
        raise ValueError(f"{path} must have columns: ticker,sector_etf")
    m: dict[str, str] = {}
    for _, r in df.iterrows():
        t = str(r["ticker"]).strip().upper()
        s = str(r["sector_etf"]).strip().upper()
        if t and s:
            m[t] = s
    return m
