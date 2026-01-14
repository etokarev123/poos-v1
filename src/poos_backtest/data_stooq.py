from __future__ import annotations
import io
import logging
from dataclasses import dataclass
from datetime import date
import pandas as pd
import requests

log = logging.getLogger(__name__)

@dataclass(frozen=True)
class StooqClient:
    session: requests.Session

    @staticmethod
    def create() -> "StooqClient":
        s = requests.Session()
        s.headers.update({"User-Agent": "poos-backtest/0.1"})
        return StooqClient(session=s)

    def fetch_daily(self, ticker: str) -> pd.DataFrame:
        """
        Stooq tickers:
          - US stocks: {ticker}.us
          - ETFs often also available as {ticker}.us
        Returns columns: date, open, high, low, close, volume
        """
        sym = f"{ticker.lower()}.us"
        url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
        r = self.session.get(url, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"Stooq HTTP {r.status_code} for {ticker}: {url}")
        text = r.text.strip()
        if text.startswith("404") or "No data" in text or len(text) < 50:
            raise RuntimeError(f"No Stooq data for {ticker}")
        df = pd.read_csv(io.StringIO(text))
        df.columns = [c.strip().lower() for c in df.columns]
        df.rename(columns={"data": "date"}, inplace=True)
        if "date" not in df.columns:
            df.rename(columns={"date": "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df.sort_values("date").reset_index(drop=True)
        # normalize names
        keep = ["date", "open", "high", "low", "close", "volume"]
        df = df[keep]
        df = df.dropna()
        return df

def clip_date_range(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    return df[(df["date"] >= start) & (df["date"] <= end)].copy()
