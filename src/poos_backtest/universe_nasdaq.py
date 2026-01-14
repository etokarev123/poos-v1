from __future__ import annotations
import io
import logging
import re
from dataclasses import dataclass
import pandas as pd
import requests

log = logging.getLogger(__name__)

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"

_sym_re = re.compile(r"^[A-Z0-9]{1,5}$")

@dataclass(frozen=True)
class NasdaqSymbolDirectory:
    session: requests.Session

    @staticmethod
    def create() -> "NasdaqSymbolDirectory":
        s = requests.Session()
        s.headers.update({"User-Agent": "poos-backtest/0.1"})
        return NasdaqSymbolDirectory(session=s)

    def _fetch_text(self, url: str) -> str:
        r = self.session.get(url, timeout=60)
        r.raise_for_status()
        return r.text

    def _parse_nasdaq_listed(self) -> pd.DataFrame:
        """
        Format: Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
        Has header + footer lines. See NASDAQ Trader docs.
        """
        text = self._fetch_text(NASDAQ_LISTED_URL)
        lines = [ln for ln in text.splitlines() if ln.strip()]
        # remove footer lines that start with "File Creation Time" etc
        data_lines = []
        for ln in lines:
            if ln.startswith("File Creation Time") or ln.startswith("Symbol|") is False and "|" not in ln:
                continue
            data_lines.append(ln)
        # Use pandas with pipe separator; ignore last footer if present
        df = pd.read_csv(io.StringIO("\n".join(data_lines)), sep="|")
        df.columns = [c.strip() for c in df.columns]
        return df

    def _parse_other_listed(self) -> pd.DataFrame:
        """
        Format: ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol
        """
        text = self._fetch_text(OTHER_LISTED_URL)
        lines = [ln for ln in text.splitlines() if ln.strip()]
        # Remove footer if any
        data_lines = []
        for ln in lines:
            if ln.startswith("File Creation Time"):
                continue
            data_lines.append(ln)
        df = pd.read_csv(io.StringIO("\n".join(data_lines)), sep="|")
        df.columns = [c.strip() for c in df.columns]
        return df

    def get_common_stock_symbols(self, limit: int = 2000) -> list[str]:
        """
        Returns up to `limit` symbols:
        - excludes ETFs (ETF == 'Y')
        - excludes test issues (Test Issue == 'Y')
        - keeps simple 1-5 char alnum tickers (best compatibility with Stooq .us)
        """
        df1 = self._parse_nasdaq_listed()
        # columns: Symbol, Test Issue, ETF
        df1 = df1.rename(columns={"Symbol": "Symbol", "Test Issue": "Test Issue", "ETF": "ETF"})
        df1 = df1[["Symbol", "Test Issue", "ETF"]].copy()

        df2 = self._parse_other_listed()
        # columns include: ACT Symbol, Test Issue, ETF
        # Keep ACT Symbol as main symbol
        df2 = df2.rename(columns={"ACT Symbol": "Symbol"})
        df2 = df2[["Symbol", "Test Issue", "ETF"]].copy()

        df = pd.concat([df1, df2], ignore_index=True)
        df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
        df["ETF"] = df["ETF"].astype(str).str.strip().str.upper()
        df["Test Issue"] = df["Test Issue"].astype(str).str.strip().str.upper()

        # filters
        df = df[df["ETF"] != "Y"]
        df = df[df["Test Issue"] != "Y"]

        # keep simple tickers
        df = df[df["Symbol"].apply(lambda s: bool(_sym_re.match(s)))]

        # unique, sorted
        syms = sorted(df["Symbol"].unique().tolist())

        # take first N (we’ll filter by liquidity/perf later anyway)
        if len(syms) < limit:
            log.warning("Only %d symbols after filtering; requested %d", len(syms), limit)
            return syms
        return syms[:limit]
