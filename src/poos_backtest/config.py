from __future__ import annotations
from dataclasses import dataclass
import os

def _get_env(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None or v == "" else v

def _get_float(name: str, default: float) -> float:
    return float(_get_env(name, str(default)))

def _get_int(name: str, default: int) -> int:
    return int(_get_env(name, str(default)))

@dataclass(frozen=True)
class BacktestConfig:
    bt_days: int
    bt_end_date: str  # optional YYYY-MM-DD
    start_cash: float

    risk_per_trade: float
    max_position_pct: float

    slippage_bps: float
    commission_per_share: float
    commission_min: float

    min_dollar_volume: float
    price_max: float
    perf_3m_min: float

    tickers_file: str
    ticker_sector_file: str
    sector_etfs_file: str

    r2_endpoint: str
    r2_access_key_id: str
    r2_secret_access_key: str
    r2_bucket: str
    r2_prefix: str

def load_config() -> BacktestConfig:
    return BacktestConfig(
        bt_days=_get_int("BT_DAYS", 1095),
        bt_end_date=_get_env("BT_END_DATE", ""),
        start_cash=_get_float("BT_START_CASH", 100000),

        risk_per_trade=_get_float("BT_RISK_PER_TRADE", 0.02),
        max_position_pct=_get_float("BT_MAX_POSITION_PCT", 0.10),

        slippage_bps=_get_float("BT_SLIPPAGE_BPS", 2.0),
        commission_per_share=_get_float("BT_COMMISSION_PER_SHARE", 0.005),
        commission_min=_get_float("BT_COMMISSION_MIN", 1.0),

        min_dollar_volume=_get_float("BT_MIN_DOLLAR_VOLUME", 5_000_000),
        price_max=_get_float("BT_PRICE_MAX", 70.0),
        perf_3m_min=_get_float("BT_PERF_3M_MIN", 0.60),

        tickers_file=_get_env("BT_TICKERS_FILE", "data/tickers.csv"),
        ticker_sector_file=_get_env("BT_TICKER_SECTOR_FILE", "data/ticker_sector_etf.csv"),
        sector_etfs_file=_get_env("BT_SECTOR_ETFS_FILE", "data/sector_etfs.csv"),

        r2_endpoint=_get_env("R2_ENDPOINT", ""),
        r2_access_key_id=_get_env("R2_ACCESS_KEY_ID", ""),
        r2_secret_access_key=_get_env("R2_SECRET_ACCESS_KEY", ""),
        r2_bucket=_get_env("R2_BUCKET", ""),
        r2_prefix=_get_env("R2_PREFIX", "poos-v1"),
    )
