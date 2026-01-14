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

def _get_bool(name: str, default: bool) -> bool:
    v = _get_env(name, "1" if default else "0").strip().lower()
    return v in ("1", "true", "yes", "y", "on")

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

    # POOS filters
    perf_3m_min: float

    # Universe sources
    tickers_file: str
    ticker_sector_file: str
    sector_etfs_file: str

    # Auto-universe + auto-sector
    auto_universe: bool
    universe_size: int
    auto_sector_assign: bool
    sector_assign_lookback: int

    # Small/Mid regime filters (B defaults)
    market_ticker: str  # SPY or IWM
    price_min: float
    price_max: float
    adv20_min: float
    adv20_max: float
    min_atr_pct: float

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

        perf_3m_min=_get_float("BT_PERF_3M_MIN", 0.60),

        tickers_file=_get_env("BT_TICKERS_FILE", "data/tickers.csv"),
        ticker_sector_file=_get_env("BT_TICKER_SECTOR_FILE", "data/ticker_sector_etf.csv"),
        sector_etfs_file=_get_env("BT_SECTOR_ETFS_FILE", "data/sector_etfs.csv"),

        auto_universe=_get_bool("BT_AUTO_UNIVERSE", True),
        universe_size=_get_int("BT_UNIVERSE_SIZE", 2000),
        auto_sector_assign=_get_bool("BT_AUTO_SECTOR_ASSIGN", True),
        sector_assign_lookback=_get_int("BT_SECTOR_ASSIGN_LOOKBACK", 252),

        # Regime B defaults (small+mid)
        market_ticker=_get_env("BT_MARKET_TICKER", "IWM").strip().upper(),
        price_min=_get_float("BT_PRICE_MIN", 2.0),
        price_max=_get_float("BT_PRICE_MAX", 50.0),
        adv20_min=_get_float("BT_ADV20_MIN", 2_000_000.0),
        adv20_max=_get_float("BT_ADV20_MAX", 80_000_000.0),
        min_atr_pct=_get_float("BT_MIN_ATR_PCT", 0.02),

        r2_endpoint=_get_env("R2_ENDPOINT", ""),
        r2_access_key_id=_get_env("R2_ACCESS_KEY_ID", ""),
        r2_secret_access_key=_get_env("R2_SECRET_ACCESS_KEY", ""),
        r2_bucket=_get_env("R2_BUCKET", ""),
        r2_prefix=_get_env("R2_PREFIX", "poos-v1"),
    )
