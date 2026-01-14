from __future__ import annotations
from datetime import date, datetime, timezone, timedelta

def utc_today() -> date:
    return datetime.now(timezone.utc).date()

def parse_ymd(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()

def ymd(d: date) -> str:
    return d.strftime("%Y-%m-%d")

def days_ago(end: date, days: int) -> date:
    return end - timedelta(days=days)
