"""acm_daily.py — WIRING ONLY (root pipeline; touches no authored engine file).

The authored engine reads the ACM 10Y term premium from the NY Fed "ACM Monthly"
sheet (yield_curve/data/acm.py -> load_acm_tp10), so the Overview's Signal-3
"trust check" lagged the live daily curve by up to a month and moved in monthly
steps (e.g. it sat at the June reading while the premium had already widened).
Everything else on the dashboard — the yield-curve spread — is read off the live
DAILY curve, and the research paper reads the ACM term premium daily as well.

This thin wrapper reads the NY Fed "ACM Daily" sheet for the CURRENT reading only,
so the trust check is on the same live-daily footing as the rest of the tool. The
monthly series (from the authored engine) still drives the history chart and the
2020-21 / 2022-23 context. Same file, same series (ACMTP10) — just the daily sheet.
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

# Same NY Fed file the engine's monthly loader uses; we just read the "ACM Daily" sheet.
_ACM_URL = (
    "https://www.newyorkfed.org/medialibrary/media/research/"
    "data_indicators/ACMTermPremium.xls"
)
_CACHE = Path("data/cache/acm_tp10_daily_current.parquet")


def load_acm_tp10_daily(refresh: bool = False) -> pd.Series:
    """ACM 10Y term premium as a DAILY Series (pp), from the NY Fed 'ACM Daily' sheet.

    Mirrors the engine's caching pattern: on a fresh checkout (the weekly GitHub
    Action) the cache is absent, so the daily sheet is re-downloaded and is always
    current; the daily sheet carries no month-end lag.
    """
    if refresh or not _CACHE.exists():
        import requests

        _CACHE.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(_ACM_URL, timeout=90)
        df = pd.read_excel(io.BytesIO(r.content), sheet_name="ACM Daily")
        df = df[["DATE", "ACMTP10"]].rename(columns={"DATE": "date", "ACMTP10": "tp"})
        df["date"] = pd.to_datetime(df["date"])
        df.dropna().to_parquet(_CACHE)
    df = pd.read_parquet(_CACHE)
    return pd.Series(
        df["tp"].values, index=pd.to_datetime(df["date"]), name="acm_tp10_daily"
    ).dropna()


def current_daily(refresh: bool = False):
    """(value_pp: float, date: pandas.Timestamp) — latest daily ACM term-premium reading."""
    s = load_acm_tp10_daily(refresh=refresh)
    return float(s.iloc[-1]), s.index[-1]
