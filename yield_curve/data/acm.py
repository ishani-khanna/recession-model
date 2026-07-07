"""NY Fed ACM term premium loader (ACMTP10, 10-year).

The Adrian-Crump-Moench term premium is the rule's trust-check gauge. It is NOT on FRED, so we
download the NY Fed data file and cache the parsed 10Y series to data/cache (keyed by date) so
reruns are offline and fast. Kim-Wright (FRED THREEFYTP10) remains available as a cross-check.
"""

from __future__ import annotations

import datetime as _dt
import io
import warnings

import pandas as pd
import requests

from . import config

warnings.filterwarnings("ignore")

ACM_URL = "https://www.newyorkfed.org/medialibrary/media/research/data_indicators/ACMTermPremium.xls"


def _cache(name):
    return config.CACHE_DIR / f"{name}_{_dt.date.today().isoformat()}.parquet"


def load_acm_tp10(refresh: bool = False) -> pd.Series:
    """Daily ACM 10-year term premium (percentage points), indexed by date."""
    cache = _cache("acm_tp10_daily")
    if cache.exists() and not refresh:
        df = pd.read_parquet(cache)
        return pd.Series(df["value"].values, index=pd.DatetimeIndex(df["date"]), name="acm_tp10")
    r = requests.get(ACM_URL, timeout=180)
    r.raise_for_status()
    df = pd.read_excel(io.BytesIO(r.content), sheet_name="ACM Daily", engine="xlrd")
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    s = pd.Series(pd.to_numeric(df["ACMTP10"], errors="coerce").values, index=df["DATE"]).dropna().sort_index()
    s.name = "acm_tp10"
    pd.DataFrame({"date": s.index, "value": s.values}).to_parquet(cache, index=False)
    return s


def acm_reading(refresh: bool = False) -> dict:
    """Current daily ACM 10Y term premium with its percentile vs monthly history since 1961."""
    daily = load_acm_tp10(refresh=refresh)
    monthly = daily.resample("MS").mean()
    cur = float(daily.iloc[-1])
    pct = round(float((monthly < cur).mean()) * 100)
    cls = "compressed" if cur < 0.50 else ("elevated" if pct > 75 else "near normal")
    return {
        "value_pp": round(cur, 3), "value_bp": round(cur * 100),
        "date": daily.index[-1].date().isoformat(), "percentile": pct, "class": cls,
        "compressed_2020_21": round(float(monthly.loc["2020-06":"2021-12"].mean()), 3),
        "low_2022_23": round(float(monthly.loc["2022-01":"2023-12"].min()), 3),
        "low_2022_23_date": monthly.loc["2022-01":"2023-12"].idxmin().strftime("%Y-%m"),
        "monthly": monthly,
    }
