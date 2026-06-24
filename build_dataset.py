"""
build_dataset.py  -  PHASE 1 of the recession model.

Plain English:
  This script reaches out to FRED (the St. Louis Fed's free data service),
  downloads every economic series we need, and lines them all up on the same
  MONTHLY calendar. The few series FRED only reports every 3 months (quarterly)
  are gently "filled in" to monthly by interpolation so everything matches.

  The result is one tidy table - one row per month - saved to data/monthly.csv.
  We do NOT compute any spreads or model anything here; this is purely the
  clean, assembled data so the user can eyeball that the numbers look right.

The FRED API key is read from the .env file and is never written into code.
"""

import os
import warnings
import requests
import pandas as pd

warnings.filterwarnings("ignore")  # hide the harmless LibreSSL/urllib3 notice

# ---------------------------------------------------------------------------
# 1. Read the API key from .env (one line: FRED_API_KEY=xxxx)
# ---------------------------------------------------------------------------
def load_api_key(path=".env"):
    for line in open(path):
        if line.startswith("FRED_API_KEY"):
            return line.strip().split("=", 1)[1]
    raise RuntimeError("FRED_API_KEY not found in .env")

API_KEY = load_api_key()

# ---------------------------------------------------------------------------
# 2. The series we want, grouped by how often FRED reports them.
#    The dictionary value is a short, human-friendly column name.
# ---------------------------------------------------------------------------
MONTHLY_SERIES = {
    "GS10":      "y10",        # 10-year Treasury yield (%)
    "TB3MS":     "y3m",        # 3-month Treasury bill yield (%)
    "GS2":       "y2",         # 2-year Treasury yield (%)
    "FEDFUNDS":  "fedfunds",   # Federal funds rate (%)
    "USREC":     "recession",  # NBER recession indicator (1 = recession month)
    "BAA":       "baa",        # Moody's Baa corporate bond yield (%)
    "AAA":       "aaa",        # Moody's Aaa corporate bond yield (%)
    "CPIAUCSL":  "cpi",        # Consumer Price Index (for deflating house prices)
    "CSUSHPINSA":"hpi",        # Case-Shiller national house price index (nominal)
    "UNRATE":    "unrate",     # Unemployment rate (%)
    "LNU01073395":"foreign_born",  # Foreign-born civilian labor force (thousands, from 2007)
    "JTSJOL":    "job_openings",   # Total nonfarm job openings, JOLTS (thousands, from 2000)
}

# These come out quarterly; we will interpolate them to monthly.
QUARTERLY_SERIES = {
    "CMDEBT":    "hh_debt",    # Household debt level
    "DPI":       "dpi",        # Disposable personal income
    "GDP":       "gdp",        # Gross domestic product
    "BCNSDODNS": "corp_debt",  # Nonfinancial corporate business debt
    "DRTSCILM":  "sloos",      # Sr Loan Officer Survey: net % banks tightening C&I loans
    "TDSP":      "tdsp",       # Household debt-service ratio (% of disposable income, from 2005)
}

# ---------------------------------------------------------------------------
# 3. Download one series from FRED as a date-indexed pandas Series.
# ---------------------------------------------------------------------------
def fetch(series_id):
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": API_KEY,
        "file_type": "json",
        "observation_start": "1950-01-01",
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    obs = r.json()["observations"]
    dates, vals = [], []
    for o in obs:
        v = o["value"]
        if v in (".", ""):          # FRED uses "." for missing
            continue
        dates.append(pd.to_datetime(o["date"]))
        vals.append(float(v))
    s = pd.Series(vals, index=pd.DatetimeIndex(dates))
    # snap every observation to the FIRST of its month so series align cleanly
    s.index = s.index.to_period("M").to_timestamp()
    return s

# ---------------------------------------------------------------------------
# 4. Pull everything and assemble a single monthly table.
# ---------------------------------------------------------------------------
def main():
    cols = {}

    print("Downloading monthly series ...")
    for sid, name in MONTHLY_SERIES.items():
        s = fetch(sid)
        cols[name] = s
        print(f"  {sid:11s} -> {name:10s} {s.index.min().date()} .. {s.index.max().date()}  ({len(s)} obs)")

    print("Downloading quarterly series (will interpolate to monthly) ...")
    for sid, name in QUARTERLY_SERIES.items():
        s = fetch(sid)
        cols[name] = s
        print(f"  {sid:11s} -> {name:10s} {s.index.min().date()} .. {s.index.max().date()}  ({len(s)} obs)")

    # Build a continuous monthly index spanning everything we downloaded.
    start = min(s.index.min() for s in cols.values())
    end   = max(s.index.max() for s in cols.values())
    monthly_index = pd.date_range(start=start, end=end, freq="MS")  # month-start

    df = pd.DataFrame(index=monthly_index)
    df.index.name = "date"

    # Place each series onto the monthly grid.
    for name, s in cols.items():
        df[name] = s.reindex(monthly_index)

    # Interpolate the quarterly series to monthly (straight-line between quarters).
    # We only interpolate WITHIN the range each series actually covers (no guessing
    # before the first or after the last real reading).
    for name in QUARTERLY_SERIES.values():
        df[name] = df[name].interpolate(method="linear", limit_area="inside")

    # The recession flag is a 0/1 state, so carry it forward to fill any gaps.
    df["recession"] = df["recession"].ffill()

    df.to_csv("data/monthly.csv")
    print(f"\nSaved data/monthly.csv  ->  {df.shape[0]} months x {df.shape[1]} columns")
    print(f"Coverage: {df.index.min().date()} to {df.index.max().date()}")
    return df

if __name__ == "__main__":
    main()
