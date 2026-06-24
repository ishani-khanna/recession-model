"""
build_features.py  -  PHASE 2 of the recession model.

Plain English:
  Phase 1 gave us the raw monthly ingredients. Here we cook the first few
  derived measures the model will actually use:

   1. The THREE candidate yield-curve spreads (each is "long rate minus a short
      rate"; a NEGATIVE value means the curve is INVERTED, the classic warning):
        - 10-year minus 3-month   (the NY Fed's favorite)
        - 10-year minus fed funds
        - 10-year minus 2-year
   2. The REAL (inflation-adjusted) house price and its year-over-year change.
      We must use REAL prices because nominal prices almost never fall - in a
      high-inflation year like 2022, prices can rise in dollars yet fall after
      inflation. Real change is what actually signals housing stress.
   3. The corporate CREDIT spread (Baa minus 10-year Treasury) - we build it now
      since it's a simple subtraction; it gets used as a confirming variable later.

  Output: data/features.csv (the monthly table plus these new columns).
"""

import warnings
import pandas as pd

warnings.filterwarnings("ignore")

df = pd.read_csv("data/monthly.csv", index_col="date", parse_dates=True)

# ---------------------------------------------------------------------------
# 1. The three candidate yield-curve spreads (in percentage points).
# ---------------------------------------------------------------------------
df["spread_10y_3m"] = df["y10"] - df["y3m"]
df["spread_10y_ff"] = df["y10"] - df["fedfunds"]
df["spread_10y_2y"] = df["y10"] - df["y2"]

# ---------------------------------------------------------------------------
# 2. Real house price and its year-over-year % change.
#    real price = nominal index / CPI  (the level is arbitrary; only its
#    growth rate matters). YoY change compares each month to 12 months earlier.
# ---------------------------------------------------------------------------
df["real_hpi"] = df["hpi"] / df["cpi"]
df["real_hpi_yoy"] = df["real_hpi"].pct_change(12) * 100   # percent

# (For reference/interpretation we also keep nominal YoY, to show the contrast.)
df["nominal_hpi_yoy"] = df["hpi"].pct_change(12) * 100

# ---------------------------------------------------------------------------
# 3. Corporate credit spread = Baa corporate yield minus 10-year Treasury.
#    Wider = investors demanding more compensation for risk = stress.
# ---------------------------------------------------------------------------
df["credit_spread"] = df["baa"] - df["y10"]

df.to_csv("data/features.csv")
print(f"Saved data/features.csv  ->  {df.shape[0]} months x {df.shape[1]} columns")
