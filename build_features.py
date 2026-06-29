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
# NOTE: pandas pct_change defaults to fill_method='pad', which silently FORWARD-FILLS
# missing months before computing - that would fabricate a house-price reading for months
# we don't actually have data for (and quietly break vintage honesty, rule 3). We pass
# fill_method=None everywhere so a missing input stays missing instead of being invented.
df["real_hpi"] = df["hpi"] / df["cpi"]
df["real_hpi_yoy"] = df["real_hpi"].pct_change(12, fill_method=None) * 100   # percent

# (For reference/interpretation we also keep nominal YoY, to show the contrast.)
df["nominal_hpi_yoy"] = df["hpi"].pct_change(12, fill_method=None) * 100

# ---------------------------------------------------------------------------
# 3. Credit spreads.
#    credit_spread = Baa corporate yield minus 10-year Treasury (default-risk premium).
#    baa_aaa       = Baa minus Aaa (quality spread, riskier vs safest corporates).
#    Wider = investors demanding more compensation for risk = stress.
# ---------------------------------------------------------------------------
df["credit_spread"] = df["baa"] - df["y10"]
df["baa_aaa"] = df["baa"] - df["aaa"]

# ---------------------------------------------------------------------------
# 4. Debt-to-X ratios (FRED units: CMDEBT/BCNSDODNS in millions, DPI/GDP in billions,
#    so a percent = millions/1000 / billions * 100). Forward-fill so the live read
#    never breaks when the quarterly source lags a quarter.
# ---------------------------------------------------------------------------
df["hh_debt_income"] = (df["hh_debt"] / 1000 / df["dpi"] * 100).ffill()
df["hh_debt_gdp"]    = (df["hh_debt"] / 1000 / df["gdp"] * 100).ffill()
df["corp_debt_gdp"]  = (df["corp_debt"] / 1000 / df["gdp"] * 100).ffill()

# ---------------------------------------------------------------------------
# 5. Debt GROWTH / credit impulse (Zandi: growth in debt outstanding as a proxy for
#    credit availability). Year-over-year % growth of household and corporate debt.
# ---------------------------------------------------------------------------
df["hh_debt_growth"]   = df["hh_debt"].pct_change(12, fill_method=None) * 100
df["corp_debt_growth"] = df["corp_debt"].pct_change(12, fill_method=None) * 100

# ---------------------------------------------------------------------------
# 6. Labor supply / migration proxy: YoY growth of the foreign-born labor force.
# ---------------------------------------------------------------------------
df["foreign_born_growth"] = df["foreign_born"].pct_change(12, fill_method=None) * 100

df.to_csv("data/features.csv")
print(f"Saved data/features.csv  ->  {df.shape[0]} months x {df.shape[1]} columns")
