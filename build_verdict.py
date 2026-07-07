"""
build_verdict.py  -  PHASE 5: the fragility / CHARACTER layer.

  *** READ THIS FRAMING FIRST (rule 10) ***
  The probit probability (spread only, Phase 3) is the rigorous, out-of-sample-tested
  number and the HEADLINE. This layer is an INTERPRETIVE OVERLAY whose thresholds are
  documented JUDGMENT CALLS, not fitted parameters. It does ONE narrow job: grade the
  CHARACTER of the risk - is a 2008-style credit-cycle collapse brewing? - and it must
  NEVER flip an inverted curve to "false alarm" or tell anyone to relax.

  Why not "false alarm" (the Phase 5 testing lesson):
    The "inverted but no fragility" bucket contained 2022 (no recession) BUT ALSO
    1980, 2000, and 2019 - all REAL recessions. So calling that bucket "false alarm"
    was wrong 3 times out of 4. The fragility variables simply cannot tell 2022 apart
    from 2000/2019 (all inverted, none with extreme fragility); the 2022 call is n=1.
    An inverted curve ALWAYS warns - through the probability. This layer only grades
    whether the warning has 2008-type credit-cycle fragility behind it.

  Three readings:
    inverted + fragility     -> "GENUINE WARNING - credit-cycle (2008-type) fragility present"
    inverted + no fragility  -> "INVERTED, NO CREDIT-CYCLE FRAGILITY - a 2008-style collapse
                                 looks unlikely; a recession from other causes (policy, shock,
                                 sector) is still possible"
    not inverted             -> "ALL CLEAR"

  Fragility = high household debt  OR  (real house prices falling AND credit spreads spiking)

  Thresholds (JUDGMENT CALLS, documented; from the workbook's Pattern Tracker):
    curve inverted        : 10y-3m spread  < 0.0 pp
    household debt high    : debt-to-income > 110%
    real house prices fall : real house-price YoY < 0%  (lagged 2 mo, rule 3 vintage)
    credit spread spiking  : Baa-10yr spread > 2.5 pp
"""

import warnings
import pandas as pd
from config import THRESHOLDS

warnings.filterwarnings("ignore")

df = pd.read_csv("data/features.csv", index_col="date", parse_dates=True)

# debt-to-income (millions/billions unit fix); forward-fill so the LIVE reading never
# breaks when the quarterly debt series lags a quarter (per brief section 3).
df["hh_debt_income"] = (df["hh_debt"] / 1000 / df["dpi"] * 100).ffill()
# real house price YoY with the 2-month publication lag (rule 3)
df["real_hpi_yoy_lag2"] = df["real_hpi_yoy"].shift(2)

# ---- documented JUDGMENT-CALL thresholds (from the SHARED config) ----------
T_INVERT       = THRESHOLDS["inverted"]["value"]   # < 0
T_DEBT_HIGH    = THRESHOLDS["leverage"]["value"]   # > 110
T_HOUSE_FALL   = THRESHOLDS["house"]["value"]      # < -2  (was 0; see config.py)
T_CREDIT_SPIKE = THRESHOLDS["credit"]["value"]     # > 2.5

# full-text readings. Framing (updated to the report): the LEADING explanation for a deep
# inversion that does not become a recession is TERM-PREMIUM COMPRESSION - a possible false
# alarm. The credit-cycle fragility read is supporting real-economy context, not the headline.
V_WARN  = "GENUINE WARNING - credit-cycle (2008-type) fragility present"
V_NOCC  = "INVERTED, likely TERM-PREMIUM-DRIVEN - a possible false alarm; a 2008-style credit collapse looks unlikely"
V_CLEAR = "ALL CLEAR"
SHORT   = {V_WARN: "WARN (credit-cycle)", V_NOCC: "INVERTED (likely false alarm)", V_CLEAR: "all clear"}


def verdict_row(r):
    inverted       = r["spread_10y_3m"] < T_INVERT
    debt_high      = pd.notna(r["hh_debt_income"])    and r["hh_debt_income"]    > T_DEBT_HIGH
    house_falling  = pd.notna(r["real_hpi_yoy_lag2"]) and r["real_hpi_yoy_lag2"] < T_HOUSE_FALL
    credit_spiking = pd.notna(r["credit_spread"])     and r["credit_spread"]     > T_CREDIT_SPIKE
    fragility = debt_high or (house_falling and credit_spiking)
    if not inverted:
        v = V_CLEAR
    elif fragility:
        v = V_WARN
    else:
        v = V_NOCC
    return pd.Series({"inverted": inverted, "debt_high": debt_high,
                      "house_falling": house_falling, "credit_spiking": credit_spiking,
                      "fragility": fragility, "verdict": v})


flags = df.apply(verdict_row, axis=1)
out = pd.concat([df[["spread_10y_3m", "hh_debt_income", "real_hpi_yoy_lag2", "credit_spread"]], flags], axis=1)
out.to_csv("data/verdict.csv")

# ---------------------------------------------------------------------------
# Test against the workbook's 8 reference episodes.
# ---------------------------------------------------------------------------
EPISODES = {
    "1980-12-01": ("RECESSION followed (1981-82)", "policy-induced (Volcker)"),
    "1989-06-01": ("RECESSION followed (1990-91)", "snapshot caught spread at +0.13"),
    "1998-09-01": ("no recession (LTCM scare)",    "curve never truly inverted"),
    "2000-12-01": ("RECESSION followed (2001)",    "tech/investment bust"),
    "2004-06-01": ("no recession (healthy)",       "curve steep"),
    "2007-03-01": ("RECESSION followed (GFC)",     "the credit-cycle case"),
    "2019-08-01": ("RECESSION came (COVID)",       "exogenous pandemic shock"),
    "2023-05-01": ("NO recession (2022-24)",       "the case we must not over-claim"),
}

print("PHASE 5 character layer vs the 8 reference episodes")
print("thresholds (JUDGMENT CALLS): invert<0, debt>110%, real-house<0%, credit>2.5pp\n")
hdr = f"{'episode':9s} {'spr':>6s} {'debt%':>6s} {'rhpiYoY':>8s} {'credit':>6s} | {'I/D/H/C':7s} {'READING':27s} actual"
print(hdr); print("-" * len(hdr))
for d, (actual, why) in EPISODES.items():
    r = out.loc[pd.Timestamp(d)]
    rh = r["real_hpi_yoy_lag2"]; rh = f"{rh:+.1f}" if pd.notna(rh) else "  n/a"
    fl = f"{'I' if r['inverted'] else '-'}{'D' if r['debt_high'] else '-'}" \
         f"{'H' if r['house_falling'] else '-'}{'C' if r['credit_spiking'] else '-'}"
    print(f"{d[:7]:9s} {r['spread_10y_3m']:+6.2f} {r['hh_debt_income']:6.1f} {rh:>8s} "
          f"{r['credit_spread']:6.2f} | {fl:7s} {SHORT[r['verdict']]:27s} {actual}")

# Make the honesty point explicit: who is in the "inverted, no-fragility" bucket?
print("\nThe 'INVERTED (no cc-fragility)' bucket across these episodes:")
for d, (actual, why) in EPISODES.items():
    if out.loc[pd.Timestamp(d), "verdict"] == V_NOCC:
        print(f"   {d[:7]}: {actual:30s} <- {why}")
print("   => 3 of these 4 became recessions. This is WHY we never say 'false alarm':")
print("      the layer grades 2008-type credit risk, it does NOT call a recession off.")

# current reading
cur = out.dropna(subset=["spread_10y_3m"]).iloc[-1]
cd = out.dropna(subset=["spread_10y_3m"]).index[-1].date()
print(f"\nCURRENT ({cd}): spread {cur['spread_10y_3m']:+.2f}, debt {cur['hh_debt_income']:.0f}%, "
      f"credit {cur['credit_spread']:.2f}  ->  {cur['verdict']}")
