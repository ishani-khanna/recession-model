"""
fit_phase4.py  -  PHASE 4: add fragility / confirming variables, one at a time.

Plain English:
  Phase 3 left us a strong spread-only baseline (10y-3m, 12-month probit). Now we
  ask, for each extra economic variable: "Does adding this to the spread actually
  improve real-time prediction?" We keep a variable ONLY if it earns its place.

  The trap we must avoid (rule 9): the extra variables have SHORTER histories
  (house prices from 1987, bank survey from 1990, etc.). If we judged the bigger
  model on its short window but compared it to the baseline's full-history score,
  the extra variable could look good purely because the recent period is easier.

  So for EACH candidate we do a fair, head-to-head test on the SAME months:
    1. Find the common window where BOTH the spread and the candidate exist.
    2. Run the expanding-window out-of-sample AUC for the spread-ONLY baseline on
       exactly that window.
    3. Run it again for spread + candidate on exactly that window.
    4. Compare. Report the recession-onset count (usually small) so we know how
       much to trust the difference. Keep the candidate only if it BEATS the
       baseline on that same period.

  Vintage honesty (rule 3): house prices (Case-Shiller) publish ~2 months late and
  use the CPI deflator, so we LAG the real-house-price feature by 2 months to mimic
  what was actually knowable in real time. Treasury-based and credit spreads aren't
  revised, so they need no lag. (Debt ratios from the Fed's Z.1 are also revised;
  we note that as a known limitation rather than silently ignore it.)
"""

import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

df = pd.read_csv("data/features.csv", index_col="date", parse_dates=True)

# ---------------------------------------------------------------------------
# Build the candidate variables (with correct units; validated below).
#   FRED units: CMDEBT & BCNSDODNS in MILLIONS; DPI & GDP in BILLIONS.
#   So a debt/income percent = (debt_millions/1000) / income_billions * 100.
# ---------------------------------------------------------------------------
df["hh_debt_income"] = df["hh_debt"] / 1000 / df["dpi"] * 100
df["hh_debt_gdp"]    = df["hh_debt"] / 1000 / df["gdp"] * 100
df["corp_debt_gdp"]  = df["corp_debt"] / 1000 / df["gdp"] * 100

# Real house price YoY already exists (real_hpi_yoy). Apply 2-month publication lag.
df["real_hpi_yoy_lag2"] = df["real_hpi_yoy"].shift(2)

PRIMARY_SPREAD = "spread_10y_3m"
H = 12

# Candidate confirming / context variables (short name -> column, role)
CANDIDATES = [
    ("Credit spread (Baa-10yr)",     "credit_spread",      "confirming"),
    ("Real house price YoY (lag2)",  "real_hpi_yoy_lag2",  "confirming"),
    ("Unemployment rate",            "unrate",             "confirming"),
    ("Bank tightening (SLOOS)",      "sloos",              "leading"),
    ("Household debt / income",      "hh_debt_income",     "context"),
    ("Household debt / GDP",         "hh_debt_gdp",        "context"),
    ("Corporate debt / GDP",         "corp_debt_gdp",      "context"),
    ("Fed funds rate",               "fedfunds",           "context"),
]

INITIAL_TRAIN_N = 120
rec = df["recession"]
ONSETS = df.index[(rec == 1) & (rec.shift(1) == 0)]


def make_xy(cols, h):
    """Aligned (date-indexed) X[cols] and y = recession AT EXACTLY t+h (point-in-time)."""
    y = df["recession"].shift(-h)
    d = pd.concat([df[cols], y.rename("y")], axis=1).dropna()
    return d


def fit_predict(Xtr, ytr, Xte):
    Xtr_c = sm.add_constant(Xtr, has_constant="add")
    Xte_c = sm.add_constant(Xte, has_constant="add")
    try:
        res = sm.Probit(ytr, Xtr_c).fit(disp=0, maxiter=200)
        return np.asarray(res.predict(Xte_c))
    except Exception:
        return None


def walk_forward_auc(d, feat_cols, h):
    """Expanding-window OOS AUC on dataset d using feat_cols. Train origins s<=t-h."""
    dates = d.index
    preds, actual, tdates = [], [], []
    for i in range(INITIAL_TRAIN_N, len(d)):
        t = dates[i]
        train_mask = dates <= (t - pd.DateOffset(months=h))
        if train_mask.sum() < INITIAL_TRAIN_N:
            continue
        ytr = d.loc[train_mask, "y"]
        if ytr.nunique() < 2:
            continue
        Xtr = d.loc[train_mask, feat_cols]
        score = fit_predict(Xtr, ytr, d[feat_cols].iloc[[i]])
        if score is None:
            continue
        preds.append(score[0]); actual.append(d["y"].iloc[i]); tdates.append(t)
    if len(set(actual)) < 2:
        return np.nan, 0, 0, 0
    auc = roc_auc_score(actual, preds)
    lo, hi = min(tdates) + pd.DateOffset(months=h), max(tdates) + pd.DateOffset(months=h)
    n_onsets = int(((ONSETS >= lo) & (ONSETS <= hi)).sum())
    return auc, n_onsets, int(sum(actual)), len(actual)


# ---------------------------------------------------------------------------
# Unit validation against the brief's targets, before we trust the ratios.
# ---------------------------------------------------------------------------
print("UNIT CHECK (household debt-to-income):")
for d_, tgt in [("2007-10-01", "~132%"), ("2023-06-01", "~95%")]:
    print(f"   {d_[:7]}: {df.loc[pd.Timestamp(d_), 'hh_debt_income']:.1f}%   (target {tgt})")

# ---------------------------------------------------------------------------
# Head-to-head, same-sample comparisons.
# ---------------------------------------------------------------------------
print("\n" + "=" * 84)
print(f"PHASE 4: spread + ONE variable vs spread-ONLY baseline, judged on the SAME window")
print(f"Primary spread = 10y-3m, horizon = {H} months, probit, expanding-window OOS\n")
print(f"{'candidate':28s} {'role':10s} | {'window':17s} {'onsets':>6s} | "
      f"{'base':>6s} {'+var':>6s} {'delta':>7s}  verdict")
print("-" * 100)

results = []
for label, col, role in CANDIDATES:
    d = make_xy([PRIMARY_SPREAD, col], H)        # common sample: both present
    if len(d) <= INITIAL_TRAIN_N + 12:
        print(f"{label:28s} {role:10s} | too few obs"); continue
    base_auc, on, pos, n = walk_forward_auc(d, [PRIMARY_SPREAD], H)
    aug_auc,  _,  _,  _   = walk_forward_auc(d, [PRIMARY_SPREAD, col], H)
    delta = aug_auc - base_auc
    win = f"{d.index.min():%Y-%m}..{d.index.max():%Y-%m}"
    verdict = "KEEP" if delta > 0.005 else ("~flat" if abs(delta) <= 0.005 else "drop")
    results.append((label, role, win, on, base_auc, aug_auc, delta, verdict))
    print(f"{label:28s} {role:10s} | {win:17s} {on:>6d} | "
          f"{base_auc:6.3f} {aug_auc:6.3f} {delta:+7.3f}  {verdict}")

print("\nNotes:")
print(" - 'base' = spread-only on the SAME shortened window (rule 9), NOT the full-sample 0.797.")
print(" - 'onsets' is how many recessions the OOS window actually contains; small = fragile.")
print(" - KEEP only means it beat the baseline on THIS common sample; final selection considered jointly.")
