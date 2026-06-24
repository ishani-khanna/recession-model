"""
phase4c_compare.py  -  PHASE 4C: the three yield-spread models, side by side as equals.

Plain English:
  Zandi's advice was "try at least three" curve measures and show them as real
  alternatives, not one star and two footnotes. So here we present all three on equal
  footing - each a probit on its own spread at the 12-month horizon - with the honest
  expanding-window (real-time) AUC and the number of recession onsets behind it.

  10y-3m stays the DEFAULT headline (rule 7: pre-committed, no scan-crowning), but the
  table shows all three so the user can see - and in the dashboard, select - each one.
  Note from Phase 3: 10y-fedfunds actually scores highest out-of-sample; that is shown
  honestly here, with the caveat that it leans on the policy rate (Wright 2006).
"""

import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

H = 12
INITIAL_TRAIN_N = 120
df = pd.read_csv("data/features.csv", index_col="date", parse_dates=True)

SPREADS = {"10y-3m": "spread_10y_3m", "10y-fedfunds": "spread_10y_ff", "10y-2y": "spread_10y_2y"}
NOTE = {"10y-3m": "DEFAULT headline (NY Fed standard)",
        "10y-fedfunds": "highest OOS; leans on policy rate (Wright 2006)",
        "10y-2y": "short history (2y from 1976)"}

rec = df["recession"]
ONSETS = df.index[(rec == 1) & (rec.shift(1) == 0)]


def make_xy(col):
    y = df["recession"].shift(-H)
    return pd.DataFrame({"x": df[col], "y": y}).dropna()


def fit(model, Xtr, ytr, Xte):
    Xtr_c, Xte_c = sm.add_constant(Xtr, has_constant="add"), sm.add_constant(Xte, has_constant="add")
    try:
        res = sm.Probit(ytr, Xtr_c).fit(disp=0, maxiter=200) if model == "probit" else sm.OLS(ytr, Xtr_c).fit()
        return np.asarray(res.predict(Xte_c)), res
    except Exception:
        return None, None


def walk(d, model):
    dates = d.index; preds, actual, td = [], [], []
    for i in range(INITIAL_TRAIN_N, len(d)):
        t = dates[i]
        m = dates <= (t - pd.DateOffset(months=H))
        if m.sum() < INITIAL_TRAIN_N or d.loc[m, "y"].nunique() < 2:
            continue
        s, _ = fit(model, d.loc[m, "x"], d.loc[m, "y"], d["x"].iloc[[i]])
        if s is None:
            continue
        preds.append(s[0]); actual.append(d["y"].iloc[i]); td.append(t)
    if len(set(actual)) < 2:
        return np.nan, 0
    auc = roc_auc_score(actual, preds)
    lo, hi = min(td) + pd.DateOffset(months=H), max(td) + pd.DateOffset(months=H)
    return auc, int(((ONSETS >= lo) & (ONSETS <= hi)).sum())


def insample(d, model):
    s, _ = fit(model, d["x"], d["y"], d["x"])
    return roc_auc_score(d["y"], s) if s is not None else np.nan


print("PHASE 4C - three yield-spread models as first-class equals (probit, 12-month horizon)\n")
print(f"{'spread':13s} {'window':17s} {'in-AUC':>7s} {'OOS-AUC':>8s} {'onsets':>7s} {'cur.prob':>9s}  note")
print("-" * 100)
for name, col in SPREADS.items():
    d = make_xy(col)
    ia = insample(d, "probit")
    oa, on = walk(d, "probit")
    # current probability from a full-sample fit of THIS spread
    _, res = fit("probit", d["x"], d["y"], d["x"])
    cur_x = df[col].dropna().iloc[-1]
    cur_p = float(res.predict(sm.add_constant(pd.Series([cur_x], name="x"), has_constant="add"))[0]) * 100
    win = f"{d.index.min():%Y-%m}..{d.index.max():%Y-%m}"
    star = " <==" if name == "10y-3m" else ""
    print(f"{name:13s} {win:17s} {ia:7.3f} {oa:8.3f} {on:7d} {cur_p:8.1f}%  {NOTE[name]}{star}")

print("\nDefault headline stays 10y-3m (rule 7). All three are selectable in the dashboard;")
print("the higher 10y-fedfunds OOS score is shown honestly, not crowned as 'the model'.")
