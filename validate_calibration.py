"""
validate_calibration.py  -  PHASE 8: calibration / reliability of the headline gauge.

Plain English:
  The dashboard's big number is a PROBABILITY. Calibration asks the fair question:
  when the model said "~30%", did a recession actually follow about 30% of the time?

  Crucial honesty choice (per the brief): we calibrate on the OUT-OF-SAMPLE
  (expanding-window) predictions - what a user would actually have seen live - NOT
  the in-sample fit, which would flatter the model. We bin the predictions, compare
  the observed recession frequency to the predicted probability in each bin, and
  report a single Brier score. With only ~11 recessions and overlapping 12-month
  labels, the bins are lumpy - we say so rather than draw a falsely clean curve.
"""

import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

warnings.filterwarnings("ignore")

H, INITIAL_TRAIN_N, SPREAD = 12, 120, "spread_10y_3m"
df = pd.read_csv("data/features.csv", index_col="date", parse_dates=True)

y_all = df["recession"].shift(-H)
d = pd.DataFrame({"x": df[SPREAD], "y": y_all}).dropna()

# expanding-window OOS predictions: to predict origin t, train only on origins s<=t-H
dts = d.index
rows = []
for i in range(INITIAL_TRAIN_N, len(d)):
    t = dts[i]; m = dts <= (t - pd.DateOffset(months=H))
    if m.sum() < INITIAL_TRAIN_N or d.loc[m, "y"].nunique() < 2:
        continue
    try:
        r = sm.Probit(d.loc[m, "y"], sm.add_constant(d.loc[m, "x"], has_constant="add")).fit(disp=0, maxiter=200)
        p = float(r.predict(sm.add_constant(pd.Series([d["x"].iloc[i]], name="x"), has_constant="add"))[0])
        rows.append((t, p, d["y"].iloc[i]))
    except Exception:
        continue

oos = pd.DataFrame(rows, columns=["date", "p", "y"]).set_index("date")
print(f"OOS predictions: {len(oos)} months, {oos.index.min().date()}..{oos.index.max().date()}")

# Brier score (lower = better; 0.0 perfect, 0.25 = always guessing 50%)
brier = float(((oos["p"] - oos["y"]) ** 2).mean())
base_rate = float(oos["y"].mean())
brier_base = float(((base_rate - oos["y"]) ** 2).mean())   # always-predict-base-rate baseline
print(f"Brier score: {brier:.3f}   (always-predict-base-rate {base_rate:.2f} -> {brier_base:.3f})")

# Reliability table (coarse bins)
bins = [0, 0.10, 0.25, 0.50, 1.01]
labels = ["0-10%", "10-25%", "25-50%", "50-100%"]
oos["bin"] = pd.cut(oos["p"], bins=bins, labels=labels, right=False)
print("\nReliability table (out-of-sample):")
print(f"{'bin':8s} {'n months':>9s} {'mean pred':>10s} {'observed':>9s}")
tbl = []
for lab in labels:
    g = oos[oos["bin"] == lab]
    if len(g):
        mp, ob = g["p"].mean() * 100, g["y"].mean() * 100
        tbl.append((lab, len(g), mp, ob))
        print(f"{lab:8s} {len(g):>9d} {mp:>9.1f}% {ob:>8.1f}%")
    else:
        print(f"{lab:8s} {0:>9d} {'-':>10s} {'-':>9s}")

# The 2022-24 episode: what the model said, OOS, at the deepest inversion
ep = oos.loc["2022-06":"2023-12"] if len(oos.loc["2022-06":"2023-12"]) else oos.tail(0)
if len(ep):
    peak = ep["p"].idxmax()
    print(f"\n2022-24 episode OOS peak reading: {ep['p'].max()*100:.0f}% at {peak.date()} "
          f"(outcome at +12mo: {'recession' if ep.loc[peak,'y']==1 else 'NO recession'})")
# NY Fed published-coefficient reading at the May-2023 deepest inversion, for reference
sp_2023 = df.loc[pd.Timestamp('2023-05-01'), SPREAD]
print(f"NY Fed published formula at May-2023 spread {sp_2023:+.2f}: {norm.cdf(-0.5333-0.6629*sp_2023)*100:.0f}%")
