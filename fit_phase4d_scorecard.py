"""
fit_phase4d_scorecard.py  -  PHASE 4D: rank the three spreads on Zandi's Step-1 criteria.

Zandi asked: before ranking the curve measures, agree what "most accurate" MEANS, then
rank-order them. We report THREE criteria per spread (not just AUC):

  1. ACCURACY     - in-sample AUC and expanding-window OOS AUC (with onset count), plus a
                    simple HIT RATE: of the recessions in that spread's data window, how many
                    were preceded by an inversion within the prior 18 months.
  2. FALSE POSITIVES - count of inversion episodes that were NOT followed by a recession
                    within 18 months (2022-24 is the marquee example).
  3. LEAD TIME    - for the inversions that DID lead a recession, the average months from the
                    first inverted month to the recession onset (mean and range).

SIGNAL = the curve inverts (spread < 0). We merge inverted runs separated by <=3 calm
months into one episode (a brief un-inversion mid-warning is the same signal). 10y-3m@12mo
stays the pre-committed default headline regardless of the ranking (rule 7); the ranking is
reported, never used to silently re-crown the headline.
"""

import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

H = 12
INITIAL_TRAIN_N = 120
MERGE_GAP = 3          # merge inversion runs separated by <= this many calm months
LEAD_WINDOW = 18       # a signal "leads" a recession if onset is within this many months

df = pd.read_csv("data/features.csv", index_col="date", parse_dates=True)
SPREADS = {"10y-3m": "spread_10y_3m", "10y-ff": "spread_10y_ff", "10y-2y": "spread_10y_2y"}

rec = df["recession"]
ONSETS = list(df.index[(rec == 1) & (rec.shift(1) == 0)])


def auc_pair(col):
    """in-sample AUC and expanding-window OOS AUC (+ onset count) for the spread's probit."""
    y = df["recession"].shift(-H)
    d = pd.DataFrame({"x": df[col], "y": y}).dropna()
    # in-sample
    r = sm.Probit(d["y"], sm.add_constant(d["x"])).fit(disp=0)
    ia = roc_auc_score(d["y"], r.predict(sm.add_constant(d["x"])))
    # OOS expanding window
    dts = d.index; preds, act, td = [], [], []
    for i in range(INITIAL_TRAIN_N, len(d)):
        t = dts[i]; m = dts <= (t - pd.DateOffset(months=H))
        if m.sum() < INITIAL_TRAIN_N or d.loc[m, "y"].nunique() < 2:
            continue
        try:
            rr = sm.Probit(d.loc[m, "y"], sm.add_constant(d.loc[m, "x"], has_constant="add")).fit(disp=0, maxiter=200)
            preds.append(float(rr.predict(sm.add_constant(pd.Series([d["x"].iloc[i]], name="x"), has_constant="add"))[0]))
            act.append(d["y"].iloc[i]); td.append(t)
        except Exception:
            continue
    oa = roc_auc_score(act, preds)
    lo, hi = min(td) + pd.DateOffset(months=H), max(td) + pd.DateOffset(months=H)
    on = sum(1 for o in ONSETS if lo <= o <= hi)
    return round(ia, 3), round(oa, 3), on


def inversion_episodes(col):
    """Maximal runs of months with spread<0, merging runs separated by <= MERGE_GAP calm months."""
    s = df[col].dropna()
    inv = s < 0
    months = list(s.index)
    eps, start, prev = [], None, None
    gap = 0
    for d_, isinv in inv.items():
        if isinv:
            if start is None:
                start = d_
            prev = d_; gap = 0
        else:
            if start is not None:
                gap += 1
                if gap > MERGE_GAP:
                    eps.append((start, prev)); start = None
    if start is not None:
        eps.append((start, prev))
    return eps


def scorecard(col):
    eps = inversion_episodes(col)
    span_start, span_end = df[col].dropna().index.min(), df[col].dropna().index.max()
    onsets_in = [o for o in ONSETS if span_start <= o <= span_end]
    leads, fp, caught = [], 0, set()
    for (a, b) in eps:
        nxt = [o for o in ONSETS if 0 < (o.year - a.year) * 12 + (o.month - a.month) <= LEAD_WINDOW]
        if nxt:
            o = min(nxt); caught.add(o)
            leads.append((o.year - a.year) * 12 + (o.month - a.month))
        else:
            fp += 1
    hit_rate = len(caught) / len(onsets_in) if onsets_in else float("nan")
    ia, oa, on = auc_pair(col)
    return {
        "in_auc": ia, "oos_auc": oa, "onsets": on,
        "inversions": len(eps), "false_pos": fp,
        "hit_rate": round(hit_rate, 2), "recessions_in_window": len(onsets_in),
        "lead_mean": round(float(np.mean(leads)), 1) if leads else None,
        "lead_min": min(leads) if leads else None, "lead_max": max(leads) if leads else None,
    }


cards = {name: scorecard(col) for name, col in SPREADS.items()}

print("PHASE 4D - Zandi Step-1 scorecard (signal = curve inverts; lead window 18 months)\n")
hdr = (f"{'spread':9s} | {'in-AUC':>6s} {'OOS':>6s} {'onsets':>6s} | {'inversions':>10s} "
       f"{'false+':>6s} {'hit-rate':>8s} | {'lead mean':>9s} {'range':>9s}")
print(hdr); print("-" * len(hdr))
for name, c in cards.items():
    lr = f"{c['lead_min']}-{c['lead_max']}" if c['lead_mean'] is not None else "n/a"
    print(f"{name:9s} | {c['in_auc']:6.3f} {c['oos_auc']:6.3f} {c['onsets']:6d} | {c['inversions']:10d} "
          f"{c['false_pos']:6d} {c['hit_rate']:8.2f} | {str(c['lead_mean']):>9s} {lr:>9s}")

print("\nNote: hit-rate = recessions in window preceded by an inversion within 18 months /")
print("recessions in window. false+ = inversion episodes with no recession within 18 months.")

# ---- explicit rank-ordering on each criterion -----------------------------
def rank(metric, higher_better=True):
    items = [(n, c[metric]) for n, c in cards.items() if c[metric] is not None]
    items.sort(key=lambda kv: kv[1], reverse=higher_better)
    return " > ".join(f"{n}({v})" for n, v in items)

print("\nRANK-ORDERING by criterion:")
print(f"  1. Accuracy (OOS AUC, higher better): {rank('oos_auc', True)}")
print(f"  2. Fewest false positives (lower better): {rank('false_pos', False)}")
print(f"  3. Longest lead time (higher better): {rank('lead_mean', True)}")
