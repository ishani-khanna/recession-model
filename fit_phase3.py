"""
fit_phase3.py  -  PHASE 3: the first real models (spread alone).

Plain English, what this does and the discipline behind it:

  We try to predict whether the U.S. is IN a recession exactly h months from now,
  using only one number today: a yield-curve spread. We fit two kinds of model:
    * PROBIT  - the professional standard; gives a clean 0-100% probability.
    * LINEAR  - a plain straight-line model on the 0/1 outcome; simpler to read
                but can spit out probabilities below 0% or above 100%.

  We test 3 spreads x 3 horizons (6/12/18 mo) x 2 models = 18 specifications.
  BUT (per the standing inference rules) we PRE-COMMIT to one primary model -
  10y-3m spread, 12-month horizon, probit - the theory- and literature-backed
  choice. The other 17 are reported only as ROBUSTNESS, never as "the winner we
  found." Picking the best of 18 against ~11 recessions would overstate skill.

  Honesty rules baked in here:
   - LABEL is POINT-IN-TIME: recession at EXACTLY month t+h (matches the NY Fed),
     not "any month within the next h." Only this lines up with the 10/50/90 check.
   - OUT-OF-SAMPLE is EXPANDING WINDOW (walk-forward), never random splits. To
     predict month t+h we train only on (feature_s -> label_{s+h}) pairs whose
     label was already KNOWN by time t, i.e. s+h <= t. No peeking at the future.
   - We report IN-SAMPLE AUC and OUT-OF-SAMPLE AUC side by side; the gap matters.
   - We report the NUMBER OF RECESSION ONSETS behind each out-of-sample AUC.
   - The PRIMARY probit also gets HAC / Newey-West errors (maxlags = h-1) because
     overlapping forward windows make naive standard errors fiction.
   - Separately, we run the NY Fed PUBLISHED-coefficient check (an implementation
     test of the probit math), kept distinct from our own fitted coefficients.
"""

import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

df = pd.read_csv("data/features.csv", index_col="date", parse_dates=True)

SPREADS  = {"10y-3m": "spread_10y_3m", "10y-ff": "spread_10y_ff", "10y-2y": "spread_10y_2y"}
HORIZONS = [6, 12, 18]
INITIAL_TRAIN_N = 120          # first ~10 years used to seed the walk-forward
PRIMARY = ("10y-3m", 12, "probit")

# NBER recession onset months (0 -> 1 transitions), for counting OOS events.
rec = df["recession"]
ONSETS = df.index[(rec == 1) & (rec.shift(1) == 0)]


def make_xy(col, h):
    """Build aligned (origin-date, x, y) where y = recession AT EXACTLY t+h (point-in-time)."""
    x = df[col]
    y = df["recession"].shift(-h)          # label observed h months ahead
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    return d


def fit_predict(model, Xtr, ytr, Xte):
    """Fit probit or linear (LPM); return predicted scores for Xte. None if it fails."""
    Xtr_c = sm.add_constant(Xtr, has_constant="add")
    Xte_c = sm.add_constant(Xte, has_constant="add")
    try:
        if model == "probit":
            res = sm.Probit(ytr, Xtr_c).fit(disp=0, maxiter=200)
        else:                              # linear probability model via OLS
            res = sm.OLS(ytr, Xtr_c).fit()
        return np.asarray(res.predict(Xte_c))
    except Exception:
        return None


def walk_forward_auc(d, model, h):
    """Expanding-window OOS AUC. To predict origin t we train only on origins s<=t-h
       (so the label y_{s+h} was knowable by time t). Returns (auc, n_onsets, n_pos, n_test)."""
    dates = d.index
    preds, actual, test_dates = [], [], []
    # Walk over test origins, starting after the initial training block.
    for i in range(INITIAL_TRAIN_N, len(d)):
        t = dates[i]
        # training origins whose label was already observed by time t: s + h <= t
        train_mask = dates <= (t - pd.DateOffset(months=h))
        if train_mask.sum() < INITIAL_TRAIN_N:
            continue
        ytr = d.loc[train_mask, "y"]
        if ytr.nunique() < 2:              # need both classes to fit
            continue
        Xtr = d.loc[train_mask, "x"]
        score = fit_predict(model, Xtr, ytr, d["x"].iloc[[i]])
        if score is None:
            continue
        preds.append(score[0]); actual.append(d["y"].iloc[i]); test_dates.append(t)
    if len(set(actual)) < 2:
        return np.nan, 0, 0, len(actual)
    auc = roc_auc_score(actual, preds)
    # how many recession ONSETS fall inside the OOS *target* window (t+h months)
    tgt_lo = (min(test_dates) + pd.DateOffset(months=h))
    tgt_hi = (max(test_dates) + pd.DateOffset(months=h))
    n_onsets = int(((ONSETS >= tgt_lo) & (ONSETS <= tgt_hi)).sum())
    return auc, n_onsets, int(sum(actual)), len(actual)


def in_sample_auc(d, model):
    score = fit_predict(model, d["x"], d["y"], d["x"])
    if score is None or d["y"].nunique() < 2:
        return np.nan
    return roc_auc_score(d["y"], score)


# ---------------------------------------------------------------------------
# A. NY Fed PUBLISHED-coefficient check (implementation test, separate from our fit)
# ---------------------------------------------------------------------------
print("=" * 78)
print("A. NY Fed PUBLISHED-coefficient check  (prob = Phi(-0.5333 - 0.6629 * spread))")
print("   This tests our probit MATH, not our fitted model.")
for s, tgt in [(0.76, "~10%"), (-0.82, "~50%"), (-2.40, "~90%")]:
    p = norm.cdf(-0.5333 - 0.6629 * s)
    print(f"   spread {s:+.2f}  ->  {p*100:5.1f}%   (target {tgt})")

# ---------------------------------------------------------------------------
# B. All 18 specifications: in-sample vs expanding-window OOS AUC (+ onset counts)
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("B. All specifications  (PRIMARY is pre-committed: 10y-3m, 12mo, probit)")
print(f"{'spread':7s} {'h':>3s} {'model':7s} | {'in-AUC':>7s} {'OOS-AUC':>8s} {'onsets':>7s} {'pos/N':>10s}  note")
rows = []
for sname, scol in SPREADS.items():
    for h in HORIZONS:
        for model in ("probit", "linear"):
            d = make_xy(scol, h)
            ia = in_sample_auc(d, model)
            oa, onsets, pos, n = walk_forward_auc(d, model, h)
            is_primary = (sname, h, model) == PRIMARY
            note = "<-- PRIMARY" if is_primary else "robustness"
            rows.append((sname, h, model, ia, oa, onsets, pos, n, is_primary))
            print(f"{sname:7s} {h:>3d} {model:7s} | {ia:7.3f} {oa:8.3f} {onsets:7d} {pos:>4d}/{n:<5d}  {note}")

# ---------------------------------------------------------------------------
# C. The PRIMARY model in detail: HAC errors + current reading
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("C. PRIMARY model detail: 10y-3m spread, 12-month horizon, probit, HAC errors")
d = make_xy("spread_10y_3m", 12)
Xc = sm.add_constant(d["x"])
res = sm.Probit(d["y"], Xc).fit(disp=0)
res_hac = sm.Probit(d["y"], Xc).fit(disp=0, cov_type="HAC", cov_kwds={"maxlags": 11})
print(f"   Fitted on {len(d)} months ({d.index.min().date()} .. {d.index.max().date()})")
print(f"   const  = {res.params['const']:+.4f}   (HAC SE {res_hac.bse['const']:.4f})   [NY Fed published: -0.5333]")
print(f"   spread = {res.params['x']:+.4f}   (HAC SE {res_hac.bse['x']:.4f})   [NY Fed published: -0.6629]")

# current probability from our own fitted primary model
cur_spread = df["spread_10y_3m"].dropna().iloc[-1]
cur_date   = df["spread_10y_3m"].dropna().index[-1].date()
cur_p_ours = float(res.predict(sm.add_constant(pd.Series([cur_spread], name="x"), has_constant="add"))[0])
cur_p_nyf  = float(norm.cdf(-0.5333 - 0.6629 * cur_spread))
print(f"\n   CURRENT ({cur_date}): 10y-3m spread = {cur_spread:+.2f} pp")
print(f"     our primary probit  -> 12-mo recession prob = {cur_p_ours*100:.1f}%")
print(f"     NY Fed published    -> 12-mo recession prob = {cur_p_nyf*100:.1f}%")
