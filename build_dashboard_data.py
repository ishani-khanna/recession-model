"""
build_dashboard_data.py  -  PHASE 6 (part 1 of 2): compute everything the dashboard shows.

Plain English:
  This produces the numbers the dashboard draws, and writes them to a JSON file.
  It then injects that JSON into the HTML template to produce a single, fully
  self-contained output/dashboard.html (no internet needed to open it).

  The two probability lines (the heart of the honesty):
    * FITTED   - the probit fit on ALL data, then asked to "predict" every month.
                 Smooth and flattering: it has seen the whole history. This is the
                 NY-Fed-style line. We show it only as a light DASHED reference.
    * REAL-TIME- the expanding-window walk-forward. To get the value for month t we
                 train ONLY on data knowable by t (pairs whose 12-month outcome had
                 already happened), then predict month t. This is the model's HONEST
                 track record - bumpier, sometimes late. We show it as the PROMINENT
                 solid line, and its AUC (~0.80, 8 onsets) is the headline number.

  Both lines are dated at the FORECAST month t (probability of recession 12 months
  later), so in a good model the line rises ~12 months BEFORE a shaded recession.
"""

import json
import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from config import THRESHOLDS, is_danger

warnings.filterwarnings("ignore")

H = 12
INITIAL_TRAIN_N = 120
SPREAD = "spread_10y_3m"

df = pd.read_csv("data/features.csv", index_col="date", parse_dates=True)
verdict = pd.read_csv("data/verdict.csv", index_col="date", parse_dates=True)

# Origins we can plot: every month with a spread reading.
s_all = df[SPREAD].dropna()
y_all = df["recession"].shift(-H)             # point-in-time label (recession at t+12)


def probit_fit(x, y):
    return sm.Probit(y, sm.add_constant(x, has_constant="add")).fit(disp=0, maxiter=200)


def probit_predict(res, x):
    return float(res.predict(sm.add_constant(pd.Series([x], name=SPREAD), has_constant="add"))[0])


# ---- FITTED line: one fit on the full sample, predicted everywhere ----------
train_full = pd.DataFrame({"x": s_all, "y": y_all}).dropna()
res_full = probit_fit(train_full["x"], train_full["y"])
fitted = {d: probit_predict(res_full, x) for d, x in s_all.items()}

# LIVE daily curve from the Overview generator (build_two_clock_data writes it first in the
# pipeline) so this Explore page shows the SAME current reading as the Overview. Falls back to
# the latest monthly spread if the live file isn't present.
latest_date = s_all.index.max()
LIVE = None
try:
    with open("output/two_clock_data.json") as _f:
        LIVE = json.load(_f).get("live_curve")
except Exception:
    LIVE = None

if LIVE and LIVE.get("flagship_prob") is not None:
    gauge_prob = float(LIVE["flagship_prob"])          # daily bond-equivalent reading (matches Overview)
    gauge_asof = LIVE.get("as_of") or latest_date.strftime("%Y-%m-%d")
    gauge_label = f"current 12-month probability — live daily curve, as of {gauge_asof}"
else:
    gauge_prob = probit_predict(res_full, s_all.loc[latest_date]) * 100
    gauge_label = f"current model probability (all data through {latest_date.strftime('%b %Y')})"

# ---- REAL-TIME line: expanding-window walk-forward -------------------------
realtime = {}
dates = s_all.index
pairs = pd.DataFrame({"x": s_all, "y": y_all})          # y NaN for last H months
for t in dates:
    # training pairs whose 12-month outcome was already known by month t: s <= t-H
    tr = pairs.loc[(pairs.index <= (t - pd.DateOffset(months=H)))].dropna()
    if len(tr) < INITIAL_TRAIN_N or tr["y"].nunique() < 2:
        continue
    try:
        res = probit_fit(tr["x"], tr["y"])
        realtime[t] = probit_predict(res, s_all.loc[t])
    except Exception:
        continue

# ---- Honest performance numbers -------------------------------------------
# real-time AUC over months where the actual 12-month outcome is known
rt = pd.Series(realtime)
ev = pd.DataFrame({"p": rt, "y": y_all}).dropna()
rt_auc = roc_auc_score(ev["y"], ev["p"])
# onsets inside the real-time evaluation window
rec = df["recession"]
onsets = df.index[(rec == 1) & (rec.shift(1) == 0)]
lo = ev.index.min() + pd.DateOffset(months=H)
hi = ev.index.max() + pd.DateOffset(months=H)
rt_onsets = int(((onsets >= lo) & (onsets <= hi)).sum())
# in-sample AUC (fitted line) for the contrast
fit_eval = pd.DataFrame({"p": pd.Series(fitted), "y": y_all}).dropna()
is_auc = roc_auc_score(fit_eval["y"], fit_eval["p"])

# ---- Recession spans (for shading) ----------------------------------------
spans = []
in_rec = False
for d, v in rec.items():
    if v == 1 and not in_rec:
        start = d; in_rec = True
    elif v == 0 and in_rec:
        spans.append((start, d)); in_rec = False
if in_rec:
    spans.append((start, rec.index.max()))
recessions = [{"start": s.strftime("%Y-%m"), "end": e.strftime("%Y-%m")} for s, e in spans]

# ---- Current fragility lights (values + thresholds, judgment calls visible) -
cv = verdict.loc[latest_date]
rh = cv["real_hpi_yoy_lag2"]
# thresholds pulled from the SHARED config so the lights can never drift from the matrix/verdict
T = THRESHOLDS
# current 10y-3m spread: the live daily (bond-equivalent) value when available, matching the Overview
_spread_now = float(LIVE["spread_10y3m"]) if LIVE else float(cv["spread_10y_3m"])
lights = [
    {"name": "Curve inverted",            "value": f"{_spread_now:+.2f} pp",
     "threshold": f"warns if < {T['inverted']['value']:.2f} pp", "on": _spread_now < T["inverted"]["value"]},
    {"name": "Household debt high",        "value": f"{cv['hh_debt_income']:.0f}% of income",
     "threshold": f"warns if > {T['leverage']['value']:.0f}%",   "on": bool(cv["debt_high"])},
    {"name": "Real house prices falling",  "value": ("n/a" if pd.isna(rh) else f"{rh:+.1f}% YoY"),
     "threshold": f"warns if < {T['house']['value']:.0f}%",      "on": bool(cv["house_falling"])},
    {"name": "Credit spread spiking",      "value": f"{cv['credit_spread']:.2f} pp",
     "threshold": f"warns if > {T['credit']['value']:.2f} pp",   "on": bool(cv["credit_spiking"])},
]

# character banner
full = str(cv["verdict"])
if "GENUINE WARNING" in full:
    cls, short = "warn", "GENUINE WARNING"
elif "INVERTED" in full:
    cls, short = "nocc", "INVERTED — likely a false alarm"
else:
    cls, short = "clear", "ALL CLEAR"
    full = "The yield curve is not inverted, so the spread is not signalling a recession warning."

# ===========================================================================
# PHASE 6B: scenario-lab data (three coefficient pairs, defaults, thresholds, episodes)
# ===========================================================================
LAB_SPREADS = {"10y-3m": "spread_10y_3m", "10y-ff": "spread_10y_ff", "10y-2y": "spread_10y_2y"}
LAB_NOTE = {"10y-3m": "default headline (NY Fed standard)",
            "10y-ff": "highest OOS; leans on policy rate (Wright 2006)",
            "10y-2y": "short history (2y from 1976)"}


def spread_model(name, col):
    """Full-sample probit coefficients + honest expanding-window OOS AUC/onsets for one spread."""
    yy = df["recession"].shift(-H)
    d = pd.DataFrame({"x": df[col], "y": yy}).dropna()
    dts = d.index; preds, act, td = [], [], []
    for i in range(INITIAL_TRAIN_N, len(d)):
        t = dts[i]; m = dts <= (t - pd.DateOffset(months=H))
        if m.sum() < INITIAL_TRAIN_N or d.loc[m, "y"].nunique() < 2:
            continue
        try:
            r = sm.Probit(d.loc[m, "y"], sm.add_constant(d.loc[m, "x"], has_constant="add")).fit(disp=0, maxiter=200)
            preds.append(float(r.predict(sm.add_constant(pd.Series([d["x"].iloc[i]], name="x"), has_constant="add"))[0]))
            act.append(d["y"].iloc[i]); td.append(t)
        except Exception:
            continue
    auc = roc_auc_score(act, preds)
    lo, hi = min(td) + pd.DateOffset(months=H), max(td) + pd.DateOffset(months=H)
    on = int(((onsets >= lo) & (onsets <= hi)).sum())
    rf = sm.Probit(d["y"], sm.add_constant(d["x"], has_constant="add")).fit(disp=0, maxiter=200)
    return {"const": round(float(rf.params["const"]), 4), "slope": round(float(rf.params["x"]), 4),
            "oos_auc": round(auc, 3), "onsets": on,
            "current": round(float(df[col].dropna().iloc[-1]), 2), "note": LAB_NOTE[name]}

lab_models = {name: spread_model(name, col) for name, col in LAB_SPREADS.items()}

# --- Step-1 scorecard stats (Phase 4D): inversion episodes -> false positives + lead time ---
MERGE_GAP, LEAD_WINDOW = 3, 18
def _episodes(col):
    s = df[col].dropna(); inv = s < 0
    eps, start, prev, gap = [], None, None, 0
    for d_, isinv in inv.items():
        if isinv:
            if start is None: start = d_
            prev = d_; gap = 0
        elif start is not None:
            gap += 1
            if gap > MERGE_GAP: eps.append((start, prev)); start = None
    if start is not None: eps.append((start, prev))
    return eps
def _scorecard(col):
    eps = _episodes(col)
    a0, a1 = df[col].dropna().index.min(), df[col].dropna().index.max()
    onin = [o for o in onsets if a0 <= o <= a1]
    leads, fp, caught = [], 0, set()
    for (a, b) in eps:
        nxt = [o for o in onsets if 0 < (o.year-a.year)*12+(o.month-a.month) <= LEAD_WINDOW]
        if nxt:
            o = min(nxt); caught.add(o); leads.append((o.year-a.year)*12+(o.month-a.month))
        else: fp += 1
    return {"inversions": len(eps), "false_pos": fp,
            "hit_rate": round(len(caught)/len(onin), 2) if onin else None,
            "lead_mean": round(float(np.mean(leads)), 1) if leads else None,
            "lead_min": min(leads) if leads else None, "lead_max": max(leads) if leads else None}
for nm, col in LAB_SPREADS.items():
    lab_models[nm].update(_scorecard(col))

# current yields (slider defaults for the four-yield curve input)
def last(c): return round(float(df[c].dropna().iloc[-1]), 2)
# lab sliders default to TODAY's live daily curve (so the opening spread + probability match the
# Overview); fall back to the latest monthly yields if the live file is unavailable.
if LIVE:
    lab_curve = {"y3m": LIVE["y3m"], "y2": LIVE["y2"], "y10": LIVE["y10"], "fedfunds": LIVE["fedfunds"]}
else:
    lab_curve = {"y3m": last("y3m"), "y2": last("y2"), "y10": last("y10"), "fedfunds": last("fedfunds")}

# current fragility values (slider defaults) + ranges + thresholds
lab_frag = {
    "debt_income": {"value": last("hh_debt_income"), "min": 60, "max": 140, "warn_above": THRESHOLDS["leverage"]["value"],
                    "label": "Household debt-to-income (%)"},
    "real_house":  {"value": round(float(cv["real_hpi_yoy_lag2"]), 2), "min": -15, "max": 15, "warn_below": THRESHOLDS["house"]["value"],
                    "label": "Real house prices YoY (%)"},
    "credit":      {"value": last("credit_spread"),  "min": 0.5, "max": 4.0, "warn_above": THRESHOLDS["credit"]["value"],
                    "label": "Credit spread Baa-10yr (pp)"},
    "debt_growth": {"value": last("hh_debt_growth"), "min": -5, "max": 15, "warn_above": None,
                    "label": "Household debt growth YoY (%)"},
    "sloos":       {"value": last("sloos"),          "min": -25, "max": 60, "warn_above": THRESHOLDS["banks"]["value"],
                    "label": "Banks tightening, SLOOS (net %)"},
    "migration":   {"value": last("foreign_born_growth"), "min": -5, "max": 10, "warn_above": None,
                    "label": "Foreign-born labor force growth YoY (%)"},
}

# 8 reference episodes: the full scenario vector for the "most resembles [year]" match
EPISODES_LAB = {
    "1980-12-01": ("1980 (Volcker)",     "recession 1981-82 (policy-induced)"),
    "1989-06-01": ("1989",               "recession 1990-91"),
    "1998-09-01": ("1998 (LTCM)",        "no recession"),
    "2000-12-01": ("2000 (dot-com)",     "recession 2001 (tech bust)"),
    "2004-06-01": ("2004",               "no recession (healthy)"),
    "2007-03-01": ("2007 (GFC run-up)",  "recession 2008 (credit-cycle)"),
    "2019-08-01": ("2019",               "COVID recession (exogenous)"),
    "2023-05-01": ("2023 (2022-24)",     "no recession"),
}
# variables in the match vector (spread PLUS fragility), per the brief
MATCH_VARS = ["spread_10y_3m", "real_hpi_yoy", "credit_spread", "hh_debt_income",
              "hh_debt_growth", "sloos", "foreign_born_growth"]
MATCH_LABEL = {"spread_10y_3m": "the yield curve", "real_hpi_yoy": "real house prices",
               "credit_spread": "the credit spread", "hh_debt_income": "household debt",
               "hh_debt_growth": "debt growth", "sloos": "bank lending", "foreign_born_growth": "labor-force growth"}
episodes = []
for d_, (label, outcome) in EPISODES_LAB.items():
    row = df.loc[pd.Timestamp(d_)]
    vec = {v: (None if pd.isna(row[v]) else round(float(row[v]), 2)) for v in MATCH_VARS}
    episodes.append({"label": label, "outcome": outcome, "vec": vec})
# standardization stats (mean/std per variable across episodes, ignoring missing)
match_stats = {}
for v in MATCH_VARS:
    vals = df.loc[[pd.Timestamp(d_) for d_ in EPISODES_LAB], v].dropna()
    match_stats[v] = {"mean": round(float(vals.mean()), 3), "std": round(float(vals.std() or 1.0), 3)}

lab = {"models": lab_models, "curve_defaults": lab_curve, "fragility": lab_frag,
       "episodes": episodes, "match_vars": MATCH_VARS, "match_labels": MATCH_LABEL,
       "match_stats": match_stats}

# ===========================================================================
# PHASE 6D: recession conditions matrix (built from live data; shared thresholds)
# ===========================================================================
# Columns: two labeled blocks. LEADING = present before a recession; COINCIDENT =
# widen DURING/AFTER, so they often read calm at the inversion (2007 is the example).
MATRIX_COLS = [
    ("inverted", "spread_10y_3m",   "Curve inverted",       "10y-3m, pp",  "leading",   lambda v: f"{v:+.2f}"),
    ("leverage", "hh_debt_income",  "Leverage high",        "debt/income", "leading",   lambda v: f"{v:.0f}%"),
    ("house",    "real_hpi_yoy",    "House prices falling", "real YoY",    "leading",    lambda v: f"{v:+.1f}%"),
    ("credit",   "credit_spread",   "Credit stress",        "Baa-10yr",    "coincident", lambda v: f"{v:.2f}"),
    ("banks",    "sloos",           "Banks tightening",     "SLOOS net %", "coincident", lambda v: f"{v:+.0f}"),
]

def matrix_cell(key, col, fmt, row):
    v = row[col]
    if pd.isna(v):
        return {"value": "—", "danger": False, "na": True}
    return {"value": fmt(float(v)), "danger": bool(is_danger(key, float(v))), "na": False}

matrix_rows = []

# Early yield-curve era (1953+). One row per recession, ref = deepest / least-positive
# 10y-3m in the 24 months before onset. Honest point: at MONTHLY resolution the curve did
# NOT invert before 1957/1960 (only flattened to ~+0.2) - shown GREEN with a recession
# outcome, not fudged red. These were low-leverage Fed-tightening/oil-shock recessions.
EARLY = [
    ("1957",    "1957-08-01", "Recession 1957-58 (Fed tightening)"),
    ("1960",    "1960-04-01", "Recession 1960-61 (Fed tightening)"),
    ("1970",    "1969-12-01", "Recession 1970 (Fed tightening)"),
    ("1973-75", "1973-11-01", "Recession 1973-75 (oil shock)"),
]
for label, onset, outcome in EARLY:
    o = pd.Timestamp(onset)
    w = df.loc[o - pd.DateOffset(months=24):o - pd.DateOffset(months=1), "spread_10y_3m"].dropna()
    r = df.loc[w.idxmin()]
    matrix_rows.append({
        "label": label, "outcome": outcome, "outcome_rec": True, "today": False,
        "cells": {k: matrix_cell(k, col, fmt, r) for (k, col, _, _, _, fmt) in MATRIX_COLS},
    })

for d_, (label, outcome) in EPISODES_LAB.items():
    r = df.loc[pd.Timestamp(d_)]
    rec_followed = ("recession" in outcome.lower()) and ("no recession" not in outcome.lower())
    matrix_rows.append({
        "label": label, "outcome": outcome, "outcome_rec": rec_followed, "today": False,
        "cells": {k: matrix_cell(k, col, fmt, r) for (k, col, _, _, _, fmt) in MATRIX_COLS},
    })
# the live "Today" row, from the latest monthly data (uses the same as-of basis as the lights)
today = {"spread_10y_3m": float(cv["spread_10y_3m"]), "hh_debt_income": float(cv["hh_debt_income"]),
         "real_hpi_yoy": float(cv["real_hpi_yoy_lag2"]), "credit_spread": float(cv["credit_spread"]),
         "sloos": (float(df["sloos"].dropna().iloc[-1]) if df["sloos"].notna().any() else float("nan"))}
matrix_rows.append({
    "label": f"TODAY ({latest_date.strftime('%b %Y')})", "outcome": short, "outcome_rec": None, "today": True,
    "cells": {k: matrix_cell(k, col, fmt, pd.Series(today)) for (k, col, _, _, _, fmt) in MATRIX_COLS},
})
matrix = {
    "cols": [{"key": k, "label": lab_, "sub": sub, "block": blk} for (k, _, lab_, sub, blk, _) in MATRIX_COLS],
    "rows": matrix_rows,
}

# shared threshold config, exported so the front-end uses the SAME numbers everywhere
config_out = {"thresholds": {k: {"value": t["value"], "dir": t["dir"], "label": t["label"]}
                             for k, t in THRESHOLDS.items()}}

# ---- Assemble JSON --------------------------------------------------------
plot_dates = list(s_all.index)
data = {
    "as_of": latest_date.strftime("%Y-%m"),
    "gauge": {"probability": (int(round(gauge_prob)) if float(gauge_prob).is_integer()
                              else round(gauge_prob, 1)), "label": gauge_label},
    "character": {"short": short, "full": full, "class": cls},
    "lights": lights,
    "auc": {"realtime": round(rt_auc, 3), "onsets": rt_onsets, "insample": round(is_auc, 3),
            "rt_start": ev.index.min().strftime("%Y"), "rt_end": ev.index.max().strftime("%Y")},
    "series": {
        "dates":    [d.strftime("%Y-%m") for d in plot_dates],
        "fitted":   [round(fitted[d] * 100, 2) for d in plot_dates],
        "realtime": [round(realtime[d] * 100, 2) if d in realtime else None for d in plot_dates],
    },
    "recessions": recessions,
    "lab": lab,
    "matrix": matrix,
    "config": config_out,
}

with open("output/dashboard_data.json", "w") as f:
    json.dump(data, f, indent=1)

# ---- Inject into the template -> the interactive "Explore" page (docs/explore.html) --------
tpl = open("dashboard_template.html").read()
html = tpl.replace("__DASHBOARD_DATA__", json.dumps(data))
os.makedirs("docs", exist_ok=True)
for p in ("docs/explore.html", "output/explore.html"):
    with open(p, "w") as f:
        f.write(html)

print("Phase 6 data built.")
print(f"  as of           : {data['as_of']}")
print(f"  gauge (headline): {data['gauge']['probability']}%  ({data['gauge']['label']})")
print(f"  character       : {short}")
print(f"  real-time AUC   : {data['auc']['realtime']}  ({rt_onsets} onsets, {data['auc']['rt_start']}-{data['auc']['rt_end']})")
print(f"  in-sample AUC   : {data['auc']['insample']}  (fitted line, the rosier one)")
print(f"  months plotted  : {len(plot_dates)}   real-time points: {len(realtime)}   recessions shaded: {len(recessions)}")
print("  LAB models (const / slope / OOS / onsets / current):")
for nm, m in lab_models.items():
    print(f"     {nm:7s}: {m['const']:+.3f} / {m['slope']:+.3f} / AUC {m['oos_auc']} / {m['onsets']} onsets / spread {m['current']:+.2f}")
print(f"  LAB episodes: {len(episodes)}   fragility sliders: {len(lab_frag)}")
print("  wrote output/dashboard_data.json and output/dashboard.html")
