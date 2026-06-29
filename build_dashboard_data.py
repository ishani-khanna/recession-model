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
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

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

# gauge = full-sample model's probability at the latest spread ("all data through")
latest_date = s_all.index.max()
gauge_prob = probit_predict(res_full, s_all.loc[latest_date]) * 100

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
lights = [
    {"name": "Curve inverted",            "value": f"{cv['spread_10y_3m']:+.2f} pp",
     "threshold": "warns if < 0.00 pp",   "on": bool(cv["inverted"])},
    {"name": "Household debt high",        "value": f"{cv['hh_debt_income']:.0f}% of income",
     "threshold": "warns if > 110%",      "on": bool(cv["debt_high"])},
    {"name": "Real house prices falling",  "value": ("n/a" if pd.isna(rh) else f"{rh:+.1f}% YoY"),
     "threshold": "warns if < 0%",        "on": bool(cv["house_falling"])},
    {"name": "Credit spread spiking",      "value": f"{cv['credit_spread']:.2f} pp",
     "threshold": "warns if > 2.50 pp",   "on": bool(cv["credit_spiking"])},
]

# character banner
full = str(cv["verdict"])
if "GENUINE WARNING" in full:
    cls, short = "warn", "GENUINE WARNING"
elif "NO CREDIT-CYCLE FRAGILITY" in full:
    cls, short = "nocc", "INVERTED — NO CREDIT-CYCLE FRAGILITY"
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

# current yields (slider defaults for the four-yield curve input)
def last(c): return round(float(df[c].dropna().iloc[-1]), 2)
lab_curve = {"y3m": last("y3m"), "y2": last("y2"), "y10": last("y10"), "fedfunds": last("fedfunds")}

# current fragility values (slider defaults) + ranges + thresholds
lab_frag = {
    "debt_income": {"value": last("hh_debt_income"), "min": 60, "max": 140, "warn_above": 110,
                    "label": "Household debt-to-income (%)"},
    "real_house":  {"value": last("real_hpi_yoy"),   "min": -15, "max": 15, "warn_below": 0,
                    "label": "Real house prices YoY (%)"},
    "credit":      {"value": last("credit_spread"),  "min": 0.5, "max": 4.0, "warn_above": 2.5,
                    "label": "Credit spread Baa-10yr (pp)"},
    "debt_growth": {"value": last("hh_debt_growth"), "min": -5, "max": 15, "warn_above": None,
                    "label": "Household debt growth YoY (%)"},
    "sloos":       {"value": last("sloos"),          "min": -25, "max": 60, "warn_above": 15,
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

# ---- Assemble JSON --------------------------------------------------------
plot_dates = list(s_all.index)
data = {
    "as_of": latest_date.strftime("%Y-%m"),
    "gauge": {"probability": round(gauge_prob, 1),
              "label": f"current model probability (all data through {latest_date.strftime('%b %Y')})"},
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
}

with open("output/dashboard_data.json", "w") as f:
    json.dump(data, f, indent=1)

# ---- Inject into the template to make a self-contained dashboard.html ------
tpl = open("dashboard_template.html").read()
html = tpl.replace("__DASHBOARD_DATA__", json.dumps(data))
with open("output/dashboard.html", "w") as f:
    f.write(html)
# also write index.html (the entry point GitHub Pages serves by default)
with open("output/index.html", "w") as f:
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
