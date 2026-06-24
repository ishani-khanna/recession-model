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
print("  wrote output/dashboard_data.json and output/dashboard.html")
