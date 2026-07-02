"""
build_two_clock_data.py  -  export data for the two-clock + term-premium tracker panel.

This is a VISUALIZATION layer: it runs the yield_curve engine once and reads what its own
functions output (reporting.two_clock_dashboard, reporting.term_premium_diagnostic) - it does
NOT re-derive any model math. Writes output/two_clock_data.json, then injects it into
two_clock_template.html to produce a self-contained docs/two_clock.html (published by Pages).
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

from yield_curve.data import build_dataset
from yield_curve.data.fred_client import FredClient
from yield_curve.models import probit
from yield_curve import reporting

warnings.filterwarnings("ignore")

# --- run the engine once -----------------------------------------------------
panel = build_dataset.build_panel(refresh=False)
curve = reporting.fetch_current_curve()
dspreads = reporting.current_spreads_daily(curve)
term_current, term_asof = dspreads["10y3m"]

# the two clocks and the term-premium read come STRAIGHT from the engine
clocks = reporting.two_clock_dashboard(panel, term_current)          # flag12, aug3, term, credit, credit_date
tp = reporting.term_premium_diagnostic(term_current)                 # tp_current, tp_class, tp_percentile, ...

# --- history series (for the tracker chart) ---------------------------------
# 10Y term premium, monthly (FRED THREEFYTP10, 1990+)
tp_m = FredClient(pause=0.1).get_series("THREEFYTP10").dropna().resample("MS").mean()
# flagship 12-month probability over history (full-sample probit on 10y-3m), same as reporting
fr = probit.fit_spread_model(panel, "10y3m", 12, "probit")
spread = panel["spread_10y3m"].dropna()
flag_prob = pd.Series(norm.cdf(fr.intercept + fr.coef * spread.values) * 100, index=spread.index)

# common monthly axis from the term-premium start (the era the false-alarm story is about)
start = tp_m.index.min()
idx = pd.date_range(start=start, end=max(tp_m.index.max(), flag_prob.index.max()), freq="MS")
tp_series = tp_m.reindex(idx)
flag_series = flag_prob.reindex(idx)

# the 2022-23 term-premium low (the compressed / false-alarm signature marker)
tp_2022_23 = float(tp_m.loc["2022-01":"2023-12"].min())
tp_2022_23_date = tp_m.loc["2022-01":"2023-12"].idxmin()

# NBER recession bands within the charted window
rec = panel["usrec"].reindex(idx).fillna(0).astype(int)
spans, inrec, s0 = [], False, None
for d, v in rec.items():
    if v == 1 and not inrec:
        inrec, s0 = True, d
    elif v == 0 and inrec:
        spans.append((s0, d)); inrec = False
if inrec:
    spans.append((s0, rec.index[-1]))
recessions = [{"start": s.strftime("%Y-%m"), "end": e.strftime("%Y-%m")} for s, e in spans]

# --- assemble JSON ----------------------------------------------------------
data = {
    "as_of_daily": (term_asof.date().isoformat() if term_asof is not None else None),
    "clocks": {
        "flag12": clocks["flag12"], "aug3": clocks["aug3"],
        "flag_label": "12-month outlook", "flag_sub": "is a recession building this year?",
        "aug_label": "3-month watch", "aug_sub": "is one arriving next quarter?",
        "term": clocks["term"], "credit": clocks["credit"], "credit_date": clocks["credit_date"],
    },
    "term_premium": {
        "tp_current": tp["tp_current"], "tp_class": tp["tp_class"],
        "tp_percentile": tp["tp_percentile"], "tp_date": tp["tp_date"],
        "tp_compressed_2020_21": tp["tp_compressed_2020_21"],
        "tp_2022_23": round(tp_2022_23, 2), "tp_2022_23_date": tp_2022_23_date.strftime("%Y-%m"),
        "exp_component": tp["exp_component"], "spread": tp["spread"],
    },
    "series": {
        "dates": [d.strftime("%Y-%m") for d in idx],
        "term_premium": [None if pd.isna(v) else round(float(v), 3) for v in tp_series.values],
        "flagship_prob": [None if pd.isna(v) else round(float(v), 1) for v in flag_series.values],
    },
    "recessions": recessions,
}

os.makedirs("output", exist_ok=True)
with open("output/two_clock_data.json", "w") as f:
    json.dump(data, f, indent=1)

# inject into the template -> self-contained docs/two_clock.html
if os.path.exists("two_clock_template.html"):
    tpl = open("two_clock_template.html").read()
    html = tpl.replace("__TWO_CLOCK_DATA__", json.dumps(data))
    os.makedirs("docs", exist_ok=True)
    with open("docs/two_clock.html", "w") as f:
        f.write(html)
    with open("output/two_clock.html", "w") as f:
        f.write(html)

print("Two-clock data built.")
print(f"  as of daily     : {data['as_of_daily']}")
print(f"  clocks          : flagship 12mo = {clocks['flag12']}%  |  augmented 3mo = {clocks['aug3']}%")
print(f"  term premium    : {tp['tp_current']:+.2f} pp  ({tp['tp_class']}, ~{tp['tp_percentile']}th pct)")
print(f"  compressed 20-21: {tp['tp_compressed_2020_21']:+.2f}   2022-23 low: {tp_2022_23:+.2f} ({tp_2022_23_date.strftime('%Y-%m')})")
print(f"  series months   : {len(idx)} ({idx.min().strftime('%Y-%m')}..{idx.max().strftime('%Y-%m')}); recessions: {len(recessions)}")
