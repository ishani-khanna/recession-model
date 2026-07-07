"""
build_two_clock_data.py  -  export data for the THREE-SIGNAL RULE panel.

VISUALIZATION layer: runs the yield_curve engine once and reads what its own functions output
(reporting.two_clock_dashboard, reporting.term_premium_diagnostic) - it does NOT re-derive any
model math. Organizes the three signals of the rule and writes output/two_clock_data.json, then
injects it into two_clock_template.html to produce a self-contained docs/two_clock.html.

The three-signal rule (in-sample-tuned; the probabilities themselves are OOS-validated):
  1. 12-month outlook  - flagship (term spread)              fires when >= 30%
  2.  3-month watch     - augmented (term + Baa-Aa credit)    fires when >= 15%
  3. trust check        - term premium                        met (clear) when >= 50 bp
The rule calls a recession ONLY when all three confirm. We report how many of 3 are met.

DataBuffet: wired but off today (FRED-only). Signal 2's Aa leg substitutes FRED Aaa until the
Moody's keys are set; the data-source line reflects which is live.
"""

import json
import os
import warnings

import numpy as np
import pandas as pd

from yield_curve.data import build_dataset, config
from yield_curve.data.fred_client import FredClient
from yield_curve.models import probit
from yield_curve import reporting
from scipy.stats import norm

warnings.filterwarnings("ignore")

# thresholds of the rule (fixed; the values below refresh from the engine)
FLAG_FIRE, AUG_FIRE, TP_GATE_BP = 30, 15, 50


def _databuffet_on():
    try:
        config.databuffet_keys()
        return True
    except Exception:
        return False


# --- run the engine once -----------------------------------------------------
panel = build_dataset.build_panel(refresh=False)
curve = reporting.fetch_current_curve()
dspreads = reporting.current_spreads_daily(curve)
term_current, term_asof = dspreads["10y3m"]

clocks = reporting.two_clock_dashboard(panel, term_current)   # flag12, aug3, term, credit, credit_date
tp = reporting.term_premium_diagnostic(term_current)          # Kim-Wright term-premium read

db_on = _databuffet_on()
tp_bp = round(tp["tp_current"] * 100)

# the three signals
s1_met = clocks["flag12"] >= FLAG_FIRE
s2_met = clocks["aug3"] >= AUG_FIRE
s3_met = tp_bp >= TP_GATE_BP        # trust check "met" = NOT compressed (no veto)
count_met = int(s1_met) + int(s2_met) + int(s3_met)
firing = count_met == 3

# --- history series (term premium + flagship probability, for the chart) ----
tp_m = FredClient(pause=0.1).get_series("THREEFYTP10").dropna().resample("MS").mean()
fr = probit.fit_spread_model(panel, "10y3m", 12, "probit")
spread = panel["spread_10y3m"].dropna()
flag_prob = pd.Series(norm.cdf(fr.intercept + fr.coef * spread.values) * 100, index=spread.index)

start = tp_m.index.min()
idx = pd.date_range(start=start, end=max(tp_m.index.max(), flag_prob.index.max()), freq="MS")
tp_series = tp_m.reindex(idx)
flag_series = flag_prob.reindex(idx)
tp_2022_23 = float(tp_m.loc["2022-01":"2023-12"].min())
tp_2022_23_date = tp_m.loc["2022-01":"2023-12"].idxmin()

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
    "rule": {"count_met": count_met, "firing": firing,
             "thresholds": {"flag": FLAG_FIRE, "aug": AUG_FIRE, "tp_bp": TP_GATE_BP}},
    "signals": [
        {"key": "flag", "name": "12-month outlook", "model": "flagship — term spread",
         "sub": "is a recession building this year?", "value": clocks["flag12"], "unit": "%",
         "fires_at": FLAG_FIRE, "fires_text": "fires ≥ 30%", "met": s1_met, "gauge": True},
        {"key": "aug", "name": "3-month watch", "model": "augmented — term + Baa−Aa credit",
         "sub": "is one arriving next quarter?", "value": clocks["aug3"], "unit": "%",
         "fires_at": AUG_FIRE, "fires_text": "fires ≥ 15%", "met": s2_met, "gauge": True,
         "source_note": ("Aa leg: Moody's DataBuffet (IRAACM.IUSA)" if db_on
                         else "Aa leg: FRED Aaa substitute (Moody's DataBuffet off)")},
        {"key": "tp", "name": "Trust check", "model": "term premium (Kim–Wright)",
         "sub": "is the curve signal clean, or compressed?", "value_bp": tp_bp,
         "gate_bp": TP_GATE_BP, "fires_text": "clear ≥ 50 bp; vetoes if compressed (< 50 bp)",
         "met": s3_met, "gauge": False, "tp_class": tp["tp_class"], "percentile": tp["tp_percentile"],
         "acm_note": "the report's headline uses the ACM term premium (~+67 bp, ~33rd pctile), which agrees"},
    ],
    "term_premium": {
        "tp_current": tp["tp_current"], "tp_bp": tp_bp, "tp_class": tp["tp_class"],
        "tp_percentile": tp["tp_percentile"], "tp_date": tp["tp_date"],
        "tp_compressed_2020_21": tp["tp_compressed_2020_21"],
        "tp_2022_23": round(tp_2022_23, 2), "tp_2022_23_date": tp_2022_23_date.strftime("%Y-%m"),
        "gate_pp": TP_GATE_BP / 100.0,
    },
    "series": {
        "dates": [d.strftime("%Y-%m") for d in idx],
        "term_premium": [None if pd.isna(v) else round(float(v), 3) for v in tp_series.values],
        "flagship_prob": [None if pd.isna(v) else round(float(v), 1) for v in flag_series.values],
    },
    "recessions": recessions,
    "databuffet_on": db_on,
    "data_source": ("FRED + Moody's DataBuffet" if db_on
                    else "FRED only (Aa leg substituted with FRED Aaa; Moody's DataBuffet off)"),
}

os.makedirs("output", exist_ok=True)
with open("output/two_clock_data.json", "w") as f:
    json.dump(data, f, indent=1)

if os.path.exists("two_clock_template.html"):
    tpl = open("two_clock_template.html").read()
    html = tpl.replace("__TWO_CLOCK_DATA__", json.dumps(data))
    os.makedirs("docs", exist_ok=True)
    with open("docs/two_clock.html", "w") as f:
        f.write(html)
    with open("output/two_clock.html", "w") as f:
        f.write(html)

print("Three-signal panel data built.")
print(f"  as of daily : {data['as_of_daily']}   data source: {data['data_source']}")
print(f"  headline    : {'FIRING' if firing else 'NOT firing'} - {count_met} of 3 conditions met")
print(f"  1) flagship 12mo = {clocks['flag12']}%  (fires >= {FLAG_FIRE})  met={s1_met}")
print(f"  2) augmented 3mo = {clocks['aug3']}%  (fires >= {AUG_FIRE})  met={s2_met}")
print(f"  3) trust check   = {tp_bp} bp  ({tp['tp_class']}, ~{tp['tp_percentile']}th pct; gate {TP_GATE_BP} bp)  met={s3_met}")
