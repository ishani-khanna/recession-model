"""
build_two_clock_data.py  -  export data for the SINGLE report-matching dashboard (docs/index.html).

Mirrors the generated report (report.pdf): led by the THREE-SIGNAL RULE, then today's curve
inputs, the probability-by-horizon table, the term-premium trust check + history, the flagship
probability trajectory, and the caveats. VISUALIZATION only: every number is read from the
yield_curve engine (reporting.*), nothing is re-derived. Writes output/two_clock_data.json and
injects it into report_template.html to produce the self-contained docs/index.html.

The three-signal rule (in-sample-tuned; the probabilities themselves are OOS-validated):
  1. 12-month outlook - flagship (term spread)            fires >= 30%
  2.  3-month watch    - augmented (term + Baa-Aa credit)  fires >= 15%
  3. trust check       - term premium                      clear >= 50 bp; vetoes if compressed
The rule calls a recession only when all three confirm.
"""

import datetime as dt
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

from yield_curve.data import build_dataset, config
from yield_curve.data.fred_client import FredClient
from yield_curve.models import probit
from yield_curve import reporting

warnings.filterwarnings("ignore")

FLAG_FIRE, AUG_FIRE, TP_GATE_BP = 30, 15, 50


def _databuffet_on():
    try:
        config.databuffet_keys(); return True
    except Exception:
        return False


def _rec_bands(usrec):
    u = usrec.dropna().astype(int)
    spans, inrec, s0 = [], False, None
    for d, v in u.items():
        if v == 1 and not inrec:
            inrec, s0 = True, d
        elif v == 0 and inrec:
            spans.append((s0, d)); inrec = False
    if inrec:
        spans.append((s0, u.index[-1]))
    return [{"start": s.strftime("%Y-%m"), "end": e.strftime("%Y-%m")} for s, e in spans]


# --- run the engine once -----------------------------------------------------
panel = build_dataset.build_panel(refresh=False)
curve = reporting.fetch_current_curve()
dspreads = reporting.current_spreads_daily(curve)
term_current, term_asof = dspreads["10y3m"]
spread_vals = {k: v for k, (v, _) in dspreads.items()}

clocks = reporting.two_clock_dashboard(panel, term_current)
tp = reporting.term_premium_diagnostic(term_current)
db_on = _databuffet_on()
tp_bp = round(tp["tp_current"] * 100)

s1_met = clocks["flag12"] >= FLAG_FIRE
s2_met = clocks["aug3"] >= AUG_FIRE
s3_met = tp_bp >= TP_GATE_BP
count_met = int(s1_met) + int(s2_met) + int(s3_met)
firing = count_met == 3

# --- today's curve inputs: the three spreads (daily + latest monthly) --------
spreads_table = []
for k in config.SPREADS:
    col = "spread_" + k
    monthly = round(float(panel[col].dropna().iloc[-1]), 2)
    daily = round(float(spread_vals[k]), 2)
    spreads_table.append({"label": config.SPREADS[k]["label"], "daily": daily,
                          "monthly": monthly, "inverted": daily < 0})

# --- recession probability by spread x horizon -------------------------------
pt = reporting.probability_table(panel, spread_vals)
prob_table = {"horizons": [int(h) for h in pt.index],
              "spreads": [{"key": k, "label": config.SPREADS[k]["label"],
                           "vals": [round(float(v), 1) for v in pt[k].values]} for k in pt.columns]}

# --- flagship 12-month probability trajectory (full history, for the chart) --
fr = probit.fit_spread_model(panel, "10y3m", 12, "probit")
spread = panel["spread_10y3m"].dropna()
flag_full = pd.Series(norm.cdf(fr.intercept + fr.coef * spread.values) * 100, index=spread.index)

# --- term-premium history (Kim-Wright) + flagship overlay (1990+) -------------
tp_m = FredClient(pause=0.1).get_series("THREEFYTP10").dropna().resample("MS").mean()
tp_idx = pd.date_range(start=tp_m.index.min(), end=max(tp_m.index.max(), flag_full.index.max()), freq="MS")
tp_series = tp_m.reindex(tp_idx)
flag_tp = flag_full.reindex(tp_idx)
tp_2022_23 = float(tp_m.loc["2022-01":"2023-12"].min())
tp_2022_23_date = tp_m.loc["2022-01":"2023-12"].idxmin()

plain = (f"the curve is {'positively sloped' if term_current >= 0 else 'inverted'} "
         f"(10Y−3M {term_current:+.2f} pp); credit is {'calm' if clocks['credit'] < 0.7 else 'elevated'} "
         f"(Baa−Aa {clocks['credit']:+.2f} pp); the term premium is "
         f"{'not compressed' if s3_met else 'compressed'} ({tp_bp:+d} bp, ~{tp['tp_percentile']}th percentile).")

data = {
    "generated": dt.date.today().isoformat(),
    "as_of_daily": (term_asof.date().isoformat() if term_asof is not None else None),
    "monthly_through": panel.index.max().strftime("%Y-%m"),
    "rule": {"count_met": count_met, "firing": firing,
             "thresholds": {"flag": FLAG_FIRE, "aug": AUG_FIRE, "tp_bp": TP_GATE_BP}},
    "plain_english": plain,
    "signals": [
        {"key": "flag", "num": 1, "name": "12-month outlook", "model": "flagship — term spread",
         "sub": "is a recession building over the next year?", "value": clocks["flag12"], "unit": "%",
         "fires_text": "fires ≥ 30%", "met": s1_met, "gauge": True},
        {"key": "aug", "num": 2, "name": "3-month watch", "model": "augmented — term + Baa−Aa credit",
         "sub": "is one arriving in the next quarter?", "value": clocks["aug3"], "unit": "%",
         "fires_text": "fires ≥ 15%", "met": s2_met, "gauge": True,
         "source_note": ("Aa leg: Moody's DataBuffet (IRAACM.IUSA)" if db_on
                         else "Aa leg: FRED Aaa substitute (Moody's DataBuffet off)")},
        {"key": "tp", "num": 3, "name": "Trust check", "model": "term premium (Kim–Wright)",
         "sub": "is the curve's signal trustworthy, or distorted by a compressed premium?",
         "value_bp": tp_bp, "fires_text": "clear ≥ 50 bp; vetoes if compressed (< 50 bp)",
         "met": s3_met, "gauge": False, "tp_class": tp["tp_class"], "percentile": tp["tp_percentile"],
         "acm_note": "the report's headline uses the ACM term premium (~+67 bp, ~33rd pctile), which agrees"},
    ],
    "inputs": {"primary_daily": round(term_current, 2),
               "primary_monthly": spreads_table[0]["monthly"], "spreads": spreads_table},
    "prob_table": prob_table,
    "term_premium": {
        "tp_current": tp["tp_current"], "tp_bp": tp_bp, "tp_class": tp["tp_class"],
        "tp_percentile": tp["tp_percentile"], "tp_date": tp["tp_date"],
        "tp_compressed_2020_21": tp["tp_compressed_2020_21"],
        "tp_2022_23": round(tp_2022_23, 2), "tp_2022_23_date": tp_2022_23_date.strftime("%Y-%m"),
        "gate_pp": TP_GATE_BP / 100.0,
    },
    "tp_series": {
        "dates": [d.strftime("%Y-%m") for d in tp_idx],
        "term_premium": [None if pd.isna(v) else round(float(v), 3) for v in tp_series.values],
        "flagship_prob": [None if pd.isna(v) else round(float(v), 1) for v in flag_tp.values],
    },
    "trajectory": {
        "dates": [d.strftime("%Y-%m") for d in flag_full.index],
        "prob": [round(float(v), 1) for v in flag_full.values],
        "recessions": _rec_bands(panel["usrec"]),
    },
    "tp_recessions": _rec_bands(panel["usrec"].reindex(tp_idx).fillna(0)),
    "databuffet_on": db_on,
    "data_source": ("FRED + Moody's DataBuffet" if db_on
                    else "FRED only (Aa leg substituted with FRED Aaa; Moody's DataBuffet off)"),
}

os.makedirs("output", exist_ok=True)
with open("output/two_clock_data.json", "w") as f:
    json.dump(data, f, indent=1)

tpl = open("report_template.html").read()
html = tpl.replace("__REPORT_DATA__", json.dumps(data))
os.makedirs("docs", exist_ok=True)
with open("docs/index.html", "w") as f:
    f.write(html)
with open("output/index.html", "w") as f:
    f.write(html)

print("Single report-dashboard built (docs/index.html).")
print(f"  as of daily : {data['as_of_daily']}   source: {data['data_source']}")
print(f"  headline    : {'FIRING' if firing else 'NOT firing'} - {count_met} of 3 conditions met")
print(f"  1) flagship 12mo = {clocks['flag12']}%  met={s1_met}")
print(f"  2) augmented 3mo = {clocks['aug3']}%  met={s2_met}")
print(f"  3) trust check   = {tp_bp} bp ({tp['tp_class']}, ~{tp['tp_percentile']}th pct)  met={s3_met}")
print(f"  spreads/prob/traj: {len(spreads_table)} spreads, {len(prob_table['horizons'])} horizons, {len(flag_full)} traj months")
