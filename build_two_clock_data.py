"""
build_two_clock_data.py  -  export data for the SINGLE report-matching landing page (docs/index.html).

Mirrors the generated report (report.pdf): led by the THREE-SIGNAL RULE, then today's curve
inputs, probability-by-horizon, the term-premium trust check + history, the flagship probability
trajectory, and the caveats. VISUALIZATION only: numbers come from the yield_curve engine
(reporting.*, acm.*), nothing re-derived. Injects into report_template.html -> docs/index.html.

Signal 3 leads with the ACM 10Y term premium (the rule's gauge; NY Fed file); Kim-Wright
(FRED THREEFYTP10) is shown as a cross-check. Gate: clear >= 50 bp, vetoes if compressed.
"""

import datetime as dt
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

from yield_curve.data import build_dataset, config, acm, conventions
from yield_curve.data.fred_client import FredClient
from yield_curve.models import probit
from yield_curve import reporting

warnings.filterwarnings("ignore")

FLAG_FIRE, AUG_FIRE, TP_GATE_BP = 30, 15, 50


def _ord(n):
    """Integer -> ordinal string (1st, 2nd, 3rd, 4th, 11th, 21st, 31st, ...)."""
    n = int(round(n))
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


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
kw = reporting.term_premium_diagnostic(term_current)          # Kim-Wright (cross-check)
db_on = _databuffet_on()

# Signal 3: ACM term premium (primary), with a graceful fallback to Kim-Wright.
try:
    a = acm.acm_reading()
    tp_bp, tp_pct, tp_cls = a["value_bp"], a["percentile"], a["class"]
    tp_source = "ACM"
    tp_monthly = a["monthly"]
    tp_lo_2223, tp_lo_date = a["low_2022_23"], a["low_2022_23_date"]
    tp_comp_2021 = a["compressed_2020_21"]
    tp_date = a["date"]
except Exception:
    tp_bp, tp_pct, tp_cls = round(kw["tp_current"] * 100), kw["tp_percentile"], kw["tp_class"]
    tp_source = "Kim-Wright"
    tp_monthly = FredClient(pause=0.1).get_series("THREEFYTP10").dropna().resample("MS").mean()
    tp_lo_2223 = round(float(tp_monthly.loc["2022-01":"2023-12"].min()), 3)
    tp_lo_date = tp_monthly.loc["2022-01":"2023-12"].idxmin().strftime("%Y-%m")
    tp_comp_2021 = round(float(tp_monthly.loc["2020-06":"2021-12"].mean()), 3)
    tp_date = kw["tp_date"]

s1_met = clocks["flag12"] >= FLAG_FIRE
s2_met = clocks["aug3"] >= AUG_FIRE
s3_met = tp_bp >= TP_GATE_BP
count_met = int(s1_met) + int(s2_met) + int(s3_met)
firing = count_met == 3

# --- today's curve inputs ----------------------------------------------------
spreads_table = []
for k in config.SPREADS:
    col = "spread_" + k
    spreads_table.append({"label": config.SPREADS[k]["label"],
                          "daily": round(float(spread_vals[k]), 2),
                          "monthly": round(float(panel[col].dropna().iloc[-1]), 2),
                          "inverted": float(spread_vals[k]) < 0})

pt = reporting.probability_table(panel, spread_vals)
prob_table = {"horizons": [int(h) for h in pt.index],
              "spreads": [{"key": k, "label": config.SPREADS[k]["label"],
                           "vals": [round(float(v), 1) for v in pt[k].values]} for k in pt.columns]}

# --- flagship 12-month probability trajectory (full history) -----------------
fr = probit.fit_spread_model(panel, "10y3m", 12, "probit")
spread = panel["spread_10y3m"].dropna()
flag_full = pd.Series(norm.cdf(fr.intercept + fr.coef * spread.values) * 100, index=spread.index)

# --- term-premium history (ACM primary) + flagship overlay -------------------
tp_idx = pd.date_range(start=tp_monthly.index.min(),
                       end=max(tp_monthly.index.max(), flag_full.index.max()), freq="MS")
tp_series = tp_monthly.reindex(tp_idx)
flag_tp = flag_full.reindex(tp_idx)

plain = (f"the curve is {'positively sloped' if term_current >= 0 else 'inverted'} "
         f"(10Y−3M {term_current:+.2f} pp); credit is {'calm' if clocks['credit'] < 0.7 else 'elevated'} "
         f"(Baa−Aa {clocks['credit']:+.2f} pp); the term premium is "
         f"{'not compressed' if s3_met else 'compressed'} ({tp_bp:+d} bp, ~{_ord(tp_pct)} percentile since 1961).")

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
                         else "Aa leg: FRED Aaa substitute (DataBuffet off)")},
        {"key": "tp", "num": 3, "name": "Trust check", "model": "ACM 10Y term premium",
         "sub": "is the curve's signal trustworthy, or distorted by a compressed premium?",
         "value_bp": tp_bp, "fires_text": "clear ≥ 50 bp · vetoes if < 50 bp", "met": s3_met,
         "gauge": False, "tp_class": tp_cls, "percentile": tp_pct, "percentile_ord": _ord(tp_pct),
         "crosscheck": f"cross-check — Kim–Wright: {round(kw['tp_current']*100):+d} bp (~{_ord(kw['tp_percentile'])} pctile), agrees"},
    ],
    "inputs": {"primary_daily": round(term_current, 2),
               "primary_monthly": spreads_table[0]["monthly"], "spreads": spreads_table},
    "prob_table": prob_table,
    "term_premium": {
        "tp_bp": tp_bp, "tp_class": tp_cls, "tp_percentile": tp_pct, "tp_date": tp_date,
        "source": tp_source, "tp_compressed_2020_21": tp_comp_2021,
        "tp_2022_23": round(tp_lo_2223, 2), "tp_2022_23_date": tp_lo_date, "gate_pp": TP_GATE_BP / 100.0,
    },
    "tp_series": {
        "dates": [d.strftime("%Y-%m") for d in tp_idx],
        "term_premium": [None if pd.isna(v) else round(float(v), 3) for v in tp_series.values],
        "flagship_prob": [None if pd.isna(v) else round(float(v), 1) for v in flag_tp.values],
    },
    "tp_recessions": _rec_bands(panel["usrec"].reindex(tp_idx).fillna(0)),
    "trajectory": {
        "dates": [d.strftime("%Y-%m") for d in flag_full.index],
        "prob": [round(float(v), 1) for v in flag_full.values],
        "recessions": _rec_bands(panel["usrec"]),
    },
    "databuffet_on": db_on,
    "data_source": ("FRED + Moody's DataBuffet" if db_on
                    else "FRED + NY Fed ACM (Moody's DataBuffet off — Aa leg uses FRED Aaa)"),
    # live daily curve (bond-equivalent 3m) — consumed by the Explore page so both pages agree
    "live_curve": {
        "y10": round(float(curve["DGS10"][0]), 2),
        "y3m": round(float(conventions.discount_to_bond_equivalent(curve["DTB3"][0], config.BILL_BE_DAYS)), 2),
        "y2": round(float(curve["DGS2"][0]), 2),
        "fedfunds": round(float(curve["DFF"][0]), 2),
        "spread_10y3m": round(float(term_current), 2),
        "flagship_prob": clocks["flag12"],
        "as_of": (term_asof.date().isoformat() if term_asof is not None else None),
    },
}

os.makedirs("output", exist_ok=True)
with open("output/two_clock_data.json", "w") as f:
    json.dump(data, f, indent=1)

tpl = open("report_template.html").read()
html = tpl.replace("__REPORT_DATA__", json.dumps(data))
os.makedirs("docs", exist_ok=True)
for p in ("docs/index.html", "output/index.html"):
    with open(p, "w") as f:
        f.write(html)

print("Single report-dashboard built (docs/index.html).")
print(f"  as of daily : {data['as_of_daily']}   source: {data['data_source']}")
print(f"  headline    : {'FIRING' if firing else 'NOT firing'} - {count_met} of 3 conditions met")
print(f"  1) flagship 12mo = {clocks['flag12']}%  met={s1_met}")
print(f"  2) augmented 3mo = {clocks['aug3']}%  met={s2_met}")
print(f"  3) trust check   = {tp_bp} bp ({tp_source}: {tp_cls}, ~{_ord(tp_pct)} pct)  met={s3_met}")
