"""
build_validation_data.py  -  export data for the Validation & methodology page (docs/validation.html).

Regenerates outputs/validation.md live from the engine (validation.write_report) with the Signal-2
credit fallback wired in, parses its eight sections into structured blocks (paragraphs / data tables /
lists), and computes the two report charts not yet on the site — the term-spread comparison and the
horizon-sensitivity (current probability + in-sample AUC vs horizon). Injects everything into
validation_template.html -> docs/validation.html. VISUALIZATION only: every number comes from the
yield_curve engine; nothing is re-derived here.
"""

import datetime as dt
import html
import json
import os
import re
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

import credit_fallback  # noqa: F401  WIRING: Signal-2 Baa−Aa -> FRED Aaa substitute when DataBuffet off
from yield_curve.data import build_dataset, config
from yield_curve.models import probit, validation
from yield_curve import reporting

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
# Minimal, safe markdown -> HTML for the inline bits we actually use.
# --------------------------------------------------------------------------- #
def _inline(md: str) -> str:
    """Escape HTML, then render `code`, **bold**, *italic* / _italic_ (code spans protected)."""
    codes = []

    def _stash(m):
        codes.append(html.escape(m.group(1)))
        return f"\x00{len(codes) - 1}\x00"

    md = re.sub(r"`([^`]+)`", _stash, md)
    md = html.escape(md)
    md = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", md)
    md = re.sub(r"(?<!\w)\*([^*]+)\*(?!\w)", r"<em>\1</em>", md)
    md = re.sub(r"(?<!\w)_([^_]+)_(?!\w)", r"<em>\1</em>", md)
    md = re.sub(r"\x00(\d+)\x00", lambda m: f"<code>{codes[int(m.group(1))]}</code>", md)
    return md


def _is_table_sep(cells) -> bool:
    return all(re.fullmatch(r":?-{2,}:?", c.strip() or "") for c in cells) and len(cells) > 0


def _split_row(line: str):
    parts = [c.strip() for c in line.strip().strip("|").split("|")]
    return parts


def parse_markdown(md_text: str) -> list:
    """Parse the validation report into [{title, blocks:[...]}] (## sections only)."""
    lines = md_text.splitlines()
    sections, cur = [], None
    para, table, lst = [], [], []

    def flush_para():
        nonlocal para
        if para:
            cur["blocks"].append({"type": "p", "html": _inline(" ".join(para).strip())})
            para = []

    def flush_table():
        nonlocal table
        if table:
            head = _split_row(table[0])
            body = [_split_row(r) for r in table[1:] if not _is_table_sep(_split_row(r))]
            cur["blocks"].append({
                "type": "table",
                "head": [_inline(c) for c in head],
                "rows": [[_inline(c) for c in r] for r in body],
            })
            table = []

    def flush_list():
        nonlocal lst
        if lst:
            cur["blocks"].append({"type": "list", "items": [_inline(x) for x in lst]})
            lst = []

    def flush_all():
        flush_para(); flush_table(); flush_list()

    for raw in lines:
        line = raw.rstrip()
        if line.startswith("## "):
            if cur:
                flush_all(); sections.append(cur)
            cur = {"title": line[3:].strip(), "blocks": []}
            continue
        if cur is None:
            continue  # skip the H1 / preamble before section 1
        if line.startswith("!["):            # drop matplotlib image refs
            flush_all(); continue
        if line.startswith("|"):
            flush_para(); flush_list(); table.append(line); continue
        if line.lstrip().startswith("- "):
            flush_para(); flush_table(); lst.append(line.lstrip()[2:]); continue
        if not line.strip():
            flush_all(); continue
        # ordinary prose line
        flush_table(); flush_list(); para.append(line.strip())
    if cur:
        flush_all(); sections.append(cur)
    return sections


# --------------------------------------------------------------------------- #
# Regenerate validation.md live (credit fallback active) and parse it.
# --------------------------------------------------------------------------- #
panel = build_dataset.build_panel(refresh=False)
md_path = validation.write_report(panel)
with open(md_path) as f:
    md_text = f.read()
sections = parse_markdown(md_text)

# --- Section 7: replace the wide two-column table with a stack of concise cards. ---
# Trimmed, one-line reasons (the full analysis stays in the module docstrings / RESEARCH.md).
SECTION7_CANDIDATES = [
    ("Term premium as a regressor", None,
     "Works only as the veto; as a predictor its gain flips sign between samples."),
    ("Alternative credit spreads", "Baa–10Y, Baa–Aaa, GZ EBP",
     "Baa–Aa already beats them; the rest add no out-of-sample gain."),
    ("Fed balance sheet", "growth, /GDP, change",
     "Reactive, not leading; the Fed expands into weakness."),
    ("Real house-price growth", "FHFA",
     "Flatters the overall score but breaks the 2001 call."),
    ("Corporate credit growth / leverage", "Schularick–Taylor",
     "Redundant with Baa–Aa; leverage levels are non-stationary artifacts."),
    ("SLOOS C&I lending standards", None,
     "Nailed 2001 but blind to 2007; drops the 2007 call to 0%."),
    ("SLOOS consumer-loan willingness", None,
     "Higher near-term AUC, but guts the term-spread coefficient; the early edge was a small-sample artifact."),
    ("Philly Fed current general activity", None,
     "Not a predictor; ~3× the false alarms, signal and noise are the same swings."),
    ("Near-term forward spread", "Engstrom–Sharpe",
     "A quieter edge over 10Y–3M, not a sharper one; adds nothing once the veto exists."),
    ("Building permits", "Leamer",
     "Leads housing recessions only; as a 4th condition it turns 7/8 into 6/8."),
    ("5-year change in the 5Y yield", None,
     "Rate-shock specialist (1980–81 only); 4–12× the false alarms, flashing since 2023."),
    ("High-yield OAS", None,
     "Untestable; FRED's ICE series is truncated."),
]
_cards_block = {"type": "cards", "items": [
    {"name": html.escape(n), "source": (html.escape(s) if s else None), "reason": html.escape(r)}
    for (n, s, r) in SECTION7_CANDIDATES]}
for _sec in sections:
    if _sec["title"].startswith("7."):
        _sec["blocks"] = [b for b in _sec["blocks"] if b["type"] != "table"]  # drop the wide table
        _at = 1 if _sec["blocks"] and _sec["blocks"][0]["type"] == "p" else 0  # keep intro on top
        _sec["blocks"].insert(_at, _cards_block)

# Pull the headline numbers straight out of the parsed section-1 OOS table (h = 12 row).
coef = round(float(probit.fit_spread_model(panel, "10y3m", 12, "probit").coef), 3)
in_auc = round(float(probit.fit_spread_model(panel, "10y3m", 12, "probit").auc), 3)
oos_auc = brier = None
for sec in sections:
    if sec["title"].startswith("1."):
        for b in sec["blocks"]:
            if b["type"] == "table" and "horizon" in b["head"][0].lower():
                for r in b["rows"]:
                    if r[0] == "12":
                        oos_auc, brier = float(r[3]), float(r[4])

# --------------------------------------------------------------------------- #
# Chart 1 — term-spread comparison (the three spreads over time, NBER shaded).
# --------------------------------------------------------------------------- #
idx = panel.index
spread_series = {k: [None if pd.isna(v) else round(float(v), 2) for v in panel[f"spread_{k}"]]
                 for k in config.SPREADS}


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


spread_compare = {
    "dates": [d.strftime("%Y-%m") for d in idx],
    "labels": {k: config.SPREADS[k]["label"] for k in config.SPREADS},
    "series": spread_series,
    "recessions": _rec_bands(panel["usrec"]),
}

# --------------------------------------------------------------------------- #
# Chart 2 — horizon sensitivity (current probability + in-sample AUC vs horizon).
# --------------------------------------------------------------------------- #
curve = reporting.fetch_current_curve()
dspreads = reporting.current_spreads_daily(curve)
spread_now = {k: float(v) for k, (v, _) in dspreads.items()}
horizons = list(config.HORIZONS)
probs, aucs = {}, {}
for k in config.SPREADS:
    pr, au = [], []
    for h in horizons:
        fr = probit.fit_spread_model(panel, k, h, "probit")
        pr.append(round(float(norm.cdf(fr.intercept + fr.coef * spread_now[k]) * 100), 1))
        au.append(round(float(fr.auc), 3))
    probs[k], aucs[k] = pr, au
horizon = {"horizons": horizons, "labels": {k: config.SPREADS[k]["label"] for k in config.SPREADS},
           "probs": probs, "aucs": aucs}

data = {
    "generated": dt.date.today().isoformat(),
    "as_of_daily": (dspreads["10y3m"][1].date().isoformat() if dspreads["10y3m"][1] is not None else None),
    "monthly_through": panel.index.max().strftime("%Y-%m"),
    "headline": {"coef": coef, "in_auc": in_auc, "oos_auc": oos_auc, "brier": brier},
    "sections": sections,
    "spread_compare": spread_compare,
    "horizon": horizon,
}

os.makedirs("output", exist_ok=True)
with open("output/validation_data.json", "w") as f:
    json.dump(data, f, indent=1)

tpl = open("validation_template.html").read()
out_html = tpl.replace("__VALIDATION_DATA__", json.dumps(data))
os.makedirs("docs", exist_ok=True)
for p in ("docs/validation.html", "output/validation.html"):
    with open(p, "w") as fh:
        fh.write(out_html)

print("Validation page built (docs/validation.html).")
print(f"  sections parsed : {len(sections)}  -> {[s['title'].split('.')[0] for s in sections]}")
print(f"  headline        : coef {coef}  in-AUC {in_auc}  OOS-AUC {oos_auc}  Brier {brier}")
print(f"  as of daily     : {data['as_of_daily']}   monthly through {data['monthly_through']}")
