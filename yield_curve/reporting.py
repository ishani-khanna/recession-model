"""Current reading, chart suite, and report.md (Phase 5).

Pulls the latest *daily* curve for a live reading, computes current spreads and
recession probabilities across spreads x horizons (using the full-sample probit
fits), regenerates the classic two-panel indicator chart through the present plus
a spread-comparison panel and a horizon-sensitivity panel, and writes
``outputs/report.md``.
"""

from __future__ import annotations

import datetime as dt
import warnings

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import norm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .data import build_dataset, config, conventions  # noqa: E402
from .data.fred_client import FredClient  # noqa: E402
from .models import probit  # noqa: E402
from .models.credit_spread import fit_insample as _fit_augmented, load_baa_aa  # noqa: E402

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
# Live daily curve
# --------------------------------------------------------------------------- #
def fetch_current_curve() -> dict:
    """Latest valid daily value for each curve series. Returns {id: (value, date)}."""
    client = FredClient(pause=0.1)
    start = (dt.date.today() - dt.timedelta(days=400)).isoformat()
    out = {}
    for sid in config.FRED_DAILY:
        s = client.get_series(sid, observation_start=start).dropna()
        if len(s):
            out[sid] = (float(s.iloc[-1]), s.index[-1])
        else:
            out[sid] = (np.nan, None)
    return out


def current_spreads_daily(curve: dict) -> dict:
    """Daily spreads from the live curve (value, as_of_date).

    The 3-month bill (DTB3) is converted discount→bond-equivalent before differencing
    so the 10Y-3M spread is on a consistent basis with the 10Y leg.
    """
    g10, d10 = curve["DGS10"]
    out = {}
    for key, spec in config.SPREADS.items():
        short_v, short_d = curve[spec["short_d"]]
        if spec["short_d"] in config.DISCOUNT_BASIS_BILLS:
            short_v = conventions.discount_to_bond_equivalent(short_v, config.BILL_BE_DAYS)
        as_of = max(d for d in (d10, short_d) if d is not None) if (d10 and short_d) else (d10 or short_d)
        out[key] = (round(g10 - short_v, 2), as_of)
    return out


# --------------------------------------------------------------------------- #
# Term-premium diagnostic (context only -- NOT used in the model)
# --------------------------------------------------------------------------- #
def term_premium_diagnostic(spread_10y3m_current: float) -> dict:
    """Decompose the current 10Y-3M slope into an expectations component and the
    term premium, and flag whether the term premium is distorting the signal.

    Diagnostic only: an inversion driven by the expectations component (expected
    rate cuts) is recession-predictive; one driven by a compressed term premium
    (as in 2022-23) is not. Uses the Kim-Wright 10Y term premium (FRED THREEFYTP10).
    """
    tp = FredClient(pause=0.1).get_series("THREEFYTP10").dropna()  # daily, 1990+
    tp_cur, tp_date = float(tp.iloc[-1]), tp.index[-1]
    tp_m = tp.resample("MS").mean()
    pct = round(float((tp_m < tp_cur).mean()) * 100)
    compressed_era = float(tp_m.loc["2020-06":"2021-12"].mean())
    cls = "compressed" if (tp_cur < 0 or pct < 25) else ("elevated" if pct > 75 else "near normal")
    return {
        "tp_current": round(tp_cur, 2),
        "tp_date": tp_date.date().isoformat(),
        "tp_percentile": pct,
        "tp_class": cls,
        "tp_compressed_2020_21": round(compressed_era, 2),
        "spread": round(spread_10y3m_current, 2),
        "exp_component": round(spread_10y3m_current - tp_cur, 2),
    }


# --------------------------------------------------------------------------- #
# Two-clock dashboard: flagship 12-month + augmented (term + Baa-Aa) 3-month
# --------------------------------------------------------------------------- #
def two_clock_dashboard(panel: pd.DataFrame, term_current: float) -> dict:
    """Two complementary reads: the flagship 12-month outlook (term spread only) and
    the augmented 3-month watch (term + Baa-Aa). The credit spread sharpens the near-
    term read; it is NOT used at 12 months (where it degrades the term spread)."""
    df = panel.copy()
    df["baa_aa"] = load_baa_aa()
    credit = df["baa_aa"].dropna()
    credit_now, credit_date = float(credit.iloc[-1]), credit.index[-1]

    fr12 = probit.fit_spread_model(df, "10y3m", 12, "probit")
    flag12 = float(norm.cdf(fr12.intercept + fr12.coef * term_current)) * 100

    res3, _, _ = _fit_augmented(df, ["spread_10y3m", "baa_aa"], 3)
    aug3 = float(norm.cdf(res3.params["const"] + res3.params["spread_10y3m"] * term_current
                          + res3.params["baa_aa"] * credit_now)) * 100
    return {"flag12": round(flag12), "aug3": round(aug3),
            "term": round(term_current, 2), "credit": round(credit_now, 2),
            "credit_date": credit_date.date().isoformat()}


# --------------------------------------------------------------------------- #
# Current probabilities (full-sample probit, evaluated at the live spread)
# --------------------------------------------------------------------------- #
def probability_table(panel: pd.DataFrame, spread_values: dict, link: str = "probit") -> pd.DataFrame:
    """Rows = horizon, columns = spread; cells = P(recession in t+h) at the live spread."""
    data = {}
    for key in config.SPREADS:
        x = spread_values[key]
        col = {}
        for h in config.HORIZONS:
            fr = probit.fit_spread_model(panel, key, h, link)
            lin = fr.intercept + fr.coef * x
            col[h] = norm.cdf(lin) if link == "probit" else 1.0 / (1.0 + np.exp(-lin))
        data[key] = col
    df = pd.DataFrame(data)
    df.index.name = "horizon"
    return (df * 100).round(1)  # percent


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def _shade_recessions(ax, usrec: pd.Series) -> None:
    u = usrec.dropna().astype(int)
    inrec = False
    start = None
    for d, v in u.items():
        if v == 1 and not inrec:
            inrec, start = True, d
        elif v == 0 and inrec:
            ax.axvspan(start, d, color="gray", alpha=0.18, lw=0)
            inrec = False
    if inrec:
        ax.axvspan(start, u.index[-1], color="gray", alpha=0.18, lw=0)


def plot_two_panel(panel: pd.DataFrame, spread_key: str = "10y3m", horizon: int = 12) -> str:
    fr = probit.fit_spread_model(panel, spread_key, horizon, "probit")
    spread = panel[f"spread_{spread_key}"].dropna()
    probs = fr.fitted_probs * 100
    label = config.SPREADS[spread_key]["label"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    _shade_recessions(ax1, panel["usrec"])
    ax1.axhline(0, color="black", lw=0.8)
    ax1.fill_between(spread.index, spread.values, 0, where=(spread.values < 0),
                     color="tab:red", alpha=0.3)
    ax1.plot(spread.index, spread.values, color="tab:blue", lw=1.0)
    ax1.set_ylabel(f"{label} spread (pp)")
    ax1.set_title(f"Yield-curve recession indicator — {label}  (gray = NBER recessions)")

    _shade_recessions(ax2, panel["usrec"])
    ax2.plot(probs.index, probs.values, color="tab:orange", lw=1.0)
    ax2.set_ylabel(f"P(recession in {horizon}m) (%)")
    ax2.set_ylim(0, 100)
    ax2.axhline(50, color="gray", ls=":", lw=0.8)
    fig.tight_layout()
    out = config.OUTPUTS_DIR / "indicator_two_panel.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return str(out)


def plot_flagship_probability(panel: pd.DataFrame, horizon: int = 12) -> str:
    """Flagship model's recession probability over full history, with NBER recession bars."""
    fr = probit.fit_spread_model(panel, "10y3m", horizon, "probit")
    # Predict at every monthly spread through the present (the last ~h months are the
    # forward-looking current estimate, since their outcome isn't realized yet).
    spread = panel["spread_10y3m"].dropna()
    probs = pd.Series(norm.cdf(fr.intercept + fr.coef * spread.values) * 100, index=spread.index)
    fig, ax = plt.subplots(figsize=(12, 5))
    _shade_recessions(ax, panel["usrec"])
    ax.plot(probs.index, probs.values, color="tab:red", lw=1.2)
    ax.fill_between(probs.index, probs.values, 0, color="tab:red", alpha=0.12)
    ax.axhline(50, color="gray", ls=":", lw=0.9)
    last, last_d = probs.iloc[-1], probs.index[-1]
    ax.annotate(f"latest: {last:.0f}%", xy=(last_d, last),
                xytext=(-78, 30), textcoords="offset points", fontsize=9,
                arrowprops=dict(arrowstyle="->", color="gray"))
    ax.set_ylim(0, 100)
    ax.set_ylabel(f"P(recession in {horizon}m) (%)")
    ax.set_title(f"Flagship model — {horizon}-month recession probability from the 10Y−3M term "
                 f"spread\n(gray bars = NBER recessions; full-sample probit fit)")
    ax.margins(x=0.01)
    fig.tight_layout()
    out = config.OUTPUTS_DIR / "flagship_recession_probability.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return str(out)


def plot_spread_comparison(panel: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(11, 5))
    _shade_recessions(ax, panel["usrec"])
    ax.axhline(0, color="black", lw=0.8)
    colors = {"10y3m": "tab:blue", "10y2y": "tab:green", "10yffr": "tab:purple"}
    for key, spec in config.SPREADS.items():
        s = panel[f"spread_{key}"].dropna()
        ax.plot(s.index, s.values, lw=1.0, color=colors[key], label=spec["label"])
    ax.set_ylabel("Term spread (pp)")
    ax.set_title("Term-spread comparison (gray = NBER recessions)")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    out = config.OUTPUTS_DIR / "spread_comparison.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return str(out)


def plot_horizon_sensitivity(panel: pd.DataFrame, spread_values: dict) -> str:
    """Left: current probability vs horizon. Right: in-sample AUC vs horizon."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = {"10y3m": "tab:blue", "10y2y": "tab:green", "10yffr": "tab:purple"}
    for key, spec in config.SPREADS.items():
        probs, aucs = [], []
        for h in config.HORIZONS:
            fr = probit.fit_spread_model(panel, key, h, "probit")
            lin = fr.intercept + fr.coef * spread_values[key]
            probs.append(norm.cdf(lin) * 100)
            aucs.append(fr.auc)
        axL.plot(config.HORIZONS, probs, "o-", color=colors[key], label=spec["label"])
        axR.plot(config.HORIZONS, aucs, "o-", color=colors[key], label=spec["label"])
    axL.set_title("Current recession probability vs horizon")
    axL.set_xlabel("horizon (months)"); axL.set_ylabel("probability (%)")
    axL.set_ylim(0, max(20, axL.get_ylim()[1])); axL.legend(fontsize=8)
    axR.set_title("In-sample AUC vs horizon")
    axR.set_xlabel("horizon (months)"); axR.set_ylabel("AUC")
    axR.axhline(0.5, color="gray", ls=":", lw=0.8); axR.legend(fontsize=8)
    fig.tight_layout()
    out = config.OUTPUTS_DIR / "horizon_sensitivity.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return str(out)


# --------------------------------------------------------------------------- #
# report.md
# --------------------------------------------------------------------------- #
def _md_table(df: pd.DataFrame, index_label: str | None = None) -> str:
    df = df.copy()
    if index_label is not None:
        df = df.reset_index().rename(columns={df.index.name or "index": index_label})
    cols = [str(c) for c in df.columns]
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
            for row in df.itertuples(index=False, name=None)]
    return "\n".join([head, sep, *body])


def write_report(panel: pd.DataFrame) -> str:
    curve = fetch_current_curve()
    dspreads = current_spreads_daily(curve)
    spread_vals = {k: v for k, (v, _) in dspreads.items()}
    monthly_vals = {k: round(float(panel[f"spread_{k}"].dropna().iloc[-1]), 2) for k in config.SPREADS}

    prob_daily = probability_table(panel, spread_vals)
    asof_daily = max((d for _, d in curve.values() if d is not None))
    asof_monthly = panel.index.max()

    p0 = plot_flagship_probability(panel)
    p1 = plot_two_panel(panel)
    p2 = plot_spread_comparison(panel)
    p3 = plot_horizon_sensitivity(panel, spread_vals)

    # Curve table
    curve_rows = pd.DataFrame(
        [{"series": s, "desc": config.FRED_DAILY[s], "value (%)": v,
          "as of": d.date().isoformat() if d is not None else "n/a"}
         for s, (v, d) in curve.items()]
    )
    # Spread table (daily vs latest monthly)
    spread_rows = pd.DataFrame(
        [{"spread": config.SPREADS[k]["label"], "daily (pp)": spread_vals[k],
          "latest monthly (pp)": monthly_vals[k],
          "inverted?": "yes" if spread_vals[k] < 0 else "no"}
         for k in config.SPREADS]
    )

    primary_p12 = prob_daily.loc[config.DEFAULT_HORIZON, config.PRIMARY_SPREAD]

    L = []
    L.append("# Yield-Curve Recession Tracker — Current Reading\n")
    L.append(f"_Generated {dt.date.today().isoformat()}. Daily curve as of "
             f"**{asof_daily.date().isoformat()}**; monthly panel through "
             f"**{asof_monthly.date().isoformat()}**._\n")

    L.append("## Headline\n")
    verdict = ("The curve is **positively sloped** and the model reads recession risk "
               "as **modest** — it does **not** signal an imminent recession.") if primary_p12 < 30 else (
               "The model reads **elevated** recession risk — interpret with the "
               "term-premium caveats below.")
    L.append(f"- Primary spread (10Y−3M, bond-equivalent): **{spread_vals['10y3m']:+.2f} pp** "
             f"(latest monthly {monthly_vals['10y3m']:+.2f}).\n"
             f"- 12-month recession probability (primary): **{primary_p12:.0f}%**.\n"
             f"- {verdict}\n")

    # Two-clock dashboard (graceful fallback if the Baa-Aa / DataBuffet pull fails).
    try:
        tc = two_clock_dashboard(panel, spread_vals["10y3m"])
        L.append("\n## Two-clock dashboard\n")
        L.append("Two complementary reads, each from the model best at its horizon "
                 "(the augmented model adds the Baa−Aa credit spread, which sharpens the "
                 "near-term read but is _not_ used at 12 months):\n")
        L.append("| Clock | Model | Question | Reading |")
        L.append("| --- | --- | --- | --- |")
        L.append(f"| **12-month outlook** | term spread (flagship) | Is a recession building "
                 f"over the next year? | **{tc['flag12']}%** |")
        L.append(f"| **3-month watch** | term + Baa−Aa (augmented) | Is one arriving in the "
                 f"next quarter? | **{tc['aug3']}%** |")
        credit_word = "calm" if tc["credit"] < 0.7 else ("elevated" if tc["credit"] > 1.3 else "moderate")
        L.append(f"\nInputs: 10Y−3M term spread {tc['term']:+.2f} (bond-equivalent); Baa−Aa "
                 f"credit spread {tc['credit']:+.2f} ({credit_word}, as of {tc['credit_date']}).")
        both_low = tc["flag12"] < 30 and tc["aug3"] < 30
        L.append("\n\n_" + (
            "Both clocks are modest — the curve sees no recession building over the year, and "
            f"with credit markets {credit_word}, the near-term watch is low."
            if both_low else
            "At least one clock is elevated — see the per-horizon table and caveats below."
        ) + "_")

    except Exception as e:  # noqa: BLE001 - dashboard must never break the report
        L.append(f"\n## Two-clock dashboard\n\n_Augmented 3-month clock unavailable this run "
                 f"({e}); the 12-month flagship reading is in the headline above._")

    L.append("\n## Live curve\n")
    L.append(_md_table(curve_rows))
    L.append("\n\n## Current spreads\n")
    L.append(_md_table(spread_rows))
    L.append("\n\n_The 3-month bill (DTB3/TB3MS, quoted discount-basis) is converted to a "
             "coupon-equivalent (bond-equivalent) yield before differencing, matching the 10Y "
             "constant-maturity leg and the NY Fed convention. The 'Live curve' DTB3 above is the "
             "raw discount quote._")

    L.append("\n\n## Recession probability — spread × horizon (probit, % )\n")
    L.append("Evaluated at the live daily spread; full-sample probit coefficients.\n")
    L.append(_md_table(prob_daily, index_label="horizon (months)"))

    # Term-premium diagnostic (context only; the model uses the raw spread).
    try:
        tpd = term_premium_diagnostic(spread_vals["10y3m"])
        L.append("\n\n## Term-premium diagnostic — context, _not_ used in the model\n")
        L.append(
            "The 10Y−3M slope splits into an **expectations component** (expected average "
            "short rate vs today's) and a **term premium**. An inversion driven by the "
            "expectations component is recession-predictive; one driven by a *compressed* "
            "term premium (as in 2022–23) is not. Conditioning the model on the term premium "
            "tempers the 2022–23 false alarm but does **not** robustly improve out-of-sample "
            "accuracy — the OOS AUC gain from adding it flips sign between the long ACM (1961+) "
            "and shorter Kim–Wright (1990+) samples (10Y−3M: 0.79→0.80 on ACM vs 0.60→0.46 on "
            "Kim–Wright) — so it is reported here as a flag only.\n")
        L.append(
            f"\n- **10Y term premium (Kim–Wright):** {tpd['tp_current']:+.2f} pp "
            f"(as of {tpd['tp_date']}) — ~{tpd['tp_percentile']}th percentile since 1990, "
            f"**{tpd['tp_class']}** (vs ~{tpd['tp_compressed_2020_21']:+.2f} in the compressed "
            f"2020–21 era).\n"
            f"- **Decomposition (10Y term premium only):** of the {tpd['spread']:+.2f} pp slope, "
            f"{tpd['tp_current']:+.2f} pp is 10Y term premium, leaving a {tpd['exp_component']:+.2f} pp "
            f"expectations residual (= slope − 10Y term premium; the residual also nets the small "
            f"3M-bill term premium).\n")
        if tpd["tp_class"] == "compressed":
            L.append("- **Flag:** the term premium is compressed, so the raw curve signal may "
                     "**overstate** recession risk (the 2022–23 failure mode). Weight the "
                     "expectations component accordingly.\n")
        else:
            L.append("- **Flag:** on this (single, Kim–Wright) estimate the term premium is not "
                     "compressed the way it was in 2022–23, so the raw curve signal is **less likely "
                     "to be distorted** than it was then. The expectations residual is "
                     + ("roughly flat" if abs(tpd["exp_component"]) < 0.25 else
                        ("clearly positive" if tpd["exp_component"] > 0 else "mildly negative"))
                     + " — directional context, not a validated adjustment.\n")

    except Exception as e:  # noqa: BLE001 - diagnostic must never break the report
        L.append(f"\n\n## Term-premium diagnostic\n\n_Unavailable this run ({e})._\n")

    L.append("\n\n## Charts\n")
    L.append("![Flagship recession probability](flagship_recession_probability.png)\n")
    L.append("\n![Two-panel indicator](indicator_two_panel.png)\n")
    L.append("\n![Spread comparison](spread_comparison.png)\n")
    L.append("\n![Horizon sensitivity](horizon_sensitivity.png)\n")

    L.append("\n## Interpretation caveats\n")
    L.append(
        "- **Term-premium compression.** The deep 2022–2023 inversion drove the "
        "model's probability to ~77% with no NBER recession (so far). A compressed/"
        "negative term premium can make a given inversion overstate recession odds; "
        "read the raw signal conditioned on the term premium. See `validation.md`.\n"
        "- **2020 was exogenous.** The COVID recession was not curve-predicted; an "
        "exclude-2020 estimation variant is available and barely changes the fit.\n"
        "- **Monthly vs daily.** Coefficients are estimated on monthly-average spreads; "
        "the live reading plugs in the latest daily spread (typically within a few bp).\n"
        "- **2Y history is short** (from 1976), so 10Y−2Y has fewer observations than "
        "the 10Y−3M and 10Y−FFR spreads.\n"
    )

    report = "\n".join(L)
    out = config.OUTPUTS_DIR / "report.md"
    out.write_text(report)
    return str(out)


_HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Yield Curve Recession Tracker</title>
<style>
  @page { size: letter; margin: 0.7in; }
  body { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
         color: #1a1a1a; line-height: 1.5; max-width: 7.1in; margin: 0 auto; font-size: 11pt; }
  h1 { font-size: 20pt; margin: 0 0 2px; color: #0b3d66; }
  h2 { font-size: 14pt; margin: 22px 0 8px; color: #0b3d66;
       border-bottom: 2px solid #e3e8ee; padding-bottom: 3px; }
  table { border-collapse: collapse; margin: 10px 0; width: 100%; font-size: 10pt; }
  th, td { border: 1px solid #ccd4dd; padding: 5px 9px; text-align: left; }
  th { background: #f0f4f8; font-weight: 600; }
  img { max-width: 100%; height: auto; display: block; margin: 14px auto;
        border: 1px solid #e3e8ee; border-radius: 4px; }
  code, pre { background: #f5f7fa; border-radius: 3px; font-size: 9.5pt; }
  pre { padding: 8px 10px; overflow-x: auto; }
  em { color: #555; }
  h2, table, img { break-inside: avoid; }
  .meta { color: #777; font-size: 9.5pt; margin-bottom: 4px; }
  .page-break { break-before: page; }
</style></head><body>
__BODY__
</body></html>
"""


def build_html_report(html_out=None, include_validation: bool = True) -> str:
    """Render outputs/report.md (and, if present, validation.md) to a single
    self-contained HTML file -- all charts base64-embedded, so it stands alone.
    Convert to PDF with make_pdf.sh."""
    import base64
    import re

    import markdown

    exts = ["tables", "fenced_code", "sane_lists"]
    parts = [markdown.markdown((config.OUTPUTS_DIR / "report.md").read_text(), extensions=exts)]

    val_md = config.OUTPUTS_DIR / "validation.md"
    if include_validation and val_md.exists():
        parts.append('<div class="page-break"></div>')
        parts.append(markdown.markdown(val_md.read_text(), extensions=exts))

    def _embed(m):
        p = config.OUTPUTS_DIR / m.group(1)
        if p.exists():
            b64 = base64.b64encode(p.read_bytes()).decode()
            return f'src="data:image/png;base64,{b64}"'
        return m.group(0)

    body = re.sub(r'src="([^"]+\.png)"', _embed, "\n".join(parts))
    out = config.OUTPUTS_DIR / "report.html" if html_out is None else html_out
    out.write_text(_HTML_TEMPLATE.replace("__BODY__", body))
    return str(out)


def main() -> None:
    panel = build_dataset.build_panel(refresh=False)
    out = write_report(panel)

    curve = fetch_current_curve()
    dspreads = current_spreads_daily(curve)
    spread_vals = {k: v for k, (v, _) in dspreads.items()}
    print("=== Live curve ===")
    for s, (v, d) in curve.items():
        print(f"  {s:<8} {v:6.2f}%  as of {d.date() if d is not None else 'n/a'}")
    print("\n=== Current spreads (daily) ===")
    for k, (v, d) in dspreads.items():
        print(f"  {config.SPREADS[k]['label']:<16} {v:+.2f} pp  ({'inverted' if v < 0 else 'positive'})")
    print("\n=== Recession probability table (spread × horizon, %) ===")
    print(probability_table(panel, spread_vals).to_string())
    print(f"\nReport + charts written; report at {out}")


if __name__ == "__main__":
    main()
