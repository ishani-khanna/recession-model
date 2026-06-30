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

from .data import build_dataset, config  # noqa: E402
from .data.fred_client import FredClient  # noqa: E402
from .models import probit  # noqa: E402

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
    """Daily spreads from the live curve (value, as_of_date)."""
    g10, d10 = curve["DGS10"]
    out = {}
    for key, spec in config.SPREADS.items():
        short_v, short_d = curve[spec["short_d"]]
        as_of = max(d for d in (d10, short_d) if d is not None) if (d10 and short_d) else (d10 or short_d)
        out[key] = (round(g10 - short_v, 2), as_of)
    return out


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
    L.append(f"- Primary spread (10Y−3M): **{spread_vals['10y3m']:+.2f} pp** "
             f"(latest monthly {monthly_vals['10y3m']:+.2f}).\n"
             f"- 12-month recession probability (primary): **{primary_p12:.0f}%**.\n"
             f"- {verdict}\n")

    L.append("\n## Live curve\n")
    L.append(_md_table(curve_rows))
    L.append("\n\n## Current spreads\n")
    L.append(_md_table(spread_rows))

    L.append("\n\n## Recession probability — spread × horizon (probit, % )\n")
    L.append("Evaluated at the live daily spread; full-sample probit coefficients.\n")
    L.append(_md_table(prob_daily, index_label="horizon (months)"))

    L.append("\n\n## Charts\n")
    L.append("![Two-panel indicator](indicator_two_panel.png)\n")
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
