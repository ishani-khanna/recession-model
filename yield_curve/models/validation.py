"""Validation (Phase 4) -- the heart of the upgrade.

Three honest tests the prototype never ran:

1. Expanding-window OUT-OF-SAMPLE. At each forecast origin t we refit using only
   (spread_{t'}, y_{t'}) pairs whose recession outcome was already *realized* by t
   (i.e. t' + h <= t), then forecast from spread_t. No hindsight on either the
   feature side or the outcome side. Report OOS AUC and Brier score.

2. The 2022-2023 inversion CASE STUDY: the deepest 10Y-3M inversion since the
   1980s, followed (so far) by NO NBER recession -- the central test of whether
   the signal still works. Term-premium compression is the leading explanation.

3. LEAD-TIME table: months from first sustained inversion to the NBER peak, per
   spread, plus the false-signal count.

    python -m yield_curve.models.validation
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import brier_score_loss, roc_auc_score

from ..data import build_dataset, config
from .target import NBER_CYCLES, build_target

warnings.filterwarnings("ignore")

_MODELS = {"probit": sm.Probit, "logit": sm.Logit}


def _months(a: pd.Timestamp, b: pd.Timestamp) -> int:
    """Whole months from a to b (b - a)."""
    return (b.year - a.year) * 12 + (b.month - a.month)


# --------------------------------------------------------------------------- #
# 1. Expanding-window out-of-sample
# --------------------------------------------------------------------------- #
def expanding_oos(
    panel: pd.DataFrame,
    spread_key: str,
    horizon: int,
    link: str = "probit",
    min_train: int = 120,
    oos_start: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Return (path, metrics).

    path: DataFrame indexed by forecast origin t with columns
        spread_t, prob (OOS forecast of recession at t+h), y (realized USREC_{t+h}
        or NaN if not yet realized).
    metrics: {oos_auc, brier, n_scored, oos_start, ...}
    """
    spread_col = f"spread_{spread_key}"
    spread_all = panel[spread_col].dropna()
    y_full = build_target(panel["usrec"], horizon, "point")  # indexed by origin t'
    aligned = pd.concat({"spread": spread_all, "y": y_full}, axis=1).dropna()

    Model = _MODELS[link]
    floor = pd.Timestamp(oos_start) if oos_start else pd.Timestamp.min
    records = []
    for t in spread_all.index:
        if t < floor:
            continue
        cutoff = t - pd.DateOffset(months=horizon)          # outcomes realized by t
        train = aligned.loc[aligned.index <= cutoff]
        prob = np.nan
        if len(train) >= min_train and train["y"].nunique() == 2:
            X = sm.add_constant(train[["spread"]])
            try:
                res = Model(train["y"], X).fit(disp=0)
                prob = float(res.predict(np.array([[1.0, spread_all.loc[t]]]))[0])
            except Exception:  # noqa: BLE001
                prob = np.nan
        records.append((t, float(spread_all.loc[t]), prob, y_full.get(t, np.nan)))

    path = pd.DataFrame(records, columns=["date", "spread_t", "prob", "y"]).set_index("date")
    scored = path.dropna(subset=["prob", "y"])
    metrics = {
        "spread": spread_key, "horizon": horizon, "link": link,
        "oos_start": scored.index.min().date().isoformat() if len(scored) else None,
        "oos_end": scored.index.max().date().isoformat() if len(scored) else None,
        "n_scored": int(len(scored)),
        "oos_auc": round(float(roc_auc_score(scored["y"], scored["prob"])), 4)
        if scored["y"].nunique() == 2 else np.nan,
        "brier": round(float(brier_score_loss(scored["y"], scored["prob"])), 4)
        if len(scored) else np.nan,
    }
    return path, metrics


def oos_grid(panel: pd.DataFrame, links=("probit",)) -> pd.DataFrame:
    rows = []
    for spread_key in config.SPREADS:
        for h in config.HORIZONS:
            for link in links:
                _, m = expanding_oos(panel, spread_key, h, link)
                rows.append(m)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 2. The 2022-2023 inversion case study
# --------------------------------------------------------------------------- #
def case_study_2022_2023(panel: pd.DataFrame, spread_key: str = "10y3m", horizon: int = 12) -> dict:
    spread_col = f"spread_{spread_key}"
    path, _ = expanding_oos(panel, spread_key, horizon)
    window = slice("2018-01-01", None)
    spread = panel[spread_col].loc[window]

    inv = spread[spread < 0]
    deepest_date = inv.idxmin() if len(inv) else None
    inv_2022 = spread.loc["2022-01-01":]
    inv_period = inv_2022[inv_2022 < 0]

    # OOS probability through the inversion.
    prob_win = path["prob"].loc["2022-01-01":].dropna()
    # NBER recession after the inversion started?
    post = panel["usrec"].loc["2022-07-01":]
    recession_after = bool(post.sum() > 0)

    return {
        "spread": spread_key,
        "horizon": horizon,
        "first_inversion": inv_period.index.min().date().isoformat() if len(inv_period) else None,
        "last_inversion": inv_period.index.max().date().isoformat() if len(inv_period) else None,
        "months_inverted": int(len(inv_period)),
        "deepest_value_pp": round(float(inv.min()), 2) if len(inv) else None,
        "deepest_date": deepest_date.date().isoformat() if deepest_date is not None else None,
        "peak_oos_prob": round(float(prob_win.max()), 4) if len(prob_win) else None,
        "peak_oos_prob_date": prob_win.idxmax().date().isoformat() if len(prob_win) else None,
        "current_spread_pp": round(float(panel[spread_col].dropna().iloc[-1]), 2),
        "current_oos_prob": round(float(path["prob"].dropna().iloc[-1]), 4) if path["prob"].notna().any() else None,
        "nber_recession_since_2022H2": recession_after,
    }


def plot_case_study(panel: pd.DataFrame, spread_key: str = "10y3m", horizon: int = 12) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spread_col = f"spread_{spread_key}"
    path, _ = expanding_oos(panel, spread_key, horizon)
    win = slice("2018-01-01", None)
    spread = panel[spread_col].loc[win]
    prob = path["prob"].loc["2018-01-01":] * 100

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.axhline(0, color="black", lw=0.8)
    ax1.fill_between(spread.index, spread.values, 0, where=(spread.values < 0),
                     color="tab:red", alpha=0.25, label="inverted (spread < 0)")
    ax1.plot(spread.index, spread.values, color="tab:blue", lw=1.6, label=f"{config.SPREADS[spread_key]['label']} spread (pp)")
    ax1.set_ylabel("Term spread (pp)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(prob.index, prob.values, color="tab:orange", lw=1.8,
             label=f"OOS P(recession in {horizon}m) (%)")
    ax2.set_ylabel("OOS recession probability (%)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.set_ylim(0, 100)

    # Shade any NBER recession months in-window (none after 2022 -> the point).
    usrec = panel["usrec"].loc["2018-01-01":]
    for d in usrec[usrec == 1].index:
        ax1.axvspan(d, d + pd.DateOffset(months=1), color="gray", alpha=0.15)

    ax1.set_title(f"2022-2023 inversion: deep {config.SPREADS[spread_key]['label']} inversion, "
                  f"no NBER recession (through {usrec.index.max().date()})")
    lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(lines, labels, loc="upper left", fontsize=8)
    fig.tight_layout()
    out = config.OUTPUTS_DIR / "case_study_2022_2023.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return str(out)


# --------------------------------------------------------------------------- #
# 3. Lead-time table
# --------------------------------------------------------------------------- #
def lead_time_table(panel: pd.DataFrame, lookback: int = 24, persist: int = 3) -> pd.DataFrame:
    """For each spread x NBER peak: lead from first sustained inversion to the peak.

    A 'sustained inversion' uses a 3-month-average spread < 0 (filters one-month
    blips). lead = months from the first such month within `lookback` months before
    the peak, to the peak.
    """
    rows = []
    peaks = [pd.Timestamp(p) for p, _ in NBER_CYCLES]
    for spread_key in config.SPREADS:
        s = panel[f"spread_{spread_key}"].dropna()
        sm3 = s.rolling(3).mean()
        for peak in peaks:
            lo = peak - pd.DateOffset(months=lookback)
            window = sm3.loc[lo:peak]
            inv = window[window < 0]
            if len(window) == 0 or s.index.min() > lo:
                rows.append({"spread": spread_key, "recession_peak": peak.date(),
                             "inverted_before": None, "first_inversion": None, "lead_months": None})
                continue
            rows.append({
                "spread": spread_key,
                "recession_peak": peak.date(),
                "inverted_before": bool(len(inv) > 0),
                "first_inversion": inv.index.min().date() if len(inv) else None,
                "lead_months": _months(inv.index.min(), peak) if len(inv) else None,
            })
    return pd.DataFrame(rows)


def false_signals(panel: pd.DataFrame, persist: int = 3, no_recession_within: int = 24) -> pd.DataFrame:
    """Sustained inversions (3m-avg < 0 for >= `persist` months) NOT followed by an
    NBER recession within `no_recession_within` months -- i.e. false alarms."""
    peaks = [pd.Timestamp(p) for p, _ in NBER_CYCLES]
    usrec = panel["usrec"]
    rows = []
    for spread_key in config.SPREADS:
        s = panel[f"spread_{spread_key}"].dropna()
        sm3 = s.rolling(3).mean()
        neg = (sm3 < 0).astype(int)
        # Identify maximal runs of negative 3m-avg.
        grp = (neg.diff() != 0).cumsum()
        for _, idx in neg[neg == 1].groupby(grp[neg == 1]).groups.items():
            run = pd.DatetimeIndex(idx)
            if len(run) < persist:
                continue
            start = run.min()
            # An inversion that begins *during* a recession is not a leading false signal.
            if usrec.get(start, 0) == 1:
                continue
            followed = any(0 <= _months(start, p) <= no_recession_within for p in peaks)
            rows.append({
                "spread": spread_key,
                "inversion_start": start.date(),
                "inversion_end": run.max().date(),
                "months": int(len(run)),
                "followed_by_recession": followed,
            })
    df = pd.DataFrame(rows)
    return df


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def write_report(panel: pd.DataFrame) -> str:
    oos = oos_grid(panel)
    oos.to_csv(config.OUTPUTS_DIR / "oos_metrics.csv", index=False)
    leads = lead_time_table(panel)
    leads.to_csv(config.OUTPUTS_DIR / "lead_times.csv", index=False)
    fp = false_signals(panel)
    cs = case_study_2022_2023(panel)
    plot_path = plot_case_study(panel)

    def md_table(df):
        df = df.copy()
        cols = list(df.columns)
        head = "| " + " | ".join(str(c) for c in cols) + " |"
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        body = [
            "| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
            for row in df.itertuples(index=False, name=None)
        ]
        return "\n".join([head, sep, *body])

    primary = oos[(oos.spread == "10y3m")]
    lines = []
    lines.append("# Validation report — Yield-Curve Recession Indicator\n")
    lines.append("_Out-of-sample, expanding-window. Each forecast uses only outcomes "
                 "realized by the forecast date (no hindsight)._\n")

    lines.append("## 1. Out-of-sample performance (probit)\n")
    lines.append("Primary spread (10Y−3M), by horizon:\n")
    lines.append(md_table(primary[["horizon", "oos_start", "n_scored", "oos_auc", "brier"]]))
    lines.append("\n\nAll spreads × horizons: see `oos_metrics.csv`.\n")

    lines.append("\n## 2. The 2022–2023 inversion case study\n")
    lines.append(
        f"- Deepest {cs['spread']} inversion: **{cs['deepest_value_pp']} pp** on "
        f"{cs['deepest_date']} (one of the deepest since the early 1980s).\n"
        f"- Inverted from **{cs['first_inversion']}** to **{cs['last_inversion']}** "
        f"(**{cs['months_inverted']} months**).\n"
        f"- Peak OOS 12-month recession probability: **{cs['peak_oos_prob']*100:.0f}%** "
        f"on {cs['peak_oos_prob_date']}.\n"
        f"- NBER recession since 2022H2: **{cs['nber_recession_since_2022H2']}**.\n"
        f"- Current spread **{cs['current_spread_pp']} pp**, current OOS prob "
        f"**{cs['current_oos_prob']*100:.0f}%**.\n"
    )
    lines.append(
        "\n**Interpretation — term-premium compression.** The model, fit on history, "
        "read the deep 2022–2023 inversion as a high recession signal, yet no NBER "
        "recession has followed (through the latest data). The leading explanation is "
        "that the inversion was driven less by a high expected-path-of-policy/long-rate "
        "differential and more by a **compressed (even negative) term premium** — years "
        "of large-scale asset purchases and strong demand for duration pushed the term "
        "premium well below its historical norm (e.g. ACM estimates). When the term "
        "premium is unusually low, a given inversion overstates the market-implied "
        "recession odds relative to the historical relationship the probit was trained "
        "on. A robust read therefore conditions the raw spread signal on the term "
        "premium / real-time-data caveats rather than taking the probit at face value.\n"
    )
    lines.append("\n![2022-2023 case study](case_study_2022_2023.png)\n")

    lines.append("\n## 3. Lead-time table (first sustained inversion → NBER peak)\n")
    lines.append(md_table(leads))
    lines.append("\n\n## 4. False signals (sustained inversions, no recession within 24m)\n")
    lines.append(md_table(fp[~fp.followed_by_recession]))

    report = "\n".join(lines)
    out = config.OUTPUTS_DIR / "validation.md"
    out.write_text(report)
    return str(out)


def main() -> None:
    panel = build_dataset.build_panel(refresh=False)

    print("=== Out-of-sample metrics (probit) ===")
    oos = oos_grid(panel)
    with pd.option_context("display.width", 200):
        print(oos.to_string(index=False))

    print("\n=== 2022-2023 case study ===")
    cs = case_study_2022_2023(panel)
    for k, v in cs.items():
        print(f"  {k}: {v}")

    print("\n=== Lead-time table ===")
    print(lead_time_table(panel).to_string(index=False))

    print("\n=== False signals (no recession within 24m) ===")
    fp = false_signals(panel)
    print(fp[~fp.followed_by_recession].to_string(index=False))

    out = write_report(panel)
    print(f"\nValidation report written to {out}")


if __name__ == "__main__":
    main()
