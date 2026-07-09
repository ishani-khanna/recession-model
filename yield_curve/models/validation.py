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
FLAGSHIP_SIGNAL_THR = 0.30   # episode-optimal: catches all recessions, fewest false alarms


def recession_signal_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Per period (recessions AND the flagship's genuine false-alarm episodes): peak OOS
    probability from the flagship (term spread, 12m) and augmented (term + Baa-Aa, 3m)
    models, the ACM term premium + flag over the window, and whether the flagship cleared
    the episode-optimal ~30% signal threshold (✓)."""
    from . import credit_spread, term_premium  # lazy: pulls Baa-Aa (DataBuffet) + ACM (Fed)

    df = panel.copy()
    df["baa_aa"] = credit_spread.load_baa_aa()
    tp = term_premium.load_term_premium("acm")
    tp_pctl = tp.rank(pct=True) * 100
    flag, _, _ = credit_spread.expanding_oos(df[["spread_10y3m", "usrec"]].dropna(), ["spread_10y3m"], 12)
    aug, _, _ = credit_spread.expanding_oos(
        df[["spread_10y3m", "baa_aa", "usrec"]].dropna(), ["spread_10y3m", "baa_aa"], 3)
    pf, pa = flag["prob"].dropna(), aug["prob"].dropna()
    peaks = [pd.Timestamp(p) for p, _ in NBER_CYCLES]
    usrec = panel["usrec"]

    def fpk(lo, hi):
        v = pf.loc[lo:hi].max() * 100
        return f"{v:.0f}%" + (" ✓" if v >= FLAGSHIP_SIGNAL_THR * 100 else ""), v
    def apk(lo, hi):
        w = pa.loc[lo:hi]
        v = w.max() * 100 if len(w) else float("nan")
        return (f"{v:.0f}%" if len(w) else "—"), v
    def tp_read(lo, hi):
        w = tp.loc[lo:hi].dropna()
        if not len(w):
            return "—", "—", float("nan")
        lvl, pc = w.mean(), tp_pctl.loc[w.index].mean()
        flg = "compressed" if (lvl < 0 or pc < 25) else ("elevated" if pc > 75 else "normal")
        return f"{lvl:+.2f} ({pc:.0f}%ile)", flg, lvl * 100  # bp

    def row(d, period, outcome, lo, hi, tp_lo):
        fs, fv = fpk(lo, hi)
        as_, av = apk(lo, hi)
        ts, flg, tbp = tp_read(tp_lo, hi)
        return {"_d": d, "period": period, "outcome": outcome, "flagship 12m": fs,
                "augmented 3m": as_, "ACM term premium": ts, "TP diagnostic": flg,
                "_flag": fv, "_aug": av, "_tp_bp": tbp}

    rows = []
    # Recessions (peak OOS in the 24 months before the NBER peak)
    for pk, _ in NBER_CYCLES:
        p = pd.Timestamp(pk)
        if p.year < 1962:  # before ACM term premium / OOS training window
            continue
        lo = p - pd.DateOffset(months=24)
        rows.append(row(p, pk, "recession", lo, p, p - pd.DateOffset(months=18)))

    # Flagship false-alarm episodes: sustained signal (>15%), not starting in a recession,
    # with NO NBER recession within the episode or 24 months after it ends.
    sig = (pf > 0.15).astype(int)
    grp = (sig.diff() != 0).cumsum()
    for _, idx in sig[sig == 1].groupby(grp[sig == 1]).groups.items():
        run = pd.DatetimeIndex(idx)
        s, e = run.min(), run.max()
        if len(run) < 3 or usrec.get(s, 0) == 1:
            continue
        if any((s <= pk <= e) or (0 <= (pk.year - e.year) * 12 + (pk.month - e.month) <= 24) for pk in peaks):
            continue
        label = f"{s.year}" if s.year == e.year else f"{s.year}–{e.year}"
        rows.append(row(s, f"{label} (no recession)", "false alarm", s, e, s))

    return pd.DataFrame(rows).sort_values("_d").drop(columns="_d").reset_index(drop=True)


# Three-signal rule thresholds (experimental; tuned in-sample -- see caveats in the report).
THREE_SIGNAL = {"flagship": 30.0, "augmented": 15.0, "tp_bp": 50.0}


def _three_signal_verdict(r) -> str:
    fires = (r["_flag"] >= THREE_SIGNAL["flagship"] and r["_aug"] >= THREE_SIGNAL["augmented"]
             and r["_tp_bp"] >= THREE_SIGNAL["tp_bp"])
    confirmed = r["_flag"] >= THREE_SIGNAL["flagship"] and r["_aug"] >= THREE_SIGNAL["augmented"]
    if r["outcome"] == "recession":
        return "✓ signaled" if fires else ("missed (TP veto)" if confirmed else "missed")
    return "FALSE POSITIVE" if fires else ("✓ vetoed (TP)" if confirmed else "not signaled")


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

    # Section 5: signal scorecard + three-signal rule (one table -- same rows, same signals;
    # the last column is the rule's verdict).
    try:
        scorecard = recession_signal_table(panel)
        ts = scorecard.copy()
        # Merge the TP level and its diagnostic class into one column, e.g. "+21 bp (compressed)".
        ts["ACM term premium"] = ts.apply(
            lambda r: f"{r['_tp_bp']:+.0f} bp ({r['TP diagnostic']})" if pd.notna(r["_tp_bp"]) else "—", axis=1)
        ts["3-signal result"] = ts.apply(_three_signal_verdict, axis=1)
        disp = ["period", "outcome", "flagship 12m", "augmented 3m", "ACM term premium", "3-signal result"]
        ts[disp].to_csv(config.OUTPUTS_DIR / "recession_signal_table.csv", index=False)
        rec = ts[ts["outcome"] == "recession"]
        fa = ts[ts["outcome"] == "false alarm"]
        caught = int((rec["3-signal result"] == "✓ signaled").sum())
        fps = int((fa["3-signal result"] == "FALSE POSITIVE").sum())
        lines.append("\n\n## 5. Signal scorecard and the three-signal rule\n")
        lines.append(
            "Every recession and every genuine flagship **false-alarm** episode, scored by the three "
            "signals: peak OOS probability from the **flagship** (term spread, 12m, in the 24 months "
            "before each NBER peak) and the **augmented** model (term + Baa−Aa, 3m), plus the **ACM "
            "term premium** prevailing over the run-up (level and diagnostic class). **✓** marks a "
            "flagship peak ≥ the episode-optimal **30%** threshold. The final column applies the "
            f"**three-signal rule** (experimental): call a recession only when all three confirm — "
            f"flagship ≥ {THREE_SIGNAL['flagship']:.0f}% AND augmented ≥ {THREE_SIGNAL['augmented']:.0f}% "
            f"AND term premium ≥ {THREE_SIGNAL['tp_bp']:.0f} bp (not compressed). The compressed-premium "
            "**veto** is the active ingredient — it discards the inversions the curve gets wrong.\n")
        lines.append(md_table(ts[disp]))
        lines.append(
            f"\n\n**Result: {caught}/{len(rec)} recessions signaled, {fps} false positives.** The only "
            "miss is **2020 — COVID** (term premium −75 bp, vetoed), which no yield-curve model "
            "genuinely predicted.\n")
        lines.append(
            "\n_**Thresholds.** The episode-optimal flagship cut is **≈30%** — it catches all 8 "
            "recessions (1969–2020) with just **2 false alarms (1966 & 2022–23)**; the per-month "
            "Youden cut (15%) adds shallower false alarms (e.g. 1995–96, 1998) for no extra "
            "recessions, so 30% dominates it. The **augmented** 3m model is inherently noisier — to "
            "catch all 8 alone it needs ~15% (≈7 false alarms), since 1969 only reached 17% — so it "
            "works as a near-term confirmer inside the rule, not a standalone signal._\n")
        lines.append(
            "\n_**Caveats — read the rule as in-sample, not validated.** (1) Three thresholds "
            "(30% / 15% / 50 bp) are fit to 8 recessions + 2 false alarms; with ~10 events a "
            "three-knob rule cannot be out-of-sample-validated, so this is a clean *in-sample* pattern, "
            "not a proven forecasting rule. (2) **2007 clears the veto by a hair** — its premium was "
            "declining (≈56 bp over 18 months, ≈45 bp over the final 12), so a different window flips "
            "it. (3) The veto rests on only **two** false alarms (1966, 2022–23). (4) The term premium "
            "works here as a **veto**, not a regressor — adding it to the probit did not robustly help "
            "(it just added noise); its value is in discarding distorted inversions._\n")

    except Exception as e:  # noqa: BLE001 - scorecard is optional; never break the report
        lines.append(f"\n\n## 5. Signal scorecard\n\n_Unavailable this run ({e})._\n")

    lines.append(
        "\n\n## 6. Real-time limits of this validation\n\n"
        "Two ways the backtest is friendlier than real life, worth keeping in mind:\n\n"
        "- **NBER dating lag.** The recession target uses final NBER dates, but the committee "
        "announces peaks ~4–12 months after the fact. The expanding-window OOS estimation "
        "therefore trains on labels a real-time forecaster would not yet have had for the most "
        "recent months of each training window. This is standard in the literature (the yield "
        "curve itself is never revised, so the *inputs* are real-time clean), but the reported "
        "OOS metrics are best read as *pseudo* real-time. A full vintage exercise (ALFRED "
        "supports it) is possible future work.\n"
        "- **Live-data dependency.** The augmented 3-month clock needs the Moody's Aa yield "
        "(DataBuffet `IRAACM.IUSA`); if that feed is unavailable the report degrades gracefully "
        "to the flagship + term-premium signals. The flagship and the ACM term premium run on "
        "public FRED / NY Fed data alone.\n")

    lines.append(
        "\n\n## 7. What else was tried — and why it isn't in the model\n\n"
        "The model's parsimony is a result, not an assumption: each candidate below was tested "
        "in the same framework (expanding-window out-of-sample probit; per-recession peak "
        "probabilities; episode operating point) and rejected. Each remains a runnable module "
        "under `yield_curve/models/` with the full analysis in its docstring; `RESEARCH.md` in "
        "the repo has the complete log.\n\n"
        "| Candidate | Why it was rejected |\n"
        "| --- | --- |\n"
        "| **Term premium as a regressor** | Works only as the veto. As a probit factor the OOS "
        "gain flips sign between the ACM (1961+) and Kim–Wright (1990+) samples. |\n"
        "| **Alternative credit spreads** (Baa−10Y, Baa−Aaa, GZ excess bond premium) | None "
        "robustly beats Baa−Aa: the Treasury leg contaminates Baa−10Y, the Aaa universe has "
        "shrunk to ~2 issuers, and the EBP adds no OOS improvement. |\n"
        "| **Fed balance sheet** (growth, /GDP level, change) | Reactive/coincident — the Fed "
        "expands *into* weakness; the /GDP level's apparent 0.92 AUC was non-stationary "
        "spuriousness. |\n"
        "| **Real house-price growth** (FHFA, spliced to 1975) | Housing-led-recession "
        "specialist: raises aggregate AUC by being calm in expansions while *muting* the "
        "non-housing recessions (1990/2001/2020); breaks 2001 in the augmented model. |\n"
        "| **Corporate credit growth / leverage** (Schularick–Taylor) | ~30 specifications: "
        "right sign but nothing at the 12–24m horizons where credit booms should lead; the "
        "near-term flicker is redundant with Baa−Aa; leverage levels are non-stationary or "
        "post-crash equity artifacts. |\n"
        "| **SLOOS C&I lending standards** | C&I-sector specialist: nailed 2001 (98%) but was "
        "*blind to 2007* — banks eased C&I standards into the peak; adding it drops the 2007 "
        "call from ~65% to 0%. |\n"
        "| **SLOOS consumer-loan willingness** (recovered to 1966 via a validated vintage "
        "splice) | Genuinely higher near-term AUC than Baa−Aa — but discrimination-only: ~2× "
        "the false-positive months, *lower* per-recession peaks, and it guts the term-spread "
        "coefficient. The promising 1982+ result was a small-sample artifact the longer "
        "history corrected. |\n"
        "| **Philly Fed current general activity** | A strong coincident nowcaster, not a "
        "predictor: ~3× the false-positive months; no smoothing window works — the signal and "
        "the noise are the same high-frequency swings. |\n"
        "| **Near-term forward spread** (Engstrom–Sharpe, built from the GSW curve) | Only a "
        "marginal edge over 10Y−3M (quieter, not sharper) and adds nothing once the "
        "term-premium veto exists; not worth replacing a transparent flagship. |\n"
        "| **Building permits** (Leamer) | Housing specialist: −23% to −49% before housing-led "
        "recessions but flat before 2001 and *growing* into 2020 — as a fourth AND-condition it "
        "could only turn 7/8 into 6/8. |\n"
        "| **5-year change in the 5Y yield** (refinancing-wall hypothesis) | Rate-shock "
        "specialist (helps only 1980–81); dominated by the secular rate trend, 4–12× the false "
        "alarms, and flashing falsely since 2023. |\n"
        "| **High-yield OAS** | Untestable — FRED's ICE series is truncated. |\n\n"
        "_The recurring lesson: **aggregate AUC flatters.** Five candidates raised AUC while "
        "worsening the operating point — the per-recession peaks and the false-alarm count are "
        "what decide whether a variable earns a place._\n")

    lines.append(
        "\n\n## 8. Selected literature\n\n"
        "The yield curve is the most-studied single recession indicator in empirical "
        "macroeconomics. The core papers, in rough chronological order:\n\n"
        "| Study | Contribution |\n"
        "| --- | --- |\n"
        "| **Kessel (1965)**, NBER | First systematic documentation that the term structure "
        "flattens and inverts around business-cycle turning points. |\n"
        "| **Harvey (1986, 1988)**, U. Chicago dissertation; *JFE* | The origin of the modern "
        "use: an inverted real curve predicts slower consumption growth and output — the curve "
        "as a forward-looking recession signal. |\n"
        "| **Stock & Watson (1989)**, *NBER Macro Annual* | Put term-structure slopes into the "
        "modern leading-indicator index framework. |\n"
        "| **Estrella & Hardouvelis (1991)**, *J. Finance* | The 10Y−3M slope predicts real "
        "activity 1–2 years out, beating other financial variables. |\n"
        "| **Estrella & Mishkin (1996, 1998)**, NY Fed; *REStat* | The canonical **probit** "
        "P(recession in t+h | spread) — this project's framework. The curve dominates other "
        "indicators at 2–6-quarter horizons; ~30% probability historically corresponds to "
        "inversion. |\n"
        "| **Dueker (1997)**, St. Louis Fed *Review* | Dynamic/Markov-switching probit "
        "extensions strengthen the case; lagged recession state matters. |\n"
        "| **Chauvet & Potter (2005)**, *J. Forecasting* | Probit with breaks and "
        "autocorrelated errors — naive probit probabilities are optimistic about their own "
        "precision. |\n"
        "| **Kim & Wright (2005)**, Fed FEDS | The arbitrage-free term-premium estimate used "
        "here as the cross-check gauge. |\n"
        "| **Estrella & Trubin (2006)**, NY Fed *Current Issues* | The practitioner's guide: "
        "use 10Y−3M, monthly averages, secondary-market bill on a consistent basis — the "
        "conventions this tracker follows (incl. the bond-equivalent conversion). |\n"
        "| **Wright (2006)**, Fed FEDS | Adding the funds-rate level to the slope improves "
        "fit — an early hint that *why* the curve inverts matters. |\n"
        "| **Ang, Piazzesi & Wei (2006)**, *J. Econometrics* | No-arbitrage term-structure "
        "model of GDP forecasting; the short rate carries information beyond the slope. |\n"
        "| **Rudebusch & Williams (2009)**, *JBES* | \"The puzzle of the enduring power of the "
        "yield curve\": the simple curve probit beats professional forecasters, who "
        "persistently ignore it. |\n"
        "| **Adrian, Crump & Moench (2013)**, *JFE* | The **ACM** term-premium decomposition — "
        "the gauge behind this report's veto/trust check (1961+). |\n"
        "| **Bauer & Mertens (2018)**, SF Fed *Economic Letters* | Post-QE re-validation: "
        "10Y−3M remains the single best predictor; skeptical of \"this time is different\" "
        "arguments — a useful counterweight to our veto (see below). |\n"
        "| **Benzoni, Chyruk & Kelley (2018)**, Chicago Fed *Letter* | Decomposes the slope's "
        "predictive power: inversions driven by falling **expected rates** predict recessions; "
        "term-premium-driven moves do not — the direct rationale for the compressed-premium "
        "veto. |\n"
        "| **Johansson & Meldrum (2018)**, Fed FEDS Notes | Term-premium-adjusted probits: "
        "conditioning recession probabilities on the premium level. |\n"
        "| **Engstrom & Sharpe (2019)**, *FAJ* | The **near-term forward spread** (market-priced "
        "cuts) as a \"less distorted mirror\" — tested directly here (§7): only a marginal edge "
        "in this framework once the veto exists. |\n\n"
        "Adjacent literatures behind the §7 candidates: **Gilchrist & Zakrajšek (2012)** "
        "(excess bond premium), **Schularick & Taylor (2012)** (credit booms gone bust), "
        "**Leamer (2007)** (housing is the business cycle), and **Lown & Morgan (2006)** "
        "(SLOOS lending standards).\n\n"
        "_Where this tracker sits: Estrella–Mishkin probit mechanics with Estrella–Trubin "
        "data conventions; the Benzoni-et-al. decomposition logic implemented as an explicit "
        "ACM-based veto rather than a regressor (per section 5); and the Bauer–Mertens caution "
        "acknowledged — the veto is what makes this tracker's read of 2022–23 \"different,\" "
        "and it is flagged as in-sample-tuned for exactly that reason._\n")

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
