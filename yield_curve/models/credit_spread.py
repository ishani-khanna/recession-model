"""Does a corporate credit spread (Baa - 10Y) improve the recession model?

The Baa-10Y "default spread" captures credit/default risk and financial stress -- a
different signal from the term-structure slope. It WIDENS (rises) into recessions, so
its probit coefficient is POSITIVE (vs negative for the term spread), and it tends to
be more coincident (shorter lead). Crucially it does NOT react to a yield-curve
inversion that lacks credit stress -- e.g. 2022-23 -- so it should complement the term
spread. Evaluated standalone and combined with the primary 10Y-3M term spread.

    python -m yield_curve.models.credit_spread
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import brier_score_loss, roc_auc_score

from ..data import build_dataset, config
from ..data.databuffet_client import DataBuffetClient
from ..data.fred_client import FredClient
from .target import NBER_CYCLES, build_target

warnings.filterwarnings("ignore")

SPECS = {
    "term (10Y-3M)": ["spread_10y3m"],
    "credit (Baa-10Y)": ["credit"],
    "term + credit": ["spread_10y3m", "credit"],
}


def _to_ms(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = pd.to_datetime(s.index).to_period("M").to_timestamp()
    return s[~s.index.duplicated(keep="last")].sort_index()


_EBP_URL = "https://www.federalreserve.gov/econresdata/notes/feds-notes/2016/files/ebp_csv.csv"
_EBP_CACHE = config.CACHE_DIR / "ebp.parquet"


def load_credit_spread(panel: pd.DataFrame) -> pd.Series:
    """Baa - 10Y default spread (Moody's Seasoned Baa minus GS10), monthly."""
    baa = _to_ms(FredClient().get_series("BAA"))
    return (baa - panel["gs10"]).dropna().rename("credit")


def load_baa_aaa() -> pd.Series:
    """Baa - Aaa: same-maturity corporate quality spread (pure credit, no Treasury leg)."""
    fc = FredClient()
    return (_to_ms(fc.get_series("BAA")) - _to_ms(fc.get_series("AAA"))).dropna().rename("baa_aaa")


def load_baa_aa() -> pd.Series:
    """Baa - Aa quality spread. Aa is from DataBuffet (Moody's Seasoned Aa, IRAACM.IUSA);
    the Aa universe is far larger and more stable than the shrunken Aaa universe (~2 names),
    so Baa-Aa is a more robust 'pure credit' quality spread than Baa-Aaa."""
    aa = _to_ms(DataBuffetClient().get_series("IRAACM.IUSA"))
    baa = _to_ms(FredClient().get_series("BAA"))
    return (baa - aa).dropna().rename("baa_aa")


def load_ebp(refresh: bool = False) -> pd.Series:
    """Gilchrist-Zakrajšek Excess Bond Premium (Fed; 1973+). The purest credit-risk
    premium -- strips out expected-default compensation and the maturity/rate level."""
    if refresh or not _EBP_CACHE.exists():
        import io
        import requests
        r = requests.get(_EBP_URL, timeout=30)
        df = pd.read_csv(io.StringIO(r.text))
        df["date"] = pd.to_datetime(df["date"])
        df[["date", "ebp", "gz_spread", "est_prob"]].to_parquet(_EBP_CACHE)
    df = pd.read_parquet(_EBP_CACHE)
    return _to_ms(pd.Series(df["ebp"].values, index=pd.to_datetime(df["date"]), name="ebp"))


# Alternative credit measures -> panel column names.
CREDIT_MEASURES = {
    "Baa-10Y": "credit_baa10y",
    "Baa-Aaa": "credit_baaaaa",
    "Baa-Aa": "credit_baa_aa",
    "EBP (GZ)": "credit_ebp",
}


def make_panel_multi(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    df["credit_baa10y"] = load_credit_spread(panel)
    df["credit_baaaaa"] = load_baa_aaa()
    df["credit_baa_aa"] = load_baa_aa()
    df["credit_ebp"] = load_ebp()
    return df


def make_panel(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    df["credit"] = load_credit_spread(panel)
    return df


# --------------------------------------------------------------------------- #
def fit_insample(df, cols, horizon):
    y = build_target(df["usrec"], horizon)
    d = df[cols].join(y.rename("y"), how="inner").dropna()
    X = sm.add_constant(d[cols])
    res = sm.Probit(d["y"], X).fit(disp=0)
    return res, roc_auc_score(d["y"], res.predict(X)), int(len(d))


def expanding_oos(df, cols, horizon, min_train=120):
    feats = df[cols].dropna()
    y_full = build_target(df["usrec"], horizon)
    aligned = feats.join(y_full.rename("y"), how="inner").dropna()
    recs = []
    for t in feats.index:
        cutoff = t - pd.DateOffset(months=horizon)
        tr = aligned.loc[aligned.index <= cutoff]
        prob = np.nan
        if len(tr) >= min_train and tr["y"].nunique() == 2:
            Xtr = sm.add_constant(tr[cols])
            try:
                res = sm.Probit(tr["y"], Xtr).fit(disp=0)
                prob = float(res.predict(np.array([[1.0, *[feats.loc[t, c] for c in cols]]]))[0])
            except Exception:  # noqa: BLE001
                pass
        recs.append((t, prob, y_full.get(t, np.nan)))
    path = pd.DataFrame(recs, columns=["date", "prob", "y"]).set_index("date")
    sc = path.dropna()
    auc = roc_auc_score(sc["y"], sc["prob"]) if sc["y"].nunique() == 2 else np.nan
    brier = brier_score_loss(sc["y"], sc["prob"]) if len(sc) else np.nan
    return path, auc, brier


def analyze(df, horizon=12):
    rows, paths = [], {}
    for name, cols in SPECS.items():
        res, in_auc, n = fit_insample(df, cols, horizon)
        path, oos_auc, brier = expanding_oos(df, cols, horizon)
        paths[name] = path
        coefs = "  ".join(f"{c}={res.params[c]:+.2f}(p={res.pvalues[c]:.3f})" for c in cols)
        rows.append({
            "spec": name, "n": n, "in_auc": round(in_auc, 3),
            "oos_auc": round(oos_auc, 3), "brier": round(brier, 4),
            "peak_2022_23": round(float(path["prob"].loc["2022":"2024"].max()) * 100),
            "coefs": coefs,
        })
    return pd.DataFrame(rows).set_index("spec"), paths


def operating_point(path, name):
    """Threshold that catches every recession, and false-positive months at it."""
    p = path["prob"].dropna()
    peaks = [pd.Timestamp(pk) for pk, _ in NBER_CYCLES]
    covered = [pk for pk in peaks if (pk - pd.DateOffset(months=24)) >= p.index.min()
               and pk <= p.index.max() + pd.DateOffset(months=2)]
    thr = min(p.loc[pk - pd.DateOffset(months=24):pk].max() for pk in covered)

    def near(t):
        return any(0 <= (pk.year - t.year) * 12 + (pk.month - t.month) <= 24 for pk in peaks)
    fp = [t for t, v in p.items() if v >= thr and not near(t)]
    return {"name": name, "n_recessions": len(covered), "threshold_pct": round(thr * 100),
            "false_positive_months": len(fp), "fp_years": sorted({t.year for t in fp})}


def main():
    panel = build_dataset.build_panel(refresh=False)
    df = make_panel(panel)

    print("=== h=12, full sample (1953+) ===")
    grid, paths = analyze(df, 12)
    with pd.option_context("display.width", 220, "display.max_colwidth", 60):
        print(grid.to_string())

    print("\n=== OOS AUC by horizon ===")
    print(f"{'spec':<18}" + "".join(f"{h:>7}" for h in config.HORIZONS))
    for name, cols in SPECS.items():
        aucs = [expanding_oos(df, cols, h)[1] for h in config.HORIZONS]
        print(f"{name:<18}" + "".join(f"{a:>7.3f}" for a in aucs))

    print("\n=== Operating point: threshold to catch ALL recessions -> false-positive months ===")
    for name in SPECS:
        op = operating_point(paths[name], name)
        print(f"  {name:<18} catches {op['n_recessions']} at thr={op['threshold_pct']}%  "
              f"-> {op['false_positive_months']} FP months  {op['fp_years']}")


if __name__ == "__main__":
    main()
