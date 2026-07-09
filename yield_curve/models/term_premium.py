"""Does conditioning on the term premium improve the recession models?

Decompose the long yield into an expectations component and a term premium:
``10Y = E[avg short rate] + TP``. An inversion driven by *expected rate cuts*
(the expectations component) is genuinely recession-predictive; one driven by a
*compressed term premium* (as in 2022-23) is not. So the term-premium-adjusted
"expectations spread" = ``spread - TP10`` should be a cleaner signal, and TP added
as a second regressor should carry incremental, positively-signed information.

Three specs are compared per (spread, horizon), all on the same sample:
  - baseline    : Phi(b0 + b1*spread)
  - spread+tp   : Phi(b0 + b1*spread + b2*tp)        (b2>0 expected)
  - exp_spread  : Phi(b0 + b1*(spread - tp))         (expectations component)

Term premium: ACM 10Y (NY Fed, 1961+, cached) by default; Kim-Wright THREEFYTP10
(FRED, 1990+) as a robustness cross-check.

    python -m yield_curve.models.term_premium [--tp acm|kw]
"""

from __future__ import annotations

import argparse
import io
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import brier_score_loss, roc_auc_score

from ..data import build_dataset, config
from ..data.fred_client import FredClient
from .target import build_target

warnings.filterwarnings("ignore")

SPECS = {
    "baseline": ["spread"],
    "spread+tp": ["spread", "tp"],
    "exp_spread": ["exp_spread"],
}
_ACM_URL = "https://www.newyorkfed.org/medialibrary/media/research/data_indicators/ACMTermPremium.xls"
_ACM_CACHE = config.CACHE_DIR / "acm_tp10.parquet"


def _to_ms(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = pd.to_datetime(s.index).to_period("M").to_timestamp()
    return s[~s.index.duplicated(keep="last")].sort_index()


def load_term_premium(source: str = "acm", refresh: bool = False) -> pd.Series:
    """10Y term premium as a monthly Series (percentage points)."""
    if source == "kw":
        return _to_ms(FredClient().get_series("THREEFYTP10")).rename("tp")
    # ACM
    if refresh or not _ACM_CACHE.exists():
        import requests
        r = requests.get(_ACM_URL, timeout=90)
        df = pd.read_excel(io.BytesIO(r.content), sheet_name="ACM Monthly")
        df = df[["DATE", "ACMTP10"]].rename(columns={"DATE": "date", "ACMTP10": "acmtp10"})
        df["date"] = pd.to_datetime(df["date"])
        df.dropna().to_parquet(_ACM_CACHE)
    df = pd.read_parquet(_ACM_CACHE)
    s = pd.Series(df["acmtp10"].values, index=pd.to_datetime(df["date"]), name="tp")
    return _to_ms(s)


def make_features(panel: pd.DataFrame, tp: pd.Series, spread_key: str) -> pd.DataFrame:
    df = pd.concat({"spread": panel[f"spread_{spread_key}"], "tp": tp}, axis=1).dropna()
    df["exp_spread"] = df["spread"] - df["tp"]
    return df


# --------------------------------------------------------------------------- #
def fit_insample(panel, tp, spread_key, horizon, spec):
    feats = make_features(panel, tp, spread_key)
    y = build_target(panel["usrec"], horizon)
    d = feats.join(y.rename("y"), how="inner").dropna()
    X = sm.add_constant(d[SPECS[spec]])
    res = sm.Probit(d["y"], X).fit(disp=0)
    auc = roc_auc_score(d["y"], res.predict(X))
    return res, auc, int(len(d))


def expanding_oos(panel, tp, spread_key, horizon, spec, min_train=120):
    feats = make_features(panel, tp, spread_key)
    y_full = build_target(panel["usrec"], horizon)
    aligned = feats.join(y_full.rename("y"), how="inner").dropna()
    cols = SPECS[spec]
    recs = []
    for t in feats.index:
        cutoff = t - pd.DateOffset(months=horizon)
        tr = aligned.loc[aligned.index <= cutoff]
        prob = np.nan
        if len(tr) >= min_train and tr["y"].nunique() == 2:
            Xtr = sm.add_constant(tr[cols])
            try:
                res = sm.Probit(tr["y"], Xtr).fit(disp=0)
                xrow = np.array([[1.0, *[feats.loc[t, c] for c in cols]]])
                prob = float(res.predict(xrow)[0])
            except Exception:  # noqa: BLE001
                pass
        recs.append((t, prob, y_full.get(t, np.nan)))
    path = pd.DataFrame(recs, columns=["date", "prob", "y"]).set_index("date")
    scored = path.dropna()
    auc = roc_auc_score(scored["y"], scored["prob"]) if scored["y"].nunique() == 2 else np.nan
    brier = brier_score_loss(scored["y"], scored["prob"]) if len(scored) else np.nan
    return path, auc, brier


def analyze(panel, tp, spread_key, horizon=12):
    rows = []
    paths = {}
    for spec in SPECS:
        res, in_auc, n = fit_insample(panel, tp, spread_key, horizon, spec)
        path, oos_auc, brier = expanding_oos(panel, tp, spread_key, horizon, spec)
        paths[spec] = path
        tp_coef = res.params.get("tp", np.nan)
        tp_p = res.pvalues.get("tp", np.nan)
        peak2223 = path["prob"].loc["2022-01-01":"2024-12-31"].max()
        rows.append({
            "spec": spec, "n": n, "in_auc": round(in_auc, 3),
            "oos_auc": round(oos_auc, 3), "brier": round(brier, 4),
            "tp_coef": round(tp_coef, 3) if pd.notna(tp_coef) else None,
            "tp_p": round(tp_p, 3) if pd.notna(tp_p) else None,
            "peak_2022_23": round(float(peak2223) * 100, 0) if pd.notna(peak2223) else None,
        })
    return pd.DataFrame(rows).set_index("spec"), paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", choices=["acm", "kw"], default="acm")
    ap.add_argument("--horizon", type=int, default=config.DEFAULT_HORIZON)
    args = ap.parse_args()

    panel = build_dataset.build_panel(refresh=False)
    tp = load_term_premium(args.tp)
    print(f"Term premium: {args.tp.upper()}  ({tp.index.min().date()} -> {tp.index.max().date()})  "
          f"h={args.horizon}\n")
    for spread_key in config.SPREADS:
        sample = make_features(panel, tp, spread_key)
        grid, _ = analyze(panel, tp, spread_key, args.horizon)
        print(f"=== {config.SPREADS[spread_key]['label']}  "
              f"(sample {sample.index.min().date()} -> {sample.index.max().date()}) ===")
        print(grid.to_string())
        print()


if __name__ == "__main__":
    main()
