"""Credit-spread measures + the augmented (term + credit) model.

Reconstructed to match reporting.py's usage:
    from .models.credit_spread import fit_insample, load_baa_aa

- load_baa_aa(): monthly credit-quality spread (pp).
- fit_insample(df, feature_cols, horizon): in-sample probit on the given features;
  returns (statsmodels_result, auc, n). The result's .params is indexed by
  'const' plus the feature column names (e.g. 'spread_10y3m', 'baa_aa').

RECONSTRUCTION NOTE: Zandi's README labels the credit-quality spread "Baa-Aa". FRED
publishes Moody's Aaa (AAA) and Baa (BAA) but not a clean long-history "Aa", so this
uses BAA - AAA (the classic default-risk spread). If Zandi used a distinct Aa series,
replace load_baa_aa only; fit_insample and all downstream code are unaffected.
"""

from __future__ import annotations

import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

from ..data.fred_client import FredClient
from .target import build_target

_MODELS = {"probit": sm.Probit, "logit": sm.Logit}


def _to_month_start(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = pd.to_datetime(s.index).to_period("M").to_timestamp()
    return s[~s.index.duplicated(keep="last")].sort_index()


def load_baa_aa() -> pd.Series:
    """Monthly credit-quality spread = Moody's Baa - Aaa (pp), from FRED (BAA, AAA)."""
    c = FredClient()
    baa = _to_month_start(c.get_series("BAA").dropna())
    aaa = _to_month_start(c.get_series("AAA").dropna())
    spread = (baa - aaa).dropna()
    spread.name = "baa_aa"
    return spread


def fit_insample(df: pd.DataFrame, feature_cols: list[str], horizon: int,
                 link: str = "probit"):
    """In-sample probit of P(recession in t+h) on feature_cols.

    df must contain 'usrec' and every column in feature_cols.
    Returns (statsmodels_result, auc, n).
    """
    y = build_target(df["usrec"], horizon, "point")          # int, no look-ahead
    X = df[feature_cols].dropna()
    data = pd.concat({"y": y}, axis=1).join(X, how="inner").dropna()
    yy = data["y"].astype(int)
    XX = sm.add_constant(data[feature_cols])
    res = _MODELS[link](yy, XX).fit(disp=0)
    probs = res.predict(XX)
    auc = float(roc_auc_score(yy, probs)) if yy.nunique() == 2 else float("nan")
    return res, auc, int(len(yy))
