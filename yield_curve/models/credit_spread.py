"""Credit-spread measures + the augmented (term + credit) model.

Reconstructed to match reporting.py's usage:
    from .models.credit_spread import fit_insample, load_baa_aa

- load_baa_aa(): monthly credit-quality spread (pp).
- fit_insample(df, feature_cols, horizon): in-sample probit on the given features;
  returns (statsmodels_result, auc, n). The result's .params is indexed by
  'const' plus the feature column names (e.g. 'spread_10y3m', 'baa_aa').

RECONSTRUCTION NOTE: the project README labels the credit-quality spread "Baa-Aa". FRED
publishes Moody's Aaa (AAA) and Baa (BAA) but not a clean long-history "Aa", so this
uses BAA - AAA (the classic default-risk spread). If a distinct Aa series is used,
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


def _load_aa_leg() -> pd.Series:
    """The Aa leg of the credit-quality spread.

    WIRED FOR DATABUFFET, FLIP-ON-LATER (config-only): when the Moody's DataBuffet
    credentials are present, use Moody's Aa yield (IRAACM.IUSA); otherwise fall back to
    FRED Aaa (AAA) as the FRED-only substitute. Adding the two keys is the only change
    needed to switch signal 2 to the true Baa-Aa - no code edits.
    """
    try:
        from ..data import config as _cfg
        _cfg.databuffet_keys()                     # raises if the two keys are absent
        from ..data.databuffet_client import DataBuffetClient
        s = DataBuffetClient().get_series("IRAACM.IUSA")
        return _to_month_start(pd.to_numeric(s, errors="coerce").dropna())
    except Exception:
        return _to_month_start(FredClient().get_series("AAA").dropna())


def load_baa_aa() -> pd.Series:
    """Monthly credit-quality spread = Moody's Baa - Aa (pp). Baa from FRED (BAA); the Aa leg
    is Moody's Aa via DataBuffet when keys are set, else FRED Aaa as the FRED-only substitute."""
    baa = _to_month_start(FredClient().get_series("BAA").dropna())
    aa = _load_aa_leg()
    spread = (baa - aa).dropna()
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
