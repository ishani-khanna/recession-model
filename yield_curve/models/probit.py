"""Probit recession model (Phase 3).

Fits ``P(recession in t+h) = Phi(b0 + b1 * spread_t)`` for each (spread, horizon)
over the full available sample, with a Logit robustness variant and an exclude-2020
variant. Standard errors are HAC (Newey-West) with ``maxlags = horizon`` because the
h-month-ahead target induces overlapping observations and hence serially correlated
errors -- the textbook correction for this forecasting setup.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

from ..data import build_dataset, config
from .target import build_target


def _spread_col(spread_key: str) -> str:
    return f"spread_{spread_key}"


def prepare_xy(
    panel: pd.DataFrame,
    spread_key: str,
    horizon: int,
    exclude_2020: bool = False,
    target_type: str = "point",
) -> tuple[pd.Series, pd.Series]:
    """Aligned (spread_t, y_t) where y_t = USREC_{t+h}; drops the unobservable tail."""
    spread = panel[_spread_col(spread_key)].dropna()
    y = build_target(panel["usrec"], horizon, target_type)
    df = pd.concat({"spread": spread, "y": y}, axis=1).dropna()
    if exclude_2020:
        target_month = df.index + pd.DateOffset(months=horizon)
        df = df[target_month.year != 2020]  # drop predictions of the COVID recession
    return df["spread"], df["y"].astype(int)


@dataclass
class FitResult:
    spread: str
    horizon: int
    link: str
    sample: str            # "full" or "ex2020"
    target_type: str
    n: int
    coef: float            # b1 on the spread
    z: float
    pval: float
    intercept: float
    pseudo_r2: float
    auc: float
    cov_type: str
    converged: bool
    current_spread: float
    current_prob: float
    fitted_probs: pd.Series = field(repr=False)

    def row(self) -> dict:
        d = {k: getattr(self, k) for k in (
            "spread", "horizon", "link", "sample", "n",
            "coef", "z", "pval", "pseudo_r2", "auc",
            "current_spread", "current_prob", "cov_type", "converged",
        )}
        return d


def fit_spread_model(
    panel: pd.DataFrame,
    spread_key: str,
    horizon: int,
    link: str = "probit",
    exclude_2020: bool = False,
    target_type: str = "point",
) -> FitResult:
    x, y = prepare_xy(panel, spread_key, horizon, exclude_2020, target_type)
    X = sm.add_constant(x.to_frame("spread"))
    Model = {"probit": sm.Probit, "logit": sm.Logit}[link]

    # HAC SEs for the overlapping-horizon autocorrelation; fall back gracefully.
    try:
        res = Model(y, X).fit(disp=0, cov_type="HAC", cov_kwds={"maxlags": horizon})
        cov_type = f"HAC(maxlags={horizon})"
    except Exception:  # noqa: BLE001
        try:
            res = Model(y, X).fit(disp=0, cov_type="HC1")
            cov_type = "HC1"
        except Exception:  # noqa: BLE001
            res = Model(y, X).fit(disp=0)
            cov_type = "nonrobust"

    probs = pd.Series(res.predict(X), index=X.index, name="prob")
    auc = float(roc_auc_score(y, probs)) if y.nunique() == 2 else float("nan")

    cur_spread = float(panel[_spread_col(spread_key)].dropna().iloc[-1])
    cur_prob = float(res.predict(np.array([[1.0, cur_spread]]))[0])

    return FitResult(
        spread=spread_key,
        horizon=horizon,
        link=link,
        sample="ex2020" if exclude_2020 else "full",
        target_type=target_type,
        n=int(len(y)),
        coef=float(res.params["spread"]),
        z=float(res.tvalues["spread"]),
        pval=float(res.pvalues["spread"]),
        intercept=float(res.params["const"]),
        pseudo_r2=float(res.prsquared),
        auc=auc,
        cov_type=cov_type,
        converged=bool(res.mle_retvals.get("converged", True)),
        current_spread=cur_spread,
        current_prob=cur_prob,
        fitted_probs=probs,
    )


def results_grid(
    panel: pd.DataFrame,
    spreads: list[str] | None = None,
    horizons: list[int] | None = None,
    links: tuple[str, ...] = ("probit", "logit"),
    samples: tuple[bool, ...] = (False, True),
    target_type: str = "point",
) -> pd.DataFrame:
    spreads = spreads or list(config.SPREADS.keys())
    horizons = horizons or config.HORIZONS
    rows = []
    for spread_key in spreads:
        for h in horizons:
            for link in links:
                for ex in samples:
                    try:
                        fr = fit_spread_model(panel, spread_key, h, link, ex, target_type)
                        rows.append(fr.row())
                    except Exception as e:  # noqa: BLE001
                        rows.append({
                            "spread": spread_key, "horizon": h, "link": link,
                            "sample": "ex2020" if ex else "full", "n": 0,
                            "coef": np.nan, "z": np.nan, "pval": np.nan,
                            "pseudo_r2": np.nan, "auc": np.nan,
                            "current_spread": np.nan, "current_prob": np.nan,
                            "cov_type": f"ERROR: {e}", "converged": False,
                        })
    grid = pd.DataFrame(rows)
    num = ["coef", "z", "pval", "pseudo_r2", "auc", "current_spread", "current_prob"]
    grid[num] = grid[num].astype(float).round(4)
    return grid


def main() -> None:
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    panel = build_dataset.build_panel(refresh=False)
    grid = results_grid(panel)
    grid.to_csv(config.OUTPUTS_DIR / "results_grid.csv", index=False)

    # Headline view: primary probit, full sample.
    headline = grid[(grid.link == "probit") & (grid["sample"] == "full")]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("=== Probit, full sample (coef should be NEGATIVE & significant) ===")
        print(headline.to_string(index=False))
        print(f"\nFull grid ({len(grid)} rows) saved to {config.OUTPUTS_DIR/'results_grid.csv'}")


if __name__ == "__main__":
    main()
