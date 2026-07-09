"""ACM 10Y term-premium reading for the web pages — thin re-export of the engine's
canonical source so the report and the dashboard never diverge.

The single source of truth is ``yield_curve.models.term_premium.load_term_premium``
(NY Fed ACM "ACM Monthly" sheet, cached; Kim-Wright THREEFYTP10 is the cross-check).
``acm_reading`` just packages that series into the dict the landing-page generator
needs, using the SAME current-value / percentile / gate definitions as the report
(reporting.two_clock_dashboard): current = latest monthly reading, percentile =
share of monthly history below it, compressed = below the 50 bp veto gate.
"""

from __future__ import annotations

import pandas as pd


def load_acm_tp10(refresh: bool = False) -> pd.Series:
    """ACM 10-year term premium as a monthly Series (pp) — the canonical engine feed."""
    from ..models.term_premium import load_term_premium  # lazy: avoids import cycle
    return load_term_premium("acm", refresh=refresh).rename("acm_tp10")


def acm_reading(refresh: bool = False) -> dict:
    """Current ACM 10Y term premium packaged for the landing page, defined exactly as
    the engine report defines it (latest monthly reading; percentile vs monthly history)."""
    monthly = load_acm_tp10(refresh=refresh).dropna()
    cur = float(monthly.iloc[-1])
    pct = round(float((monthly < cur).mean()) * 100)
    cls = "compressed" if cur < 0.50 else ("elevated" if pct > 75 else "near normal")
    return {
        "value_pp": round(cur, 3), "value_bp": round(cur * 100),
        "date": monthly.index[-1].date().isoformat(), "percentile": pct, "class": cls,
        "compressed_2020_21": round(float(monthly.loc["2020-06":"2021-12"].mean()), 3),
        "low_2022_23": round(float(monthly.loc["2022-01":"2023-12"].min()), 3),
        "low_2022_23_date": monthly.loc["2022-01":"2023-12"].idxmin().strftime("%Y-%m"),
        "monthly": monthly,
    }
