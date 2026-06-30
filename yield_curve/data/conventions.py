"""Day-count / quote conventions.

The 3-month Treasury bill (FRED TB3MS / DTB3) is quoted on a DISCOUNT basis, while the
10Y and 2Y constant-maturity series are coupon-equivalent yields. Before differencing
the 3M leg against the 10Y leg, convert the bill's discount rate to a coupon-equivalent
(bond-equivalent) yield, matching the NY Fed / U.S. Treasury convention.
"""

from __future__ import annotations


def discount_to_bond_equivalent(discount_pct: float, days_to_maturity: int = 91) -> float:
    """T-bill discount rate (percent) -> coupon-equivalent (bond-equivalent) yield
    (percent), for bills with <= 182 days to maturity.

        CE = (365 * d) / (360 - d * N)

    where d is the discount rate as a decimal and N is days to maturity. For a 13-week
    bill N ~= 91, which raises the quoted rate by roughly 8-12 bp at today's levels (so
    the 10Y-3M spread comes out a touch lower than the raw difference).
    """
    d = discount_pct / 100.0
    denom = 360.0 - d * days_to_maturity
    if denom <= 0:
        return discount_pct  # degenerate guard
    return (365.0 * d) / denom * 100.0
