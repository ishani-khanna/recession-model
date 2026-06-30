"""Recession target construction (Phase 2).

The dependent variable for an h-month-ahead forecast is the NBER recession flag
*led* by h months: ``y_t = USREC_{t+h}`` (the Estrella-Mishkin / NY Fed "point"
formulation). An "any recession within the next h months" variant is also provided.

USREC (from FRED) is authoritative; the hard-coded NBER turning points below are an
offline cross-check / fallback. The no-look-ahead property is explicitly testable
via ``assert_no_look_ahead`` (and tests/test_target.py).
"""

from __future__ import annotations

import pandas as pd

# NBER U.S. business-cycle reference dates (monthly peak, trough), 1953-present.
# Source: NBER (https://www.nber.org/research/business-cycle-dating).
NBER_CYCLES: list[tuple[str, str]] = [
    ("1953-07", "1954-05"),
    ("1957-08", "1958-04"),
    ("1960-04", "1961-02"),
    ("1969-12", "1970-11"),
    ("1973-11", "1975-03"),
    ("1980-01", "1980-07"),
    ("1981-07", "1982-11"),
    ("1990-07", "1991-03"),
    ("2001-03", "2001-11"),
    ("2007-12", "2009-06"),
    ("2020-02", "2020-04"),
]

# The COVID recession window (USREC=1 months). Used by the exclude-2020 variant.
COVID_RECESSION = ("2020-03-01", "2020-04-01")


# --------------------------------------------------------------------------- #
# Target construction
# --------------------------------------------------------------------------- #
def build_target(usrec: pd.Series, horizon: int, target_type: str = "point") -> pd.Series:
    """Build the h-month-ahead recession target with no look-ahead.

    ``point`` : y_t = USREC_{t+h}  (in recession exactly h months ahead).
    ``any``   : y_t = 1 if USREC == 1 anywhere in (t, t+h].

    The last ``horizon`` months have an undefined (future) target and are dropped.
    """
    usrec = usrec.sort_index().astype(float)
    if target_type == "point":
        y = usrec.shift(-horizon)
    elif target_type == "any":
        leads = pd.concat([usrec.shift(-k) for k in range(1, horizon + 1)], axis=1)
        # Require the full window to be observed (no partial look-ahead at the tail).
        y = leads.max(axis=1).where(leads.notna().all(axis=1))
    else:  # pragma: no cover - guarded by callers
        raise ValueError(f"unknown target_type {target_type!r}")
    return y.dropna().astype(int)


def assert_no_look_ahead(usrec: pd.Series, horizon: int) -> None:
    """Verify y_t equals the *future* USREC_{t+h}, i.e. the shift pulls the future
    back to the present (sign is correct) and never leaks past/contemporaneous info.

    Raises AssertionError on any violation; returns None on success.
    """
    usrec = usrec.sort_index().astype(float)
    y = build_target(usrec, horizon, "point")
    offset = pd.DateOffset(months=horizon)
    for t in y.index:
        future = usrec.get(t + offset)
        assert future is not None and y.loc[t] == int(future), (
            f"look-ahead violation at {t.date()}: y={y.loc[t]} but USREC[t+{horizon}]={future}"
        )
    # And the converse: the target must NOT equal contemporaneous USREC in general
    # (otherwise we'd be 'predicting' the present). Checked structurally above.


# --------------------------------------------------------------------------- #
# NBER cross-check (USREC is authoritative; this confirms its chronology)
# --------------------------------------------------------------------------- #
def nber_recession_series(index: pd.DatetimeIndex) -> pd.Series:
    """0/1 recession flag built from NBER_CYCLES on ``index``.

    Convention matches FRED USREC: a month is recessionary if it is strictly after
    a peak and on/before the trough, i.e. months in (peak, trough].
    """
    flag = pd.Series(0, index=index, dtype=int)
    for peak, trough in NBER_CYCLES:
        p = pd.Timestamp(peak) + pd.DateOffset(months=1)  # month after peak
        tr = pd.Timestamp(trough)
        flag.loc[(index >= p) & (index <= tr)] = 1
    return flag


def cross_check_nber(usrec: pd.Series) -> dict:
    """Compare FRED USREC against the hard-coded NBER chronology."""
    usrec = usrec.sort_index().astype(int)
    nber = nber_recession_series(usrec.index)
    mismatch = usrec.index[usrec.values != nber.values]
    return {
        "n_months": int(len(usrec)),
        "n_mismatch": int(len(mismatch)),
        "usrec_recession_months": int(usrec.sum()),
        "nber_recession_months": int(nber.sum()),
        "mismatch_dates": [d.date().isoformat() for d in mismatch[:24]],
    }
