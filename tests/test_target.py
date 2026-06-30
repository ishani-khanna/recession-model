"""Tests for the recession target (Phase 2): the no-look-ahead property is the
one the implementation plan most insists on."""

from __future__ import annotations

import pandas as pd
import pytest

from yield_curve.data import config
from yield_curve.models.target import (
    assert_no_look_ahead,
    build_target,
    cross_check_nber,
)


def _synthetic_usrec() -> pd.Series:
    idx = pd.date_range("2000-01-01", "2002-12-01", freq="MS")
    s = pd.Series(0, index=idx, dtype=int)
    s.loc["2001-04-01":"2001-09-01"] = 1  # a 6-month recession
    return s


@pytest.mark.parametrize("h", [3, 6, 12])
def test_point_target_equals_future_usrec(h):
    u = _synthetic_usrec()
    y = build_target(u, h, "point")
    off = pd.DateOffset(months=h)
    for t in y.index:
        assert y.loc[t] == u.loc[t + off]


def test_target_drops_unobservable_tail():
    u = _synthetic_usrec()
    h = 6
    y = build_target(u, h, "point")
    assert y.index.max() == u.index.max() - pd.DateOffset(months=h)


def test_shift_direction_predicts_in_advance():
    """A recession starting 2001-04 must light up the target h months earlier,
    never later -- this is what catches a shift sign error."""
    u = _synthetic_usrec()
    y = build_target(u, 3, "point")
    assert y.loc["2001-01-01"] == 1   # 3 months before the recession start
    assert y.loc["2000-06-01"] == 0   # t+3 lands well before the recession


@pytest.mark.parametrize("h", [3, 6, 12, 18, 24])
def test_assert_no_look_ahead_passes(h):
    assert_no_look_ahead(_synthetic_usrec(), h)  # must not raise


def test_any_within_window():
    u = _synthetic_usrec()
    y = build_target(u, 6, "any")
    assert y.loc["2000-10-01"] == 1   # window (2000-11..2001-04] hits 2001-04
    assert y.loc["2000-09-01"] == 0   # window (2000-10..2001-03] misses it


def test_nber_chronology_matches_usrec():
    """Hard-coded NBER turning points must reproduce FRED USREC exactly."""
    p = config.CACHE_DIR / "panel_monthly.parquet"
    if not p.exists():
        pytest.skip("panel not built; run yield_curve.data.build_dataset first")
    panel = pd.read_parquet(p)
    cc = cross_check_nber(panel["usrec"])
    assert cc["n_mismatch"] == 0, cc["mismatch_dates"]
