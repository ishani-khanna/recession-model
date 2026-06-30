"""Source-reconciliation tests (Phase 6)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from yield_curve.data import build_dataset


def test_compare_within_tolerance_no_flags():
    idx = pd.date_range("2000-01-01", periods=24, freq="MS")
    fred = pd.Series(np.linspace(2, 4, 24), index=idx)
    db = fred + 0.005  # 0.5 bp apart -> within the 2 bp tolerance
    c = build_dataset.compare_series(fred, db)
    assert c["n_overlap"] == 24
    assert c["n_gt_tol"] == 0
    assert c["max_abs_diff_bp"] <= 2.0


def test_compare_flags_divergence():
    idx = pd.date_range("2000-01-01", periods=12, freq="MS")
    fred = pd.Series(3.0, index=idx)
    db = fred.copy()
    db.iloc[5] += 0.10  # a 10 bp divergence in one month
    c = build_dataset.compare_series(fred, db)
    assert c["n_gt_tol"] == 1
    assert c["max_abs_diff_bp"] == pytest.approx(10.0, abs=1e-6)
    assert c["worst_date"] == idx[5]


def test_compare_empty_overlap():
    a = pd.Series([1.0], index=pd.to_datetime(["2000-01-01"]))
    b = pd.Series([1.0], index=pd.to_datetime(["2010-01-01"]))
    assert build_dataset.compare_series(a, b)["n_overlap"] == 0


@pytest.mark.integration
def test_live_reconciliation_within_tolerance(panel):
    """FRED vs DataBuffet must agree within 2 bp on the live pull. Skips offline."""
    try:
        rec = build_dataset.reconcile(panel)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"reconciliation needs network/credentials: {e}")
    usable = rec[rec["db_source"].isin(["api", "csv"])]
    if usable.empty:
        pytest.skip("no DataBuffet source available")
    assert (usable["mean_abs_diff_bp"].fillna(0) <= 2.0).all()
    assert (usable["n_gt_2bp"].fillna(0) == 0).all()
