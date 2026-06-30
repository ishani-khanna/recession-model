"""Current-reading tests (Phase 6): probabilities must be valid in [0, 1]."""

from __future__ import annotations

import numpy as np
import pytest

from yield_curve import reporting
from yield_curve.data import config


def test_probability_table_in_unit_interval(panel):
    tbl = reporting.probability_table(panel, {k: 0.0 for k in config.SPREADS})
    arr = tbl.values.astype(float)
    assert np.isfinite(arr).all()
    assert (arr >= 0).all() and (arr <= 100).all()


@pytest.mark.parametrize("spread_value", [-3.0, -1.0, 0.0, 1.0, 3.0])
def test_probabilities_bounded_for_extreme_spreads(panel, spread_value):
    tbl = reporting.probability_table(panel, {k: spread_value for k in config.SPREADS})
    assert (tbl.values >= 0).all() and (tbl.values <= 100).all()


def test_inversion_raises_probability_vs_steep_curve(panel):
    inverted = reporting.probability_table(panel, {k: -2.0 for k in config.SPREADS})
    steep = reporting.probability_table(panel, {k: 2.0 for k in config.SPREADS})
    assert (inverted.loc[config.DEFAULT_HORIZON] > steep.loc[config.DEFAULT_HORIZON]).all()
