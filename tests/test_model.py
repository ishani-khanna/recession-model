"""Model tests (Phase 6): the spread coefficient must be negative and significant."""

from __future__ import annotations

import pytest

from yield_curve.data import config
from yield_curve.models import probit


def test_full_sample_primary_coef_negative_significant(panel):
    fr = probit.fit_spread_model(panel, config.PRIMARY_SPREAD, config.DEFAULT_HORIZON, "probit")
    assert fr.coef < 0, "an inverted curve must raise recession probability"
    assert fr.pval < 0.05, f"spread coefficient not significant (p={fr.pval})"
    assert fr.converged


@pytest.mark.parametrize("spread", ["10y3m", "10y2y", "10yffr"])
def test_all_spreads_negative_at_12m(panel, spread):
    assert probit.fit_spread_model(panel, spread, 12, "probit").coef < 0


def test_results_grid_full_sample_all_negative(panel):
    grid = probit.results_grid(panel, links=("probit",), samples=(False,))
    assert (grid["coef"] < 0).all()
    assert grid["auc"].max() > 0.7  # the best spec is a genuinely good classifier


def test_logit_agrees_in_sign(panel):
    p = probit.fit_spread_model(panel, "10y3m", 12, "probit")
    l = probit.fit_spread_model(panel, "10y3m", 12, "logit")
    assert (p.coef < 0) and (l.coef < 0)
