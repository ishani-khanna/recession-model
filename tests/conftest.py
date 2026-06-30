"""Shared pytest fixtures and markers."""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from yield_curve.data import config

warnings.filterwarnings("ignore")


def pytest_configure(config):  # noqa: D401
    config.addinivalue_line(
        "markers", "integration: hits live FRED/DataBuffet APIs; skips offline"
    )


@pytest.fixture(scope="session")
def panel() -> pd.DataFrame:
    """The cached monthly panel; skip cleanly if it hasn't been built yet."""
    p = config.CACHE_DIR / "panel_monthly.parquet"
    if not p.exists():
        pytest.skip("panel not built; run `python -m yield_curve.run data` first")
    return pd.read_parquet(p)
