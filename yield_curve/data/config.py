"""Central configuration: paths, credentials, and series/spread definitions.

The single source of truth for *what* data we pull and *how* the spreads are
built. FRED is the source-of-truth; DataBuffet is a cross-check (and optional
real-time/vintage source). DataBuffet mnemonics are confirmed empirically by
``discover_databuffet.py`` and recorded here.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
ENV_PATH = PROJECT_ROOT / ".env"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Load .env once on import (does not override already-set environment variables).
load_dotenv(ENV_PATH)

# --------------------------------------------------------------------------- #
# Credentials
# --------------------------------------------------------------------------- #
def _require(name: str) -> str:
    val = os.environ.get(name, "").strip().strip('"').strip("'")
    if not val:
        raise RuntimeError(
            f"Missing credential {name!r}. Add it to {ENV_PATH} "
            f"(see .env.example). These keys are copied from the GDP Tracker project."
        )
    return val


def fred_api_key() -> str:
    return _require("FRED_API_KEY")


def databuffet_keys() -> tuple[str, str]:
    """Return (access_key, encryption_key) for the DataBuffet OAuth2 client."""
    return _require("DATABUFFET_ACC_KEY"), _require("DATABUFFET_ENC_KEY")


# --------------------------------------------------------------------------- #
# Series definitions
# --------------------------------------------------------------------------- #
# Monthly FRED series that make up the panel. `start` is the FRED coverage start
# (documentation only; we pull the full available history).
FRED_MONTHLY = {
    "GS10": {"desc": "10-Year Treasury Constant Maturity, % p.a.", "start": "1953-04"},
    "TB3MS": {"desc": "3-Month Treasury Bill, Secondary Market, % p.a.", "start": "1934-01"},
    "GS2": {"desc": "2-Year Treasury Constant Maturity, % p.a.", "start": "1976-06"},
    "FEDFUNDS": {"desc": "Effective Federal Funds Rate, % p.a.", "start": "1954-07"},
    "USREC": {"desc": "NBER-based Recession Indicator (0/1), monthly", "start": "1854-12"},
}

# Daily FRED series used only for the live (current) reading.
FRED_DAILY = {
    "DGS10": "10-Year Treasury Constant Maturity (daily)",
    "DTB3": "3-Month Treasury Bill, Secondary Market (daily)",
    "DGS2": "2-Year Treasury Constant Maturity (daily)",
    "DFF": "Effective Federal Funds Rate (daily)",
}

# The three term spreads. `long`/`short` reference monthly FRED ids; `long_d`/
# `short_d` reference the daily ids used for the live reading.
SPREADS = {
    "10y3m": {
        "label": "10Y − 3M T-bill",
        "long": "GS10", "short": "TB3MS",
        "long_d": "DGS10", "short_d": "DTB3",
        "primary": True,
        "note": "Estrella–Mishkin / NY Fed standard. Primary spread.",
    },
    "10y2y": {
        "label": "10Y − 2Y",
        "long": "GS10", "short": "GS2",
        "long_d": "DGS10", "short_d": "DGS2",
        "primary": False,
        "note": "Most-watched market spread; 2Y available only from 1976.",
    },
    "10yffr": {
        "label": "10Y − Fed Funds",
        "long": "GS10", "short": "FEDFUNDS",
        "long_d": "DGS10", "short_d": "DFF",
        "primary": False,
        "note": "Long rate vs the policy rate.",
    },
}

PRIMARY_SPREAD = "10y3m"
HORIZONS = [3, 6, 12, 18, 24]
DEFAULT_HORIZON = 12

# Reconciliation tolerance: flag |FRED - DataBuffet| above this (in percentage
# points). 2 basis points = 0.02 pp.
RECONCILE_TOLERANCE_PP = 0.02

# 3-month bill is quoted discount-basis; convert to bond-equivalent before differencing.
DISCOUNT_BASIS_BILLS = {"DTB3"}   # the only discount-basis daily series in FRED_DAILY
BILL_BE_DAYS = 91                 # 13-week bill, days to maturity

# --------------------------------------------------------------------------- #
# DataBuffet mnemonics
# --------------------------------------------------------------------------- #
# Confirmed empirically (discover_databuffet.py + value-matching against FRED on
# 2026-06-28). Keyed by the FRED id they mirror. These are the monthly INTEREST-
# RATE HISTORY series (IR...M), which match FRED to displayed precision -- NOT the
# FR... quarterly forecast-scenario variables nor JRFEDD (the funds *target*).
DATABUFFET_MNEMONICS: dict[str, str] = {
    "GS10": "IRGT10YM.IUSA",   # Treasury Constant Maturities, 10Y, monthly
    "TB3MS": "IRTB3MM.IUSA",   # T-Bills secondary market, 3M, monthly
    "GS2": "IRGT2YM.IUSA",     # Treasury Constant Maturities, 2Y, monthly
    "FEDFUNDS": "IRFEDM.IUSA",  # Federal funds effective rate, monthly
}

# Business-daily DataBuffet analogs (for the live reading in Phase 5). 3M-daily and
# funds-daily-effective have no clean DataBuffet history match, so FRED is used there.
DATABUFFET_DAILY_MNEMONICS: dict[str, str] = {
    "DGS10": "RIRGT10YD.IUSA",
    "DGS2": "RIRGT2YD.IUSA",
}

# Offline fallback for the GS10 cross-check: a cached DataBuffet pull that already
# lives in the GDP Tracker project (monthly 10Y, from 2000).
GDP_TRACKER_DIR = PROJECT_ROOT.parent / "GDP Tracker"
DATABUFFET_CSV_FALLBACK = {
    "GS10": GDP_TRACKER_DIR / "gs10_databuffet.csv",
}
