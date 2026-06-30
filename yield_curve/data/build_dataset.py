"""Assemble the tidy monthly panel and reconcile FRED against DataBuffet.

Outputs a month-indexed panel of yields, the three term spreads, and the USREC
recession flag. FRED is the source-of-truth; DataBuffet is pulled for the same
yield series and compared, flagging divergences above the 2bp tolerance.

    python -m yield_curve.data.build_dataset [--refresh]
"""

from __future__ import annotations

import argparse
import datetime as _dt

import numpy as np
import pandas as pd

from . import config
from .databuffet_client import DataBuffetClient
from .fred_client import FredClient


# --------------------------------------------------------------------------- #
# Caching helpers (keyed by series + as-of date -> reruns are offline & fast)
# --------------------------------------------------------------------------- #
def _today() -> str:
    return _dt.date.today().isoformat()


def _cache_path(name: str) -> "config.Path":
    return config.CACHE_DIR / f"{name}_{_today()}.parquet"


def _save_series(s: pd.Series, name: str) -> None:
    s.to_frame("value").to_parquet(_cache_path(name))


def _load_series(name: str) -> pd.Series | None:
    p = _cache_path(name)
    if p.exists():
        return pd.read_parquet(p)["value"]
    return None


def _to_month_start(s: pd.Series) -> pd.Series:
    """Normalize any monthly index to first-of-month timestamps so series align."""
    s = s.copy()
    s.index = pd.to_datetime(s.index).to_period("M").to_timestamp()
    return s[~s.index.duplicated(keep="last")].sort_index()


# --------------------------------------------------------------------------- #
# FRED
# --------------------------------------------------------------------------- #
def fetch_fred_monthly(refresh: bool = False) -> dict[str, pd.Series]:
    client = FredClient()
    out: dict[str, pd.Series] = {}
    for sid in config.FRED_MONTHLY:
        cached = None if refresh else _load_series(f"fred_{sid}")
        if cached is not None:
            out[sid] = cached
            continue
        s = client.get_series(sid)
        s = _to_month_start(s)
        _save_series(s, f"fred_{sid}")
        out[sid] = s
    return out


def build_panel(refresh: bool = False) -> pd.DataFrame:
    fred = fetch_fred_monthly(refresh=refresh)
    panel = pd.DataFrame(
        {
            "gs10": fred["GS10"],
            "tb3ms": fred["TB3MS"],
            "gs2": fred["GS2"],
            "fedfunds": fred["FEDFUNDS"],
            "usrec": fred["USREC"],
        }
    ).sort_index()

    # Three term spreads (in percentage points).
    panel["spread_10y3m"] = panel["gs10"] - panel["tb3ms"]
    panel["spread_10y2y"] = panel["gs10"] - panel["gs2"]
    panel["spread_10yffr"] = panel["gs10"] - panel["fedfunds"]

    # Trim to where at least the primary spread exists, but keep full history.
    panel = panel.loc[panel[["gs10", "tb3ms"]].dropna().index.min():]
    panel.index.name = "date"

    panel.to_parquet(config.CACHE_DIR / "panel_monthly.parquet")
    panel.to_csv(config.CACHE_DIR / "panel_monthly.csv")
    return panel


# --------------------------------------------------------------------------- #
# DataBuffet (cross-check)
# --------------------------------------------------------------------------- #
def _load_databuffet_csv_fallback(fred_id: str) -> pd.Series | None:
    path = config.DATABUFFET_CSV_FALLBACK.get(fred_id)
    if path is None or not path.exists():
        return None
    df = pd.read_csv(path, index_col=0)
    s = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    s.index = pd.to_datetime(df.index)
    return _to_month_start(s.dropna())


def fetch_databuffet_yields(refresh: bool = False) -> dict[str, dict]:
    """Pull each yield series from DataBuffet where a mnemonic is confirmed.

    Returns {fred_id: {"series": Series, "source": "api"|"csv", "label": str}}.
    """
    out: dict[str, dict] = {}
    client: DataBuffetClient | None = None
    for fred_id in ("GS10", "TB3MS", "GS2", "FEDFUNDS"):
        mnemonic = config.DATABUFFET_MNEMONICS.get(fred_id)
        if mnemonic:
            cached = None if refresh else _load_series(f"db_{fred_id}")
            if cached is not None:
                out[fred_id] = {"series": cached, "source": "api", "label": mnemonic}
                continue
            try:
                if client is None:
                    client = DataBuffetClient()
                s = _to_month_start(client.get_series(mnemonic).dropna())
                _save_series(s, f"db_{fred_id}")
                out[fred_id] = {"series": s, "source": "api", "label": mnemonic}
                continue
            except Exception as e:  # noqa: BLE001
                out[fred_id] = {"series": None, "source": "error", "label": f"{mnemonic}: {e}"}
                continue
        # Fall back to a cached DataBuffet CSV if we have one (GS10 only today).
        s = _load_databuffet_csv_fallback(fred_id)
        if s is not None:
            out[fred_id] = {"series": s, "source": "csv", "label": str(config.DATABUFFET_CSV_FALLBACK[fred_id].name)}
        else:
            out[fred_id] = {"series": None, "source": "missing", "label": "no confirmed mnemonic"}
    return out


def compare_series(fred_s: pd.Series, db_s: pd.Series,
                   tol_pp: float = config.RECONCILE_TOLERANCE_PP) -> dict:
    """Pure comparison of two series on their overlap. Diffs reported in basis points."""
    joined = pd.concat({"fred": fred_s.dropna(), "db": db_s.dropna()}, axis=1).dropna()
    if joined.empty:
        return {"n_overlap": 0, "overlap_start": None, "overlap_end": None,
                "mean_abs_diff_bp": np.nan, "max_abs_diff_bp": np.nan,
                "n_gt_tol": 0, "worst_date": None}
    diff_bp = (joined["fred"] - joined["db"]).abs() * 100.0  # pp -> bp
    return {
        "n_overlap": int(len(joined)),
        "overlap_start": joined.index.min(),
        "overlap_end": joined.index.max(),
        "mean_abs_diff_bp": round(float(diff_bp.mean()), 3),
        "max_abs_diff_bp": round(float(diff_bp.max()), 3),
        "n_gt_tol": int((diff_bp > tol_pp * 100.0).sum()),
        "worst_date": diff_bp.idxmax(),
    }


def reconcile(panel: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    fred_cols = {"GS10": "gs10", "TB3MS": "tb3ms", "GS2": "gs2", "FEDFUNDS": "fedfunds"}
    db = fetch_databuffet_yields(refresh=refresh)

    rows = []
    for fred_id, col in fred_cols.items():
        info = db.get(fred_id, {})
        dbs = info.get("series")
        if dbs is None:
            rows.append(
                {"series": fred_id, "db_source": info.get("source", "missing"),
                 "db_label": info.get("label", ""), "n_overlap": 0,
                 "overlap_start": None, "overlap_end": None,
                 "mean_abs_diff_bp": np.nan, "max_abs_diff_bp": np.nan,
                 "n_gt_2bp": np.nan, "worst_date": None}
            )
            continue
        c = compare_series(panel[col], dbs)
        rows.append(
            {"series": fred_id, "db_source": info.get("source"), "db_label": info.get("label"),
             "n_overlap": c["n_overlap"],
             "overlap_start": c["overlap_start"].date() if c["overlap_start"] is not None else None,
             "overlap_end": c["overlap_end"].date() if c["overlap_end"] is not None else None,
             "mean_abs_diff_bp": c["mean_abs_diff_bp"], "max_abs_diff_bp": c["max_abs_diff_bp"],
             "n_gt_2bp": c["n_gt_tol"],
             "worst_date": c["worst_date"].date() if c["worst_date"] is not None else None}
        )
    return pd.DataFrame(rows).set_index("series")


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def coverage(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ["gs10", "tb3ms", "gs2", "fedfunds", "usrec",
                "spread_10y3m", "spread_10y2y", "spread_10yffr"]:
        s = panel[col].dropna()
        rows.append({"column": col, "n": int(len(s)),
                     "first": s.index.min().date() if len(s) else None,
                     "last": s.index.max().date() if len(s) else None})
    return pd.DataFrame(rows).set_index("column")


def main() -> None:
    import warnings
    # dbapi.py (reused verbatim) uses legacy pandas freq aliases ('M','Q-DEC').
    warnings.filterwarnings("ignore", category=FutureWarning)

    ap = argparse.ArgumentParser(description="Build monthly panel and reconcile sources.")
    ap.add_argument("--refresh", action="store_true", help="ignore today's cache and re-pull")
    args = ap.parse_args()

    print("Building monthly panel from FRED ...")
    panel = build_panel(refresh=args.refresh)
    print(f"  panel shape: {panel.shape}, saved to {config.CACHE_DIR/'panel_monthly.parquet'}")

    print("\n=== Coverage (full-sample first/last dates) ===")
    print(coverage(panel).to_string())

    print("\n=== FRED vs DataBuffet reconciliation (tolerance = 2bp) ===")
    rec = reconcile(panel, refresh=args.refresh)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(rec.to_string())
    rec.to_csv(config.OUTPUTS_DIR / "reconciliation.csv")
    print(f"\n  reconciliation saved to {config.OUTPUTS_DIR/'reconciliation.csv'}")


if __name__ == "__main__":
    main()
