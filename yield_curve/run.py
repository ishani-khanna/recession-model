"""Command-line entrypoint (Phase 6).

    python -m yield_curve.run report   [--refresh] [--spread 10y3m] [--horizon 12]
    python -m yield_curve.run data     [--refresh]      # build panel + reconcile sources
    python -m yield_curve.run model                     # full results grid
    python -m yield_curve.run validate                  # out-of-sample + case study
    python -m yield_curve.run all      [--refresh]      # everything, end to end

With no command, defaults to `report`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import warnings

import numpy as np
from scipy.stats import norm

from . import reporting
from .data import build_dataset, config
from .models import probit, validation


# Flat "interactive dashboard" pipeline (gauge + scenario lab + conditions matrix).
# Lives as standalone scripts at the repo root; we regenerate output/index.html after
# the report so a single command refreshes BOTH the report and the interactive dashboard.
_DASHBOARD_STEPS = [
    "build_two_clock_data.py",     # landing page: three-signal rule  -> docs/index.html
    "build_dataset.py", "build_features.py", "build_verdict.py",
    "build_dashboard_data.py",     # interactive explore page (gauge/lab/matrix) -> docs/explore.html
    "build_validation_data.py",    # validation & methodology page (8 sections + charts) -> docs/validation.html
]


def _regenerate_dashboard() -> None:
    root = config.PROJECT_ROOT
    print("\n=== Regenerating interactive dashboard (output/index.html) ===")
    for script in _DASHBOARD_STEPS:
        try:
            subprocess.run([sys.executable, script], cwd=root, check=True,
                           capture_output=True, text=True)
            print(f"  ✓ {script}")
        except subprocess.CalledProcessError as e:  # never let it break the report
            print(f"  ✗ {script} failed: {e.stderr.strip().splitlines()[-1] if e.stderr else e}")
            return
    print(f"  dashboard -> {root / 'output' / 'index.html'}")

warnings.filterwarnings("ignore")


def _print_reading(panel, spread_key: str, horizon: int) -> None:
    curve = reporting.fetch_current_curve()
    spreads = reporting.current_spreads_daily(curve)
    x, as_of = spreads[spread_key]
    fr = probit.fit_spread_model(panel, spread_key, horizon, "probit")
    prob = float(norm.cdf(fr.intercept + fr.coef * x)) * 100
    label = config.SPREADS[spread_key]["label"]
    print(f"\n>>> Current reading ({as_of.date() if as_of else 'n/a'})")
    print(f"    {label} spread = {x:+.2f} pp"
          f"  ->  P(recession in {horizon}m) = {prob:.0f}%"
          f"   [{'INVERTED' if x < 0 else 'positively sloped'}]")


def cmd_data(panel, args) -> None:
    print("=== Coverage ===")
    print(build_dataset.coverage(panel).to_string())
    print("\n=== FRED vs DataBuffet reconciliation (2bp tol) ===")
    print(build_dataset.reconcile(panel, refresh=args.refresh).to_string())


def cmd_model(panel, args) -> None:
    grid = probit.results_grid(panel)
    grid.to_csv(config.OUTPUTS_DIR / "results_grid.csv", index=False)
    headline = grid[(grid.link == "probit") & (grid["sample"] == "full")]
    print(headline.to_string(index=False))


def cmd_validate(panel, args) -> None:
    print(validation.oos_grid(panel).to_string(index=False))
    out = validation.write_report(panel)
    print(f"\nvalidation report -> {out}")


def cmd_report(panel, args) -> None:
    out = reporting.write_report(panel)
    print(f"report + charts -> {out}")
    _print_reading(panel, args.spread, args.horizon)
    _regenerate_dashboard()


def cmd_all(panel, args) -> None:
    cmd_data(panel, args)
    print("\n" + "=" * 60)
    cmd_model(panel, args)
    print("\n" + "=" * 60)
    cmd_validate(panel, args)
    print("\n" + "=" * 60)
    cmd_report(panel, args)


def main() -> None:
    ap = argparse.ArgumentParser(prog="yield_curve.run", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", default="report",
                    choices=["report", "data", "model", "validate", "all"])
    ap.add_argument("--refresh", action="store_true", help="ignore today's cache and re-pull")
    ap.add_argument("--spread", default=config.PRIMARY_SPREAD, choices=list(config.SPREADS))
    ap.add_argument("--horizon", type=int, default=config.DEFAULT_HORIZON, choices=config.HORIZONS)
    args = ap.parse_args()

    panel = build_dataset.build_panel(refresh=args.refresh)
    {"data": cmd_data, "model": cmd_model, "validate": cmd_validate,
     "report": cmd_report, "all": cmd_all}[args.command](panel, args)


if __name__ == "__main__":
    main()
