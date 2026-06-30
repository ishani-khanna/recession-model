# Yield-Curve Recession Indicator

A reproducible, current, full-history US recession indicator: a probit on the Treasury
yield-curve spread, drawing live data from **FRED** (source of truth) with an optional
**Moody's DataBuffet** cross-check, validated out-of-sample — including the deep 2022–2023
inversion that produced no recession.

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env          # then paste your keys into .env (see below)
python -m yield_curve.run all # build → model → validate → report
```

Outputs land in `outputs/` (`report.md`, `validation.md`, and PNG charts). Raw FRED pulls
are cached to `data/cache/` per series + date, so reruns are offline and fast.

## Credentials (`.env`)

`.env` is git-ignored — never commit it. Copy `.env.example` to `.env` and fill in:

- `FRED_API_KEY` — **required**. Free from https://fredaccount.stlouisfed.org/apikeys
- `DATABUFFET_ACC_KEY` / `DATABUFFET_ENC_KEY` — **optional** (Moody's Analytics, two-key
  OAuth). Without them the FRED panel still builds; the FRED-vs-DataBuffet reconciliation
  simply reports the keys as missing. These names match the GDP Tracker project.

## CLI

```bash
python -m yield_curve.run report   [--refresh] [--spread 10y3m] [--horizon 12]
python -m yield_curve.run data     [--refresh]   # build panel + reconcile FRED vs DataBuffet
python -m yield_curve.run model                  # full spread × horizon results grid
python -m yield_curve.run validate               # expanding-window OOS + 2022-23 case study
python -m yield_curve.run all      [--refresh]   # everything, end to end
```

`--refresh` ignores today's cache and re-pulls from FRED.

## What it does

- **Three spreads, side by side:** 10Y−3M (primary, Estrella–Mishkin / NY Fed standard),
  10Y−2Y, 10Y−Fed Funds. Probit `P(recession in t+h) = Φ(b₀ + b₁·spread)`, with a logit
  robustness variant and an exclude-2020 variant. HAC (Newey–West) standard errors.
- **Horizons** 3/6/12/18/24 months (default 12).
- **No look-ahead:** the target is the recession flag led by `h` months, built with an
  explicit, tested shift (`tests/test_target.py`).
- **Out-of-sample validation:** expanding-window estimation collecting pseudo-real-time
  predictions; reports OOS AUC and Brier beside the in-sample AUC (the gap is the headline).
- **Current reading:** pulls the latest *daily* curve for a live spread and probability.

## Interpretation caveats

- A high probability is a **probability, not a certainty.** The 2022–23 inversion drove the
  model near ~77% with no recession (so far) — term-premium compression is the leading
  explanation. See `outputs/validation.md`.
- 2020 (COVID) was an exogenous shock the curve did not truly predict; an exclude-2020
  variant is available and barely changes the fit.
- 2Y history starts in 1976, so 10Y−2Y rests on fewer observations than the other spreads.

## Tests

```bash
pytest -q -m "not integration"   # offline unit tests
pytest -q -m integration         # live FRED/DataBuffet checks (needs network + keys)
```

A weekly GitHub Action (`.github/workflows/refresh.yml`) re-pulls, recomputes, runs the
tests, and uploads `outputs/`. Store `FRED_API_KEY` (and optionally the DataBuffet keys)
as repository secrets — never in code.
