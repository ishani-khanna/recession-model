"""Thin FRED REST client.

Mirrors the proven pattern in the GDP Tracker project (gdp_nowcast_alfred.py /
gdp_scheduler.py): hit api.stlouisfed.org with file_type=json, pull the full
series in one call, do all transformation client-side, coerce FRED's "." missing
sentinel to NaN, and use a status-aware retry. ALFRED real-time vintages are
supported (realtime_start/realtime_end) for point-in-time out-of-sample work.
"""

from __future__ import annotations

import time

import pandas as pd
import requests

from . import config

BASE = "https://api.stlouisfed.org/fred"
OBS_URL = f"{BASE}/series/observations"

# ALFRED far-future sentinel; mapped to a pandas-safe date to avoid overflow.
_RT_SENTINEL = "9999-12-31"
_RT_SAFE = "2099-12-31"


class FredError(RuntimeError):
    pass


class FredClient:
    def __init__(self, api_key: str | None = None, timeout: int = 60, pause: float = 0.2):
        self.api_key = api_key or config.fred_api_key()
        self.timeout = timeout
        self.pause = pause  # politeness sleep between successful calls
        self._session = requests.Session()

    # ----------------------------------------------------------------- #
    # Low-level request with status-aware retry
    # ----------------------------------------------------------------- #
    def _get(self, params: dict, max_retries: int = 4) -> dict:
        params = {**params, "api_key": self.api_key, "file_type": "json"}
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                r = self._session.get(OBS_URL, params=params, timeout=self.timeout)
                if r.status_code == 429:  # rate limited
                    time.sleep(60)
                    continue
                if 400 <= r.status_code < 500:  # client error: do not retry
                    raise FredError(
                        f"FRED {r.status_code} for {params.get('series_id')}: {r.text[:300]}"
                    )
                r.raise_for_status()  # 5xx -> raise, caught below for backoff
                return r.json()
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
                last_exc = e
                if attempt < max_retries - 1:
                    time.sleep(2 ** (attempt + 1))
                else:
                    raise FredError(f"FRED request failed for {params.get('series_id')}: {e}") from e
        raise FredError(f"FRED request exhausted retries: {last_exc}")

    # ----------------------------------------------------------------- #
    # Public API
    # ----------------------------------------------------------------- #
    def get_series(self, series_id: str, observation_start: str | None = None) -> pd.Series:
        """Latest-vintage observations as a float Series indexed by reference date."""
        params = {"series_id": series_id}
        if observation_start:
            params["observation_start"] = observation_start
        data = self._get(params)
        obs = data.get("observations", [])
        if not obs:
            return pd.Series(dtype=float, name=series_id)
        df = pd.DataFrame(obs)
        idx = pd.to_datetime(df["date"])
        val = pd.to_numeric(df["value"], errors="coerce")  # "." -> NaN
        s = pd.Series(val.values, index=idx, name=series_id).dropna()
        if self.pause:
            time.sleep(self.pause)
        return s

    def get_alfred_vintages(
        self, series_id: str, realtime_start: str = "1990-01-01"
    ) -> pd.DataFrame:
        """Full revision history: one row per (reference date, vintage).

        Columns: date, value, realtime_start, realtime_end. Reconstruct a
        point-in-time snapshot with ``vintage_snapshot``.
        """
        params = {
            "series_id": series_id,
            "realtime_start": realtime_start,
            "realtime_end": _RT_SENTINEL,
        }
        data = self._get(params)
        obs = data.get("observations", [])
        if not obs:
            return pd.DataFrame(columns=["date", "value", "realtime_start", "realtime_end"])
        df = pd.DataFrame(obs)
        for col in ("date", "realtime_start", "realtime_end"):
            df[col] = pd.to_datetime(df[col].str.replace(_RT_SENTINEL, _RT_SAFE, regex=False))
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df[["date", "value", "realtime_start", "realtime_end"]]
        df = df.sort_values(["date", "realtime_start"]).reset_index(drop=True)
        if self.pause:
            time.sleep(self.pause)
        return df


def vintage_snapshot(vintage_df: pd.DataFrame, as_of: str | pd.Timestamp) -> pd.Series:
    """The series as it would have been known on ``as_of`` (no look-ahead)."""
    if vintage_df is None or vintage_df.empty:
        return pd.Series(dtype=float)
    as_of = pd.Timestamp(as_of)
    mask = (vintage_df["realtime_start"] <= as_of) & (vintage_df["realtime_end"] >= as_of)
    valid = vintage_df.loc[mask]
    if valid.empty:
        return pd.Series(dtype=float)
    latest = valid.sort_values(["date", "realtime_start"]).groupby("date").last()
    return latest["value"].dropna()
