"""DataBuffet (Moody's Analytics) client.

A thin wrapper over the GDP Tracker's proven ``dbapi.DataBuffetAPI`` (copied
verbatim into this package). That class handles the OAuth2 client_credentials
flow, retries, and the freq/sentinel parsing; here we just load credentials from
.env, return clean history-only Series, and expose search for mnemonic discovery.
"""

from __future__ import annotations

import pandas as pd

from . import config
from .dbapi import DataBuffetAPI


class DataBuffetClient:
    def __init__(self) -> None:
        acc, enc = config.databuffet_keys()
        # OAuth mode is the dbapi default and what the GDP Tracker uses.
        self._api = DataBuffetAPI(acc, enc)

    # ----------------------------------------------------------------- #
    def health(self) -> dict:
        return self._api.health()

    def search(self, query: str, rows: int = 30):
        """Return a list of {mnemonic, description, ...} search hits."""
        res = self._api.search(query, rows=rows)
        if isinstance(res, dict):
            # The API nests hits under 'data' (or returns them directly).
            return res.get("data", res.get("results", res))
        return res

    def get_series(
        self,
        mnemonic: str,
        start: str | None = None,
        end: str | None = None,
        history_only: bool = True,
    ) -> pd.Series:
        """Fetch a single series as a clean float Series indexed by date.

        ``history_only`` drops the forecast tail using the series' last_history
        attribute (DataBuffet series often extend into forecast), matching the
        GDP Tracker's fetch_databuffet_series behavior.
        """
        s = self._api.get_series(mnemonic, start=start, end=end)
        last_history = getattr(s, "last_history", None)
        if history_only and last_history is not None:
            s = s[s.index <= last_history]
        s = s.dropna()
        out = pd.Series(s.values, index=pd.to_datetime(s.index), name=mnemonic)
        return out
