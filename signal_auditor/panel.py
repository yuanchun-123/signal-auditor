"""Panel data handling.

The auditor works on a *long* panel with one row per (date, asset):

    date      asset   signal    fwd_ret    [available_at]
    ------    -----   ------    -------    --------------

* ``signal``  -- the value known (or claimed to be known) at ``date``.
* ``fwd_ret`` -- the return earned from ``date`` to ``date + horizon`` periods.
* ``available_at`` (optional) -- the timestamp at which the signal could first
  have been observed (e.g. the SEC filing acceptance time). If present, the
  auditor checks that it never falls after ``date``.
"""

from __future__ import annotations

import pandas as pd

REQUIRED = ("date", "asset", "signal", "fwd_ret")


def validate_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean, sorted copy of ``df`` or raise ``ValueError``."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"panel is missing columns: {missing}")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    if out.duplicated(["date", "asset"]).any():
        raise ValueError("panel has duplicate (date, asset) rows")
    out = out.sort_values(["date", "asset"]).reset_index(drop=True)
    return out


def forward_returns(prices: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Simple forward returns from a wide price table (index=date, columns=asset).

    ``fwd[t] = price[t + horizon] / price[t] - 1``. The last ``horizon`` rows are NaN.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    return prices.shift(-horizon) / prices - 1.0


def to_long(signal: pd.DataFrame, fwd_ret: pd.DataFrame) -> pd.DataFrame:
    """Stack two aligned wide tables (index=date, columns=asset) into a long panel."""
    s = signal.stack(future_stack=True).rename("signal")
    r = fwd_ret.reindex_like(signal).stack(future_stack=True).rename("fwd_ret")
    out = pd.concat([s, r], axis=1).reset_index()
    out.columns = ["date", "asset", "signal", "fwd_ret"]
    return out.dropna(subset=["signal", "fwd_ret"]).reset_index(drop=True)


def wide(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Pivot a long panel column back to wide (index=date, columns=asset)."""
    return df.pivot(index="date", columns="asset", values=col).sort_index()


def lag_signal(df: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """Shift each asset's signal back by ``periods`` dates (use older information)."""
    sig = wide(df, "signal").shift(periods)
    ret = wide(df, "fwd_ret")
    return to_long(sig, ret)


def cross_section_sizes(df: pd.DataFrame) -> pd.Series:
    valid = df.dropna(subset=["signal", "fwd_ret"])
    return valid.groupby("date").size()
