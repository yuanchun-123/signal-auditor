"""Synthetic panels with a *known* answer, used to test the auditor itself.

If the auditor cannot tell a planted real signal from a leaky one or from the
luckiest of many random ones, its verdicts on real data mean nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from . import metrics as M


def _ar1(rng: np.random.Generator, t: int, n: int, rho: float) -> np.ndarray:
    x = np.empty((t, n))
    x[0] = rng.standard_normal(n)
    s = np.sqrt(1 - rho**2)
    for i in range(1, t):
        x[i] = rho * x[i - 1] + s * rng.standard_normal(n)
    return x


def _long(dates, assets, signal: np.ndarray, ret: np.ndarray, alive: np.ndarray) -> pd.DataFrame:
    t, n = signal.shape
    out = pd.DataFrame({
        "date": np.repeat(dates, n),
        "asset": np.tile(assets, t),
        "signal": signal.ravel(),
        "fwd_ret": ret.ravel(),
    })
    return out[alive.ravel()].reset_index(drop=True)


def _rank_ic_t(signal: np.ndarray, ret: np.ndarray) -> float:
    rs = rankdata(signal, axis=1)
    rr = rankdata(ret, axis=1)
    rs -= rs.mean(1, keepdims=True)
    rr -= rr.mean(1, keepdims=True)
    ic = (rs * rr).sum(1) / np.sqrt((rs**2).sum(1) * (rr**2).sum(1))
    return M.newey_west_t(ic, 0)


def make_synthetic(n_dates: int = 240, n_assets: int = 300, ic: float = 0.03,
                   n_random: int = 200, seed: int = 7) -> dict[str, pd.DataFrame]:
    """Return panels for four planted cases on a monthly calendar (horizon = 1).

    * ``true``   -- persistent signal with a small genuine IC; timestamps provided.
    * ``leaky``  -- signal contaminated with the very return it is meant to predict.
    * ``mined``  -- pure noise: the best of ``n_random`` random signals by in-sample t-stat.
    * ``stale_timestamp`` -- the true signal, but 15% of values published after use.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2005-01-31", periods=n_dates, freq="ME")
    assets = np.array([f"A{i:04d}" for i in range(n_assets)])

    true_sig = _ar1(rng, n_dates, n_assets, rho=0.9)
    market = 0.04 * rng.standard_normal((n_dates, 1))
    noise = 0.08 * rng.standard_normal((n_dates, n_assets))
    # scale so corr(signal, idiosyncratic return) ~= ic
    beta = ic * 0.08 / np.sqrt(1 - ic**2)
    ret = market + beta * true_sig + noise

    leaky_sig = ret + 0.25 * rng.standard_normal((n_dates, n_assets))

    best_t, mined_sig = 0.0, None
    for _ in range(n_random):
        cand = _ar1(rng, n_dates, n_assets, rho=0.5)
        t = _rank_ic_t(cand, ret)
        if abs(t) > abs(best_t):
            best_t, mined_sig = t, np.sign(t) * cand

    # ~20% of assets delist at a random date, so the universe is not survivor-only
    alive = np.ones((n_dates, n_assets), dtype=bool)
    delist = rng.random(n_assets) < 0.2
    for j in np.flatnonzero(delist):
        alive[rng.integers(n_dates // 4, n_dates):, j] = False

    true_df = _long(dates, assets, true_sig, ret, alive)
    lag_days = rng.integers(1, 30, size=len(true_df))
    true_df["available_at"] = true_df["date"] - pd.to_timedelta(lag_days, unit="D")

    stale = true_df.copy()
    late = rng.random(len(stale)) < 0.15
    stale.loc[late, "available_at"] = stale.loc[late, "date"] + pd.Timedelta(days=10)

    return {
        "true": true_df,
        "leaky": _long(dates, assets, leaky_sig, ret, alive),
        "mined": _long(dates, assets, mined_sig, ret, alive),
        "stale_timestamp": stale,
    }
