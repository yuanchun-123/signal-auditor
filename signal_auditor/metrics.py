"""Core statistics. Every function here is deterministic and unit-tested."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

EULER_GAMMA = 0.5772156649015329


# --------------------------------------------------------------------------- #
# Information coefficient
# --------------------------------------------------------------------------- #
def ic_series(df: pd.DataFrame, method: str = "spearman", min_assets: int = 5) -> pd.Series:
    """Cross-sectional correlation between ``signal`` and ``fwd_ret`` on each date.

    ``method="spearman"`` gives the rank IC (robust to outliers, the usual default);
    ``"pearson"`` gives the plain IC.
    """
    d = df.dropna(subset=["signal", "fwd_ret"])
    if method == "spearman":
        x = d.groupby("date")["signal"].rank()
        y = d.groupby("date")["fwd_ret"].rank()
    elif method == "pearson":
        x, y = d["signal"], d["fwd_ret"]
    else:
        raise ValueError("method must be 'spearman' or 'pearson'")

    g = d["date"]
    xc = x - x.groupby(g).transform("mean")
    yc = y - y.groupby(g).transform("mean")
    num = (xc * yc).groupby(g).sum()
    den = np.sqrt((xc**2).groupby(g).sum() * (yc**2).groupby(g).sum())
    n = g.groupby(g).size()
    ic = (num / den).where((n >= min_assets) & (den > 0))
    return ic.dropna().rename("ic")


# --------------------------------------------------------------------------- #
# Inference with autocorrelated / overlapping observations
# --------------------------------------------------------------------------- #
def newey_west_se(x: np.ndarray | pd.Series, lags: int) -> float:
    """Newey-West (Bartlett kernel) standard error of the sample mean of ``x``.

    With ``lags=0`` this equals the ordinary ``std(ddof=0) / sqrt(n)``.
    Use ``lags >= horizon - 1`` when the series is built from overlapping
    ``horizon``-period returns.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n < 2:
        return float("nan")
    lags = max(0, min(int(lags), n - 1))
    e = x - x.mean()
    s = e @ e / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        s += 2.0 * w * (e[k:] @ e[:-k]) / n
    return math.sqrt(max(s, 0.0) / n)


def newey_west_t(x: np.ndarray | pd.Series, lags: int) -> float:
    x = np.asarray(x, dtype=float)
    se = newey_west_se(x, lags)
    return float(np.nanmean(x) / se) if se and se > 0 else float("nan")


def default_nw_lags(n: int, horizon: int) -> int:
    """Overlap-aware lag choice: at least ``horizon-1``, plus the usual n^(1/4) rule."""
    return max(horizon - 1, int(math.floor(4 * (n / 100.0) ** (2.0 / 9.0))))


# --------------------------------------------------------------------------- #
# Quantile portfolios
# --------------------------------------------------------------------------- #
def quantile_labels(df: pd.DataFrame, n_quantiles: int = 5) -> pd.Series:
    """Quantile bucket (1 = lowest signal, ``n_quantiles`` = highest) within each date."""
    pct = df.groupby("date")["signal"].rank(pct=True, method="first")
    return np.ceil(pct * n_quantiles).clip(1, n_quantiles).astype(int)


def quantile_returns(df: pd.DataFrame, n_quantiles: int = 5) -> pd.DataFrame:
    """Equal-weighted mean ``fwd_ret`` per (date, quantile). Wide: index=date, columns=q."""
    d = df.dropna(subset=["signal", "fwd_ret"]).copy()
    d["q"] = quantile_labels(d, n_quantiles)
    return d.groupby(["date", "q"])["fwd_ret"].mean().unstack("q").sort_index()


def long_short_weights(df: pd.DataFrame, n_quantiles: int = 5) -> pd.DataFrame:
    """Dollar-neutral weights: +1 spread over the top quantile, -1 over the bottom.

    Gross exposure is 2 (1 long + 1 short). Wide: index=date, columns=asset.
    """
    d = df.dropna(subset=["signal"]).copy()
    d["q"] = quantile_labels(d, n_quantiles)
    top = (d["q"] == n_quantiles).astype(float)
    bot = (d["q"] == 1).astype(float)
    d["w"] = top / top.groupby(d["date"]).transform("sum") - bot / bot.groupby(
        d["date"]
    ).transform("sum")
    return d.pivot(index="date", columns="asset", values="w").fillna(0.0).sort_index()


def turnover(weights: pd.DataFrame) -> pd.Series:
    """One-way turnover per rebalance: sum |w_t - w_{t-1}|. First date counts the entry."""
    return weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1).iloc[0])


def sharpe(x: pd.Series | np.ndarray, periods_per_year: float) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    sd = x.std(ddof=1) if x.size > 1 else float("nan")
    return float(x.mean() / sd * math.sqrt(periods_per_year)) if sd and sd > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Multiple testing
# --------------------------------------------------------------------------- #
def expected_max_sharpe(n_trials: int, sr_std: float) -> float:
    """Expected maximum Sharpe ratio among ``n_trials`` independent zero-skill trials.

    Bailey & Lopez de Prado (2014), eq. for SR_0. ``sr_std`` is the cross-trial
    standard deviation of the (per-period) Sharpe estimates.
    """
    if n_trials <= 1:
        return 0.0
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return sr_std * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)


def deflated_sharpe_ratio(
    returns: pd.Series | np.ndarray, n_trials: int, sr_std: float | None = None
) -> tuple[float, float, float]:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

    Returns ``(dsr, sr_hat, sr0)`` with Sharpe ratios in *per-period* units.
    ``dsr`` is the probability that the true Sharpe exceeds the best Sharpe one
    would expect from ``n_trials`` skill-less strategies. If ``sr_std`` is not
    supplied we use the null approximation ``1/sqrt(T)``.
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    t = r.size
    if t < 10 or r.std(ddof=1) == 0:
        return float("nan"), float("nan"), float("nan")
    sr = r.mean() / r.std(ddof=1)
    skew = stats.skew(r)
    kurt = stats.kurtosis(r, fisher=False)
    if sr_std is None:
        sr_std = 1.0 / math.sqrt(t)
    sr0 = expected_max_sharpe(n_trials, sr_std)
    denom = math.sqrt(max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2, 1e-12))
    dsr = stats.norm.cdf((sr - sr0) * math.sqrt(t - 1) / denom)
    return float(dsr), float(sr), float(sr0)


def bonferroni_t_threshold(n_trials: int, alpha: float = 0.05) -> float:
    """Two-sided |t| needed for family-wise significance after ``n_trials`` tests."""
    return float(stats.norm.ppf(1.0 - alpha / (2.0 * max(n_trials, 1))))
