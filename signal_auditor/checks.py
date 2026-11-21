"""Individual audit checks.

Each check returns a :class:`CheckResult` with a status:

* ``PASS`` -- no problem detected by this check.
* ``WARN`` -- something a careful researcher should look at before trusting the result.
* ``FAIL`` -- the evidence does not support the claim this check is about.

Checks are deliberately conservative heuristics, not proofs. A PASS never means
"this is alpha"; it means "this particular way of fooling yourself was not detected".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import metrics as M
from .panel import cross_section_sizes, lag_signal

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


@dataclass
class CheckResult:
    name: str
    status: str
    summary: str
    details: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# 1. Data integrity
# --------------------------------------------------------------------------- #
def check_timestamps(df: pd.DataFrame) -> CheckResult:
    """Signal must be observable no later than the date it is used."""
    name = "Point-in-time timestamps"
    if "available_at" not in df.columns:
        return CheckResult(
            name,
            WARN,
            "No `available_at` column: cannot verify the signal was observable on its date. "
            "Supply the filing / publication timestamp to enable this check.",
        )
    avail = pd.to_datetime(df["available_at"])
    late = avail > df["date"]
    n_late = int(late.sum())
    if n_late == 0:
        return CheckResult(name, PASS, "Every signal value was available on or before its date.")
    share = n_late / len(df)
    return CheckResult(
        name,
        FAIL,
        f"{n_late:,} rows ({share:.1%}) use a signal that became available *after* the "
        "trade date -- look-ahead bias.",
        {"n_late": n_late, "share_late": share,
         "examples": df.loc[late, ["date", "asset", "available_at"]].head(5).astype(str).to_dict("records")},
    )


def check_coverage(df: pd.DataFrame, min_assets: int = 20) -> CheckResult:
    name = "Cross-sectional coverage"
    sizes = cross_section_sizes(df)
    if sizes.empty:
        return CheckResult(name, FAIL, "No date has both a signal and a forward return.")
    small = int((sizes < min_assets).sum())
    det = {"dates": int(sizes.size), "median_assets": float(sizes.median()),
           "min_assets": int(sizes.min()), "dates_below_min": small}
    if small / sizes.size > 0.2:
        return CheckResult(name, WARN, f"{small} of {sizes.size} dates have fewer than "
                           f"{min_assets} assets; quantile portfolios will be noisy.", det)
    return CheckResult(name, PASS, f"{sizes.size} dates, median {sizes.median():.0f} assets per date.", det)


def check_universe(df: pd.DataFrame, universe_note: str | None = None) -> CheckResult:
    """Flag survivorship bias: a universe in which no asset ever disappears is suspicious."""
    name = "Universe / survivorship"
    last_seen = df.groupby("asset")["date"].max()
    final = df["date"].max()
    exits = int((last_seen < final).sum())
    det = {"assets": int(last_seen.size), "assets_that_exit": exits}
    if universe_note:
        det["note"] = universe_note
    if exits == 0 and last_seen.size >= 20:
        return CheckResult(
            name, WARN,
            "No asset ever leaves the sample. If the universe is today's constituents, "
            "results carry survivorship bias (delisted losers are missing).", det)
    return CheckResult(name, PASS, f"{exits} of {last_seen.size} assets exit before the end of the sample.", det)


# --------------------------------------------------------------------------- #
# 2. Leakage
# --------------------------------------------------------------------------- #
def check_lag_decay(df: pd.DataFrame, horizon: int, min_assets: int = 5) -> CheckResult:
    """Delay the signal by one extra period and see what survives.

    Genuine slow-moving information decays gradually. A signal whose IC collapses
    the moment it is one period stale -- while being implausibly strong -- usually
    contains information from the return window itself.
    """
    name = "Leakage: one-period-delay test"
    ic0 = M.ic_series(df, min_assets=min_assets).mean()
    ic1 = M.ic_series(lag_signal(df, 1), min_assets=min_assets).mean()
    ratio = ic1 / ic0 if np.isfinite(ic0) and abs(ic0) > 1e-12 else np.nan
    det = {"rank_ic": float(ic0), "rank_ic_delayed": float(ic1), "retained_share": float(ratio)}
    strong = abs(ic0) > 0.10
    collapses = np.isnan(ratio) or ratio < 0.25
    if strong and collapses:
        return CheckResult(
            name, FAIL,
            f"Mean rank IC {ic0:.3f} is implausibly high and keeps only {ratio:.0%} of its value "
            "after a one-period delay -- classic signature of look-ahead leakage.", det)
    if collapses and horizon > 1:
        return CheckResult(
            name, WARN,
            f"IC keeps only {ratio:.0%} after a one-period delay. Fine for a genuinely fast "
            "signal, but confirm timestamps and execution lag.", det)
    return CheckResult(name, PASS, f"IC retains {ratio:.0%} of its value after a one-period delay.", det)


# --------------------------------------------------------------------------- #
# 3. Statistical evidence
# --------------------------------------------------------------------------- #
def check_ic_significance(df: pd.DataFrame, horizon: int, n_trials: int = 1) -> CheckResult:
    """Rank IC t-stat with Newey-West errors (overlap-aware) against a multiple-testing bar."""
    name = "IC significance (Newey-West, multiple-testing aware)"
    ic = M.ic_series(df)
    lags = M.default_nw_lags(ic.size, horizon)
    t = M.newey_west_t(ic.values, lags)
    t_naive = M.newey_west_t(ic.values, 0)
    bar = max(3.0, M.bonferroni_t_threshold(n_trials))
    det = {"mean_rank_ic": float(ic.mean()), "ic_ir": float(ic.mean() / ic.std()),
           "t_newey_west": float(t), "t_naive": float(t_naive), "nw_lags": lags,
           "n_dates": int(ic.size), "n_trials": n_trials, "t_threshold": bar}
    if abs(t) >= bar:
        status, msg = PASS, f"|t| = {abs(t):.2f} clears the bar of {bar:.2f}"
    elif abs(t) >= 2.0:
        status, msg = WARN, f"|t| = {abs(t):.2f} is nominally significant but below the bar of {bar:.2f}"
    else:
        status, msg = FAIL, f"|t| = {abs(t):.2f}: no reliable evidence of predictive power"
    msg += f" (naive t = {t_naive:.2f}, {lags} NW lags, {n_trials} trial(s) declared)."
    return CheckResult(name, status, msg, det)


def check_placebo(df: pd.DataFrame, n_perm: int = 500, seed: int = 0) -> CheckResult:
    """Shuffle the signal across assets within each date and recompute the mean rank IC.

    Uses the identity that permuting the signal permutes its ranks, so each date's
    placebo IC is a dot product of centred ranks; all permutations run vectorised.
    Note: this tests the signal against *chance*, not against *selection* -- a signal
    picked as the best of many will still beat placebos. That is what the
    multiple-testing checks are for.
    """
    name = "Placebo: within-date permutation"
    rng = np.random.default_rng(seed)
    d = df.dropna(subset=["signal", "fwd_ret"])
    obs = M.ic_series(d).mean()
    null_sum = np.zeros(n_perm)
    n_dates = 0
    for _, g in d.groupby("date"):
        if len(g) < 5:
            continue
        rs = g["signal"].rank().to_numpy(dtype=float, copy=True)
        rr = g["fwd_ret"].rank().to_numpy(dtype=float, copy=True)
        rs -= rs.mean()
        rr -= rr.mean()
        den = np.sqrt((rs @ rs) * (rr @ rr))
        if den == 0:
            continue
        perms = rng.permuted(np.broadcast_to(rs, (n_perm, rs.size)), axis=1)
        null_sum += perms @ rr / den
        n_dates += 1
    null = null_sum / max(n_dates, 1)
    p = (1 + np.sum(np.abs(null) >= abs(obs))) / (n_perm + 1)
    det = {"observed_mean_ic": float(obs), "null_mean": float(null.mean()),
           "null_p95_abs": float(np.quantile(np.abs(null), 0.95)), "p_value": float(p), "n_perm": n_perm}
    if p <= 0.01:
        return CheckResult(name, PASS, f"Observed IC beats {n_perm} shuffled placebos (p = {p:.3f}).", det)
    if p <= 0.05:
        return CheckResult(name, WARN, f"Borderline vs. placebos (p = {p:.3f}).", det)
    return CheckResult(name, FAIL, f"Indistinguishable from shuffled signals (p = {p:.3f}).", det)


def check_monotonicity(df: pd.DataFrame, n_quantiles: int = 5) -> CheckResult:
    """Average return should rise (or fall) steadily across signal quantiles."""
    name = "Quantile monotonicity"
    qr = M.quantile_returns(df, n_quantiles).mean()
    rho = pd.Series(qr.values).corr(pd.Series(np.arange(len(qr))), method="spearman")
    det = {"quantile_mean_returns": {int(k): float(v) for k, v in qr.items()}, "spearman": float(rho)}
    if abs(rho) >= 0.9:
        return CheckResult(name, PASS, f"Quantile returns are monotone (Spearman {rho:+.2f}).", det)
    if abs(rho) >= 0.6:
        return CheckResult(name, WARN, f"Mostly monotone (Spearman {rho:+.2f}); check the middle buckets.", det)
    return CheckResult(name, FAIL, f"Not monotone (Spearman {rho:+.2f}): the spread may come from one extreme bucket.", det)


def check_stability(df: pd.DataFrame, n_splits: int = 4) -> CheckResult:
    """Mean IC should keep the same sign across consecutive sub-periods."""
    name = "Sub-period stability"
    ic = M.ic_series(df)
    edges = np.linspace(0, ic.size, n_splits + 1).astype(int)
    chunks = [ic.iloc[a:b] for a, b in zip(edges[:-1], edges[1:])]
    means = [float(c.mean()) for c in chunks]
    full = np.sign(ic.mean())
    agree = sum(np.sign(m) == full for m in means)
    det = {"subperiod_mean_ic": {f"{c.index[0].date()}..{c.index[-1].date()}": m
                                 for c, m in zip(chunks, means)}}
    if agree == n_splits:
        return CheckResult(name, PASS, f"Same IC sign in all {n_splits} sub-periods.", det)
    if agree >= n_splits - 1:
        return CheckResult(name, WARN, f"IC sign flips in {n_splits - agree} of {n_splits} sub-periods.", det)
    return CheckResult(name, FAIL, f"IC sign flips in {n_splits - agree} of {n_splits} sub-periods.", det)


# --------------------------------------------------------------------------- #
# 4. Economic value
# --------------------------------------------------------------------------- #
def long_short_backtest(df: pd.DataFrame, horizon: int, n_quantiles: int = 5) -> pd.DataFrame:
    """Non-overlapping long-short backtest: rebalance every ``horizon`` dates.

    Returns a frame indexed by rebalance date with gross return and one-way turnover.
    Gross return is the top-minus-bottom equal-weighted spread (2x gross exposure).
    """
    dates = np.sort(df["date"].unique())[::horizon]
    sub = df[df["date"].isin(dates)].dropna(subset=["signal", "fwd_ret"])
    w = M.long_short_weights(sub, n_quantiles)
    r = sub.pivot(index="date", columns="asset", values="fwd_ret").reindex_like(w).fillna(0.0)
    gross = (w * r).sum(axis=1)
    return pd.DataFrame({"gross": gross, "turnover": M.turnover(w)})


def check_costs(df: pd.DataFrame, horizon: int, periods_per_year: float,
                cost_bps: float = 10.0, n_quantiles: int = 5) -> CheckResult:
    """Does the spread survive a realistic one-way trading cost?"""
    name = "Transaction costs"
    bt = long_short_backtest(df, horizon, n_quantiles)
    ann = periods_per_year / horizon
    net = bt["gross"] - bt["turnover"] * cost_bps / 1e4
    avg_to = bt["turnover"].mean()
    breakeven = bt["gross"].mean() / avg_to * 1e4 if avg_to > 0 else np.inf
    det = {"rebalances": int(len(bt)), "gross_sharpe": M.sharpe(bt["gross"], ann),
           "net_sharpe": M.sharpe(net, ann), "avg_one_way_turnover": float(avg_to),
           "gross_ann_return": float(bt["gross"].mean() * ann), "net_ann_return": float(net.mean() * ann),
           "cost_bps_assumed": cost_bps, "breakeven_cost_bps": float(breakeven)}
    if breakeven < cost_bps:
        return CheckResult(name, FAIL, f"Break-even cost is {breakeven:.1f} bps, below the assumed "
                           f"{cost_bps:.0f} bps: the edge is eaten by trading.", det)
    if breakeven < 2 * cost_bps:
        return CheckResult(name, WARN, f"Break-even cost {breakeven:.1f} bps leaves little margin over "
                           f"{cost_bps:.0f} bps.", det)
    return CheckResult(name, PASS, f"Break-even cost {breakeven:.1f} bps; net Sharpe "
                       f"{det['net_sharpe']:.2f} at {cost_bps:.0f} bps.", det)


def check_deflated_sharpe(df: pd.DataFrame, horizon: int, n_trials: int,
                          n_quantiles: int = 5) -> CheckResult:
    """Is the long-short Sharpe better than the best of ``n_trials`` lucky draws?"""
    name = "Deflated Sharpe ratio"
    bt = long_short_backtest(df, horizon, n_quantiles)
    dsr, sr, sr0 = M.deflated_sharpe_ratio(bt["gross"].values, n_trials)
    det = {"dsr": dsr, "sr_per_period": sr, "sr0_expected_max_under_null": sr0, "n_trials": n_trials}
    if np.isnan(dsr):
        return CheckResult(name, WARN, "Too few rebalances to compute the deflated Sharpe ratio.", det)
    ctx = ("for a single declared trial this is the probabilistic Sharpe ratio, P(true Sharpe > 0)"
           if n_trials <= 1 else f"after accounting for {n_trials} trials")
    if dsr >= 0.95:
        return CheckResult(name, PASS, f"DSR = {dsr:.2f} ({ctx}).", det)
    if dsr >= 0.80:
        return CheckResult(name, WARN, f"DSR = {dsr:.2f}: not convincing ({ctx}).", det)
    tail = (f"consistent with the luckiest of {n_trials} skill-less trials"
            if n_trials > 1 else "the Sharpe ratio is not reliably above zero")
    return CheckResult(name, FAIL, f"DSR = {dsr:.2f}: {tail} ({ctx}).", det)
