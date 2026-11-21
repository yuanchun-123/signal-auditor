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
from .factors import align_to_rebalances
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


def check_universe(df: pd.DataFrame, universe_note: str | None = None,
                   intended_assets: int | None = None) -> CheckResult:
    """Flag survivorship bias.

    Two symptoms: (a) no asset ever leaves the sample, (b) far fewer assets reach
    the panel than the universe you meant to study (``intended_assets``) -- e.g.
    delisted firms silently dropped because a price source only covers survivors.
    The auditor cannot see firms that never entered the panel unless told.
    """
    name = "Universe / survivorship"
    last_seen = df.groupby("asset")["date"].max()
    final = df["date"].max()
    exits = int((last_seen < final).sum())
    det = {"assets": int(last_seen.size), "assets_that_exit": exits}
    if universe_note:
        det["note"] = universe_note
    problems = []
    if intended_assets:
        cov = last_seen.size / intended_assets
        det["intended_assets"], det["coverage"] = intended_assets, cov
        if cov < 0.9:
            problems.append(f"only {last_seen.size} of {intended_assets} intended assets "
                            f"({cov:.0%}) reach the panel; if the missing ones are delisted or "
                            "acquired firms, results carry survivorship bias")
    if exits == 0 and last_seen.size >= 20:
        problems.append("no asset ever leaves the sample; if the universe is today's "
                        "constituents, delisted losers are missing")
    if problems:
        return CheckResult(name, WARN, "; ".join(problems).capitalize() + ".", det)
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
    if bt["gross"].mean() <= 0:
        return CheckResult(name, FAIL, f"The long-short spread is negative before costs "
                           f"({det['gross_ann_return']:+.2%}/yr): no edge to protect. "
                           "Check the signal's sign convention.", det)
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


# --------------------------------------------------------------------------- #
# 5. Is it just a known factor?
# --------------------------------------------------------------------------- #
def check_factor_spanning(df: pd.DataFrame, factors: pd.DataFrame, horizon: int,
                          n_quantiles: int = 5, alpha_t_bar: float = 2.0) -> CheckResult:
    """Regress the long-short return on known factor returns; does an intercept survive?

    ``factors`` holds factor returns indexed by the panel's calendar (e.g. month-end
    dates for a monthly panel), excluding the risk-free rate. The long-short
    portfolio is zero-investment, so it is regressed on the factors directly.
    """
    name = "Factor spanning (time-series alpha)"
    bt = long_short_backtest(df, horizon, n_quantiles)
    cal = pd.DatetimeIndex(np.sort(df["date"].unique()))
    F = align_to_rebalances(factors, cal, bt.index, horizon)
    y = bt["gross"].to_numpy()
    ok = F.notna().all(axis=1).to_numpy()
    coverage = float(ok.mean())
    if ok.sum() < 24:
        return CheckResult(name, WARN, f"Factor data cover only {ok.sum()} rebalances "
                           "(need 24+); check that factor dates match the panel calendar.",
                           {"coverage": coverage})
    lags = M.default_nw_lags(int(ok.sum()), 1)
    raw = M.ols_newey_west(y[ok], np.ones((ok.sum(), 1)), lags)
    X = np.column_stack([np.ones(ok.sum()), F.to_numpy()[ok]])
    fit = M.ols_newey_west(y[ok], X, lags)
    names = ["alpha"] + list(F.columns)
    loadings = {n: {"coef": float(c), "t": float(t)} for n, c, t in zip(names, fit["coef"], fit["t"])}
    a, ta, traw = float(fit["coef"][0]), float(fit["t"][0]), float(raw["t"][0])
    det = {"raw_mean": float(raw["coef"][0]), "raw_t": traw, "alpha": a, "alpha_t": ta,
           "r2": fit["r2"], "n": fit["n"], "coverage": coverage, "nw_lags": lags,
           "loadings": loadings}
    big = sorted(((abs(v["t"]), k) for k, v in loadings.items() if k != "alpha"), reverse=True)[:2]
    top = ", ".join(f"{k} (t={loadings[k]['t']:+.1f})" for _, k in big)
    if abs(ta) >= alpha_t_bar:
        return CheckResult(name, PASS, f"Alpha {a * 1e4:+.1f} bps/period survives the factors "
                           f"(t = {ta:.2f}, R² = {fit['r2']:.2f}).", det)
    if abs(traw) >= alpha_t_bar:
        return CheckResult(name, FAIL, f"Raw spread is significant (t = {traw:.2f}) but alpha is not "
                           f"(t = {ta:.2f}): explained by known factors, mainly {top}.", det)
    return CheckResult(name, WARN, f"Neither the raw spread (t = {traw:.2f}) nor alpha (t = {ta:.2f}) "
                       "is significant, so the spanning test is uninformative.", det)


def neutralize_signal(df: pd.DataFrame, exposures: list[str]) -> pd.DataFrame:
    """Replace ``signal`` by its residual from a per-date OLS on ``exposures`` (+ intercept)."""
    cols = ["signal"] + list(exposures)
    out = df.dropna(subset=cols + ["fwd_ret"]).sort_values("date", kind="stable").reset_index(drop=True)
    resid = np.full(len(out), np.nan)
    pos = 0
    for _, g in out.groupby("date", sort=True):
        n = len(g)
        X = np.column_stack([np.ones(n), g[list(exposures)].to_numpy(dtype=float)])
        y = g["signal"].to_numpy(dtype=float)
        if n > X.shape[1] + 2:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid[pos:pos + n] = y - X @ beta
        pos += n
    out["signal"] = resid
    return out.dropna(subset=["signal"])


def check_neutralized_ic(df: pd.DataFrame, exposures: list[str], horizon: int) -> CheckResult:
    """How much of the IC is left after removing known characteristics from the signal?"""
    name = "Characteristic-neutral IC"
    missing = [c for c in exposures if c not in df.columns]
    if missing:
        return CheckResult(name, WARN, f"Exposure columns not in panel: {missing}.")
    base = df.dropna(subset=["signal", "fwd_ret"] + list(exposures)).sort_values("date", kind="stable")
    ic_raw = M.ic_series(base)
    neu = neutralize_signal(base, exposures)
    ic_neu = M.ic_series(neu)
    lags = M.default_nw_lags(ic_neu.size, horizon)
    t_neu = M.newey_west_t(ic_neu.values, lags)
    r_raw, r_neu = float(ic_raw.mean()), float(ic_neu.mean())
    kept = r_neu / r_raw if abs(r_raw) > 1e-12 else float("nan")
    det = {"exposures": list(exposures), "rank_ic_raw": r_raw, "rank_ic_neutral": r_neu,
           "retained_share": kept, "t_neutral": float(t_neu), "nw_lags": lags}
    lbl = ", ".join(exposures)
    if np.isfinite(kept) and kept >= 0.5 and abs(t_neu) >= 2:
        return CheckResult(name, PASS, f"After neutralising {lbl}, IC keeps {kept:.0%} "
                           f"({r_raw:.3f} → {r_neu:.3f}, t = {t_neu:.2f}).", det)
    if np.isfinite(kept) and kept >= 0.2:
        return CheckResult(name, WARN, f"After neutralising {lbl}, IC keeps {kept:.0%} "
                           f"({r_raw:.3f} → {r_neu:.3f}, t = {t_neu:.2f}).", det)
    return CheckResult(name, FAIL, f"After neutralising {lbl}, IC falls from {r_raw:.3f} to "
                       f"{r_neu:.3f} (t = {t_neu:.2f}): the signal is mostly a known characteristic.", det)
