"""Audit committee signals with signal-auditor.

Signals: each persona's round-0 score, each persona's rule-based twin, the vote
mean and the debate mean. Each is audited on next-quarter returns with
Fama-French 5 + momentum spanning (quarterly) and neutralisation on size,
value and momentum. For LLM personas, an extra check neutralises the LLM score on
its own twin: what is left is what the LLM added beyond the persona's formula.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))

from signal_auditor import audit  # noqa: E402
from signal_auditor import metrics as M  # noqa: E402
from signal_auditor.factors import load_french  # noqa: E402

CONTROLS = ["log_mcap", "book_to_market", "mom_12_1"]


def quarterly_factors(cache_dir: Path) -> pd.DataFrame:
    m = load_french("M", cache_dir=cache_dir).drop(columns="RF")
    q = m.groupby(m.index.to_period("Q")).sum(min_count=3)
    q.index = q.index.to_timestamp("Q") + pd.offsets.QuarterEnd(0)
    q.index = q.index.normalize()
    return q


def evaluate(signals: pd.DataFrame, features: pd.DataFrame, twins: pd.DataFrame,
             out_dir: Path, universe_size: int, llm: bool) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    feats = features[["date", "cik", "fwd_ret_3m"] + CONTROLS].copy()
    feats["date"] = pd.to_datetime(feats["date"])
    base = signals.merge(feats, on=["date", "cik"]).merge(twins, on=["date", "cik"], how="left")
    factors = quarterly_factors(HERE / "data")
    cols = [c for c in signals.columns if c.startswith("p_") or c.endswith("_mean")]
    cols += [c for c in twins.columns if c.startswith("twin_")]
    n_trials = len(cols)
    rows = []
    for col in cols:
        panel = base.rename(columns={col: "signal", "fwd_ret_3m": "fwd_ret", "cik": "asset"})
        neutral = list(CONTROLS)
        twin_col = "twin_" + col[2:] if col.startswith("p_") else None
        if llm and twin_col in panel.columns:
            neutral.append(twin_col)
        panel = panel.dropna(subset=["signal", "fwd_ret"] + neutral)
        if panel["date"].nunique() < 8:
            continue
        rep = audit(panel[["date", "asset", "signal", "fwd_ret"] + neutral], name=col,
                    horizon=1, periods_per_year=4, n_trials=n_trials, cost_bps=10,
                    factors=factors, neutralize=neutral, intended_assets=universe_size,
                    min_assets=20)
        rep.to_html(str(out_dir / f"{col}.html"))
        d = {r.name.split(" (")[0]: r for r in rep.results}
        sig = d["IC significance"].details
        span = d["Factor spanning"].details
        neu = d["Characteristic-neutral IC"].details
        rows.append({"signal": col, "verdict": rep.verdict, "rank_ic": sig["mean_rank_ic"],
                     "nw_t": sig["t_newey_west"], "alpha_t": span.get("alpha_t", np.nan),
                     "neutral_ic": neu.get("rank_ic_neutral", np.nan),
                     "neutral_t": neu.get("t_neutral", np.nan),
                     "dates": panel["date"].nunique()})
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    return summary


def debate_value(signals: pd.DataFrame, features: pd.DataFrame) -> dict:
    """On debated firm-dates only: does the debate mean predict better than the vote?"""
    d = signals[signals["debated"]].merge(
        features.assign(date=pd.to_datetime(features["date"]))[["date", "cik", "fwd_ret_3m"]],
        on=["date", "cik"]).dropna(subset=["fwd_ret_3m"])
    if d["date"].nunique() < 4:
        return {}
    out = {}
    for col in ["vote_mean", "debate_mean"]:
        p = d.rename(columns={col: "signal", "fwd_ret_3m": "fwd_ret"})
        ic = M.ic_series(p, min_assets=5)
        out[col] = {"mean_rank_ic": float(ic.mean()), "t": float(M.newey_west_t(ic.values, 1)),
                    "dates": int(ic.size)}
    return out
