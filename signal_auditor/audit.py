"""Run every check on a signal panel and collect the results."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

from . import checks as C
from . import metrics as M
from .panel import validate_panel


@dataclass
class AuditConfig:
    horizon: int = 1                  # forward-return horizon, in panel periods
    periods_per_year: float = 252.0   # 252 for daily panels, 12 for monthly
    n_trials: int = 1                 # how many signal variants you tried before this one
    cost_bps: float = 10.0            # assumed one-way trading cost
    n_quantiles: int = 5
    n_placebo: int = 500
    min_assets: int = 20
    universe_note: str | None = None  # e.g. "current S&P 500 constituents"


@dataclass
class AuditReport:
    name: str
    config: AuditConfig
    results: list[C.CheckResult]
    ic: pd.Series = field(repr=False)
    quantile_returns: pd.DataFrame = field(repr=False)
    backtest: pd.DataFrame = field(repr=False)

    @property
    def verdict(self) -> str:
        statuses = {r.status for r in self.results}
        if C.FAIL in statuses:
            return "NOT SUPPORTED"
        if C.WARN in statuses:
            return "PROVISIONAL"
        return "NO RED FLAGS DETECTED"

    def counts(self) -> dict[str, int]:
        out = {C.PASS: 0, C.WARN: 0, C.FAIL: 0}
        for r in self.results:
            out[r.status] += 1
        return out

    def status_of(self, check_name_prefix: str) -> str:
        for r in self.results:
            if r.name.startswith(check_name_prefix):
                return r.status
        raise KeyError(check_name_prefix)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{"check": r.name, "status": r.status, "summary": r.summary}
                             for r in self.results])

    def to_dict(self) -> dict:
        return {"name": self.name, "verdict": self.verdict, "config": asdict(self.config),
                "results": [asdict(r) for r in self.results]}

    def to_markdown(self) -> str:
        from .report import render_markdown
        return render_markdown(self)

    def to_html(self, path: str | None = None) -> str:
        from .report import render_html
        html = render_html(self)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(html)
        return html


def audit(df: pd.DataFrame, name: str = "signal", config: AuditConfig | None = None,
          **kwargs) -> AuditReport:
    """Audit a long panel with columns ``date, asset, signal, fwd_ret[, available_at]``.

    Keyword arguments override fields of :class:`AuditConfig`, e.g.
    ``audit(panel, horizon=21, periods_per_year=252, n_trials=40)``.
    """
    cfg = config or AuditConfig()
    for k, v in kwargs.items():
        if not hasattr(cfg, k):
            raise TypeError(f"unknown audit option: {k}")
        setattr(cfg, k, v)

    df = validate_panel(df)
    h, q = cfg.horizon, cfg.n_quantiles
    results = [
        C.check_timestamps(df),
        C.check_coverage(df, cfg.min_assets),
        C.check_universe(df, cfg.universe_note),
        C.check_lag_decay(df, h),
        C.check_ic_significance(df, h, cfg.n_trials),
        C.check_placebo(df, cfg.n_placebo),
        C.check_monotonicity(df, q),
        C.check_stability(df),
        C.check_costs(df, h, cfg.periods_per_year, cfg.cost_bps, q),
        C.check_deflated_sharpe(df, h, cfg.n_trials, q),
    ]
    bt = C.long_short_backtest(df, h, q)
    bt["net"] = bt["gross"] - bt["turnover"] * cfg.cost_bps / 1e4
    return AuditReport(name=name, config=cfg, results=results, ic=M.ic_series(df),
                       quantile_returns=M.quantile_returns(df, q), backtest=bt)
