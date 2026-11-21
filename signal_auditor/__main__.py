"""Command line: ``python -m signal_auditor panel.csv --horizon 1 --ppy 12 --trials 20``."""

from __future__ import annotations

import argparse

import pandas as pd

from .audit import audit


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="signal-auditor", description=__doc__)
    ap.add_argument("panel", help="CSV or Parquet with columns date, asset, signal, fwd_ret[, available_at]")
    ap.add_argument("--name", default=None)
    ap.add_argument("--horizon", type=int, default=1, help="forward-return horizon in panel periods")
    ap.add_argument("--ppy", type=float, default=252, help="periods per year (252 daily, 12 monthly)")
    ap.add_argument("--trials", type=int, default=1, help="number of signal variants tried")
    ap.add_argument("--cost-bps", type=float, default=10.0, help="one-way trading cost in bps")
    ap.add_argument("--quantiles", type=int, default=5)
    ap.add_argument("--out", default="audit_report.html")
    a = ap.parse_args(argv)

    df = pd.read_parquet(a.panel) if a.panel.endswith(".parquet") else pd.read_csv(a.panel)
    rep = audit(df, name=a.name or a.panel, horizon=a.horizon, periods_per_year=a.ppy,
                n_trials=a.trials, cost_bps=a.cost_bps, n_quantiles=a.quantiles)
    rep.to_html(a.out)
    print(rep.to_markdown())
    print(f"HTML report written to {a.out}")
    return 1 if rep.verdict == "NOT SUPPORTED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
