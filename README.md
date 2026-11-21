# signal-auditor

**Catch the usual ways a cross-sectional trading-signal backtest fools you — before you believe it.**

Most backtests that look good are wrong for boring reasons: the signal peeks at the
future, the "best" variant was the luckiest of fifty, the t-stat ignores overlapping
returns, or the edge disappears after trading costs. `signal-auditor` takes a signal
panel and runs a fixed battery of checks for these failure modes, then writes a
PASS / WARN / FAIL report.

It is deliberately *not* a backtesting engine or an alpha generator. Every number is
produced by deterministic, unit-tested Python. A clean report does not mean "this is
alpha"; it means "none of the known ways of fooling yourself was detected".

## Quick start

```bash
pip install -e ".[examples,dev]"
python examples/synthetic_demo.py      # audits planted signals with a known answer
python examples/momentum_yahoo.py      # real data: 12-1 momentum on US large caps
```

From Python:

```python
from signal_auditor import audit
from signal_auditor.factors import load_french

# long panel: one row per (date, asset); optional `available_at` timestamp column
rep = audit(panel, name="accruals", horizon=1, periods_per_year=12,
            n_trials=25,      # how many variants you tried before this one -- be honest
            cost_bps=10,
            factors=load_french("M").drop(columns="RF"),   # optional
            neutralize=["log_mcap", "mom_12_1"])           # optional panel columns
print(rep.verdict)            # NOT SUPPORTED / PROVISIONAL / NO RED FLAGS DETECTED
rep.to_html("accruals_audit.html")
```

From the command line:

```bash
signal-auditor panel.csv --horizon 1 --ppy 12 --trials 25 --cost-bps 10 \
    --french M --neutralize log_mcap mom_12_1 --out report.html
```

The exit code is 1 when the verdict is `NOT SUPPORTED`, so the audit can gate a CI job.

## What it checks

| # | Check | Failure mode it targets | How |
|---|---|---|---|
| 1 | Point-in-time timestamps | Look-ahead bias | Every `available_at` must be ≤ the trade date |
| 2 | Cross-sectional coverage | Noisy quantiles from thin cross-sections | Assets per date |
| 3 | Universe / survivorship | Hindsight universes (delisted losers missing) | Flags samples in which no asset ever exits |
| 4 | One-period-delay test | Leakage without timestamps | Delay the signal one period; a strong IC that collapses to ~0 is a leakage signature |
| 5 | IC significance | Overstated t-stats; data snooping | Rank IC with Newey-West errors (lags ≥ horizon − 1); bar = max(3, Bonferroni for `n_trials`), following Harvey, Liu & Zhu (2016) |
| 6 | Placebo | Pure chance | Within-date permutation of the signal, vectorised |
| 7 | Quantile monotonicity | Spread driven by one extreme bucket | Spearman correlation of quantile mean returns with quantile rank |
| 8 | Sub-period stability | One lucky era | Sign of mean IC across four consecutive sub-periods |
| 9 | Transaction costs | Edge eaten by trading | Non-overlapping long-short backtest, turnover, break-even cost in bps |
| 10 | Deflated Sharpe ratio | Selection among many trials | Bailey & López de Prado (2014) |
| 11 | Factor spanning *(optional: `factors=`)* | A known factor in disguise | Long-short return regressed on factor returns (e.g. Fama-French 5 + momentum via `load_french`), Newey-West t on the intercept |
| 12 | Characteristic-neutral IC *(optional: `neutralize=[...]`)* | A known characteristic in disguise | Per-date OLS of the signal on size, momentum, etc.; rank IC of the residual |

## Does it work? Planted signals with a known answer

A tool that judges signals has to be tested on signals whose truth is known.
`signal_auditor.synthetic` builds a 240-month × 300-asset panel (with delistings) and plants:

| Case | Truth | Verdict | Caught by |
|---|---|---|---|
| `true` | Persistent signal, IC ≈ 0.03 | NO RED FLAGS DETECTED | — (t = 7.1, IC unchanged after a delay) |
| `leaky` | Signal = forward return + noise | NOT SUPPORTED | One-period-delay test (IC 0.29 → 0.003) |
| `mined`, 1 trial declared | Best of 200 random signals | NOT SUPPORTED | Transaction costs only; it **passes** the t > 3 bar (t = 3.4) and has DSR 0.98 |
| `mined`, 200 trials declared | Same signal, honest trial count | NOT SUPPORTED | Bar rises to 3.66 (WARN); deflated Sharpe falls to 0.27 (FAIL) |
| `stale_timestamp` | True signal, 15% published after use | NOT SUPPORTED | Timestamp check |

`make_factor_case()` adds a market with one priced characteristic:

| Case | Truth | Factor spanning | Characteristic-neutral IC |
|---|---|---|---|
| `disguised` | Noisy copy of the priced characteristic | FAIL: raw t = 4.8, alpha t = 0.9 | FAIL: IC 0.056 → 0.005 |
| `genuine` | Independent signal with real alpha | PASS: alpha t = 9.4 | PASS: IC unchanged |

The `mined` rows are the important lesson: **the placebo test does not catch selection**.
The luckiest of 200 random signals genuinely beats shuffled versions of itself. Only an
honest `n_trials` makes the multiple-testing checks bite.

## Real-data example

`examples/momentum_yahoo.py` audits 12-1 momentum on 61 *current* US large caps
(monthly, 2000–2026, Yahoo Finance adjusted closes). The auditor flags the hindsight
universe as survivorship-biased and finds no reliable predictive power in this small
universe (Newey-West t ≈ 1), so the factor-spanning check reports itself uninformative. That is the expected outcome; it is not evidence about
momentum in general.

## Panel format

| column | meaning |
|---|---|
| `date` | decision date |
| `asset` | identifier |
| `signal` | value known at `date` |
| `fwd_ret` | return from `date` to `date + horizon` periods |
| `available_at` *(optional)* | when the signal could first be observed, e.g. SEC filing acceptance time |

`signal_auditor.forward_returns(prices, horizon)` and `to_long(signal_wide, fwd_wide)`
build this panel from wide price and signal tables.

## Limitations

- Heuristic thresholds (e.g. the delay test's "IC > 0.10 and < 25% retained") are
  judgement calls, documented in `checks.py`; a genuinely fast signal can trip the delay WARN.
- The survivorship check can only see what is in the panel; it cannot know about firms you never loaded.
- Costs are a flat one-way bps assumption; no market impact or capacity model yet.
- Factor spanning needs factor returns on the panel's own calendar (month-end dates for monthly panels).

## Roadmap

- Size-dependent cost and capacity model
- Walk-forward evaluation for fitted (ML) signals
- An optional LLM layer that *explains* failed checks — never one that decides them
