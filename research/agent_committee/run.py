"""Run the agent committee end to end.

    # no API key needed: rule-based twins + simulated debate (plumbing check)
    python run.py --backend rule

    # pilot with Claude on a few dates, then the full sample
    python run.py --backend claude --dates 4 --estimate     # print the cost estimate only
    python run.py --backend claude --dates 4
    python run.py --backend claude --mode batch            # 50% cheaper, ~1h per batch

Outputs go to runs/<backend>-<model>/: scores, debate rounds, diagnostics,
an audit summary and one HTML audit report per signal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import committee as C
from cards import prepare, render_table, chunks
from evaluate import debate_value, evaluate
from llm import PRICES, ClaudeBackend
from personas import PERSONAS, twin_scores

HERE = Path(__file__).parent


def estimate(cards: pd.DataFrame, personas: dict, chunk_size: int, model: str,
             n_disputed: int, rounds: int, batch: bool) -> dict:
    """Rough token and dollar estimate (4 characters per token; output incl. thinking)."""
    parts = chunks(cards, chunk_size)
    sys_tokens = max(len(C.system_prompt(p)) for p in personas.values()) / 4
    in_tok = sum(len(render_table(ch)) / 4 + sys_tokens for ch in parts) * len(personas)
    out_tok = sum(len(ch) * 30 + 1500 for ch in parts) * len(personas)
    n_dates = cards["date"].nunique()
    deb_in = n_dates * rounds * len(personas) * (n_disputed * 450 + sys_tokens)
    deb_out = n_dates * rounds * len(personas) * (n_disputed * 35 + 1500)
    pin, pout = PRICES[model]
    mult = 0.5 if batch else 1.0
    usd = mult * ((in_tok + deb_in) * pin + (out_tok + deb_out) * pout) / 1e6
    return {"requests": len(parts) * len(personas) + n_dates * rounds * len(personas),
            "input_tokens_M": round((in_tok + deb_in) / 1e6, 2),
            "output_tokens_M": round((out_tok + deb_out) / 1e6, 2), "usd": round(usd, 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["rule", "claude"], default="rule")
    ap.add_argument("--model", default="claude-opus-5-5", choices=sorted(PRICES))
    ap.add_argument("--effort", default="low", choices=["low", "medium", "high"])
    ap.add_argument("--mode", default="sync", choices=["sync", "batch"])
    ap.add_argument("--personas", nargs="*", default=list(PERSONAS))
    ap.add_argument("--dates", type=int, default=0, help="use only the last N dates (pilot)")
    ap.add_argument("--chunk-size", type=int, default=40)
    ap.add_argument("--n-disputed", type=int, default=15)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--estimate", action="store_true", help="print the cost estimate and exit")
    a = ap.parse_args()

    features = pd.read_csv(HERE / "data" / "features.csv", parse_dates=["date"])
    universe_size = features["cik"].nunique()
    cards = prepare(features)
    if a.dates:
        keep = sorted(cards["date"].unique())[-a.dates - 1:-1]  # last dates with known returns
        cards = cards[cards["date"].isin(keep)]
    personas = {k: PERSONAS[k] for k in a.personas}

    est = estimate(cards, personas, a.chunk_size, a.model, a.n_disputed, a.rounds,
                   a.mode == "batch")
    print(f"{cards['date'].nunique()} dates, {cards['cik'].nunique()} firms, "
          f"{len(personas)} personas | estimate for {a.model}: {est}")
    if a.estimate:
        return

    tag = "rule" if a.backend == "rule" else f"claude-{a.model}-{a.effort}"
    out = HERE / "runs" / tag
    out.mkdir(parents=True, exist_ok=True)
    if a.backend == "rule":
        backend = C.RuleBackend(personas)
    else:
        backend = ClaudeBackend(C.SCHEMA, out / "responses.jsonl", model=a.model, effort=a.effort)

    scores0 = C.independent_round(cards, personas, backend, a.chunk_size, a.mode)
    print(f"round 0: {scores0['score'].notna().mean():.1%} of firm-persona scores returned")
    debated = C.debate(cards, scores0, personas, backend, a.n_disputed, a.rounds, a.mode)
    diag = C.debate_diagnostics(scores0, debated)
    scores0.to_csv(out / "scores_round0.csv", index=False)
    debated.to_csv(out / "debate_rounds.csv", index=False)

    signals = C.committee_signals(scores0, debated)
    twins = cards[["date", "cik"]].copy()
    for k, p in personas.items():
        if p.twin is not None:
            twins[f"twin_{k}"] = twin_scores(cards, p).values
    summary = evaluate(signals, features, twins, out / "audit", universe_size,
                       llm=a.backend == "claude")
    dv = debate_value(signals, features)
    report = {"backend": tag, "estimate": est, "debate_diagnostics": diag, "debate_value": dv}
    if a.backend == "claude":
        report["usage"] = backend.usage
        report["usd_spent_approx"] = round(backend.cost(), 2)
    (out / "run_summary.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
