"""The investment committee: independent scoring, then debate on disagreement.

Protocol (after AlphaAgents' rules, with controls added):

1. **Independent round (round 0).** Every persona scores every firm on an integer
   scale from -3 (strong avoid) to +3 (strong buy) for the next quarter, from the
   anonymised fact sheet alone, without seeing other personas.
2. **Debate only on disagreement.** Per date, the ``n_disputed`` firms with the
   largest spread of round-0 scores (and at least one buy and one avoid) go to
   debate. In each round every persona sees all personas' latest scores and
   reasons and may revise.
3. **Aggregation.** Committee signals: the round-0 mean (a vote, no debate) and
   the final-round mean (after debate). Comparing them on the same firms isolates
   what debate adds over voting.

Conformity diagnostics (see :func:`debate_diagnostics`): score dispersion before
vs after, the share of revisions that move toward the panel mean, and how often
the panel ends unanimous.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cards import COLUMNS, chunks, render_table
from personas import Persona, twin_scores

SCORE_MIN, SCORE_MAX = -3, 3

SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "score": {"type": "integer"},
                    "reason": {"type": "string"},
                },
                "required": ["id", "score", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["scores"],
    "additionalProperties": False,
}

TASK = f"""You are one member of an investment committee. Apply YOUR philosophy only.

You will see anonymised fact sheets of US-listed companies, one per row. All figures
come from the latest annual report already filed and from past prices; nothing is
from the future. Company identities, tickers and calendar dates are deliberately
hidden: judge only from the numbers. Do not try to guess which company it is.

For EVERY row, give an integer score for the stock's expected return over the next
three months relative to the other companies shown, from {SCORE_MIN} (strong avoid)
to {SCORE_MAX} (strong buy), 0 if neutral or unsure, and a reason of at most 15
words citing the figures that drive it. Use the full scale across the rows.
Return one entry per id."""

DEBATE = f"""You are one member of an investment committee. Apply YOUR philosophy.

The committee disagrees about the companies below. For each company you see its
anonymised fact sheet and every member's current score ({SCORE_MIN}..{SCORE_MAX})
and reason. Reconsider your own score: change it only if another member's argument
is convincing under YOUR philosophy; keep it if not. Do not move toward the others
just to agree. Return one entry per id with your score now and a reason of at most
20 words."""


def system_prompt(persona: Persona, debate: bool = False) -> str:
    return f"{persona.philosophy}\n\n{DEBATE if debate else TASK}"


def _rid(stage: str, persona: str, date, idx: int) -> str:
    return f"{stage}_{persona}_{pd.Timestamp(date):%Y%m%d}_{idx:03d}"


def _collect(answer: dict | None, chunk: pd.DataFrame) -> pd.DataFrame:
    valid = set(chunk["firm_id"])
    rows = []
    if answer:
        for s in answer.get("scores", []):
            if s.get("id") in valid and isinstance(s.get("score"), int):
                rows.append({"firm_id": s["id"],
                             "score": int(np.clip(s["score"], SCORE_MIN, SCORE_MAX)),
                             "reason": str(s.get("reason", ""))[:300]})
    out = pd.DataFrame(rows, columns=["firm_id", "score", "reason"]).drop_duplicates("firm_id")
    return chunk[["date", "cik", "firm_id"]].merge(out, on="firm_id", how="left")


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
class RuleBackend:
    """No-LLM backend: each persona scores with its rule-based twin.

    Debate is *simulated* (each revision moves halfway to the others' mean) so the
    plumbing and the diagnostics can be exercised without an API key.
    """

    def __init__(self, personas: dict[str, Persona]):
        self.personas = personas

    def score(self, persona: Persona, chunk: pd.DataFrame) -> pd.DataFrame:
        out = chunk[["date", "cik", "firm_id"]].copy()
        out["score"] = twin_scores(chunk, persona).values
        out["reason"] = "rule-based twin"
        return out

    def revise(self, persona: Persona, panel: pd.DataFrame) -> pd.DataFrame:
        own = panel[panel["persona"] == persona.key].set_index("firm_id")["score"]
        others = panel[panel["persona"] != persona.key].groupby("firm_id")["score"].mean()
        new = np.round(0.5 * own + 0.5 * others.reindex(own.index)).astype(int)
        return pd.DataFrame({"firm_id": own.index, "score": new.values,
                             "reason": "simulated revision"})


# --------------------------------------------------------------------------- #
# Rounds
# --------------------------------------------------------------------------- #
def independent_round(cards: pd.DataFrame, personas: dict[str, Persona], backend,
                      chunk_size: int = 40, mode: str = "sync") -> pd.DataFrame:
    parts = chunks(cards, chunk_size)
    out = []
    if isinstance(backend, RuleBackend):
        for p in personas.values():
            for ch in parts:
                out.append(backend.score(p, ch).assign(persona=p.key))
    else:
        requests, index = {}, {}
        for p in personas.values():
            sys_p = system_prompt(p)
            for i, ch in enumerate(parts):
                rid = _rid("r0", p.key, ch["date"].iloc[0], i)
                requests[rid] = (sys_p, render_table(ch))
                index[rid] = (p.key, ch)
        answers = backend.run(requests, mode=mode)
        for rid, (pkey, ch) in index.items():
            out.append(_collect(answers[rid], ch).assign(persona=pkey))
    res = pd.concat(out, ignore_index=True).assign(round=0)
    return res


def disputed(scores0: pd.DataFrame, n_disputed: int = 15) -> pd.DataFrame:
    """Per date, the firms whose round-0 scores spread the most (with a buy and an avoid)."""
    g = scores0.dropna(subset=["score"]).groupby(["date", "firm_id"])["score"]
    stats = pd.DataFrame({"sd": g.std(), "lo": g.min(), "hi": g.max()}).reset_index()
    stats = stats[(stats["lo"] < 0) & (stats["hi"] > 0)]
    return (stats.sort_values("sd", ascending=False).groupby("date").head(n_disputed)
            [["date", "firm_id"]])


def _debate_prompt(cards: pd.DataFrame, panel: pd.DataFrame, labels: dict[str, str]) -> str:
    blocks = []
    for fid, g in panel.groupby("firm_id"):
        sheet = render_table(cards[cards["firm_id"] == fid])
        views = "\n".join(f"- {labels[r.persona]}: {int(r.score):+d} ({r.reason})"
                          for r in g.itertuples())
        blocks.append(f"{sheet}\nCommittee views on {fid}:\n{views}")
    return "\n\n".join(blocks)


def debate(cards: pd.DataFrame, scores0: pd.DataFrame, personas: dict[str, Persona], backend,
           n_disputed: int = 15, rounds: int = 2, mode: str = "sync") -> pd.DataFrame:
    """Run ``rounds`` debate rounds on disputed firms; returns rounds 1..R."""
    labels = {k: p.label for k, p in personas.items()}
    keys = disputed(scores0, n_disputed)
    current = scores0.merge(keys, on=["date", "firm_id"]).dropna(subset=["score"])
    out = []
    for r in range(1, rounds + 1):
        new_rows = []
        if isinstance(backend, RuleBackend):
            for (d, p), _ in current.groupby(["date", "persona"]):
                panel = current[current["date"] == d]
                rev = backend.revise(personas[p], panel)
                new_rows.append(rev.assign(date=d, persona=p))
        else:
            requests, index = {}, {}
            for d, panel in current.groupby("date"):
                user = _debate_prompt(cards[cards["date"] == d], panel, labels)
                for p in personas.values():
                    rid = _rid(f"d{r}", p.key, d, 0)
                    requests[rid] = (system_prompt(p, debate=True), user)
                    index[rid] = (p.key, d, cards[(cards["date"] == d)
                                                  & cards["firm_id"].isin(panel["firm_id"])])
            answers = backend.run(requests, mode=mode)
            for rid, (pkey, d, ch) in index.items():
                new_rows.append(_collect(answers[rid], ch).assign(persona=pkey)
                                [["firm_id", "score", "reason", "date", "persona"]])
        new = pd.concat(new_rows, ignore_index=True)
        prev = current.set_index(["date", "firm_id", "persona"])
        new = new.set_index(["date", "firm_id", "persona"])
        new["score"] = new["score"].fillna(prev["score"].reindex(new.index))  # missing -> keep
        new["cik"] = prev["cik"].reindex(new.index)
        current = new.reset_index().assign(round=r)
        out.append(current)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def debate_diagnostics(scores0: pd.DataFrame, debated: pd.DataFrame) -> dict:
    """Conformity: dispersion before/after, moves toward the mean, unanimity."""
    if debated.empty:
        return {}
    last = debated[debated["round"] == debated["round"].max()]
    keys = last[["date", "firm_id"]].drop_duplicates()
    before = scores0.merge(keys, on=["date", "firm_id"])
    sd0 = before.groupby(["date", "firm_id"])["score"].std().mean()
    sd1 = last.groupby(["date", "firm_id"])["score"].std().mean()
    m = before.merge(last, on=["date", "firm_id", "persona"], suffixes=("_0", "_1"))
    mean0 = m.groupby(["date", "firm_id"])["score_0"].transform("mean")
    moved = m["score_1"] != m["score_0"]
    toward = (m["score_1"] - mean0).abs() < (m["score_0"] - mean0).abs()
    unanimous = last.groupby(["date", "firm_id"])["score"].agg(lambda s: s.nunique() == 1).mean()
    sign_agree = last.groupby(["date", "firm_id"])["score"].agg(
        lambda s: (np.sign(s) == np.sign(s.mean())).all()).mean()
    return {"firms_debated": int(len(keys)), "dispersion_before": float(sd0),
            "dispersion_after": float(sd1), "share_revised": float(moved.mean()),
            "share_of_revisions_toward_mean": float(toward[moved].mean()) if moved.any() else 0.0,
            "share_unanimous_after": float(unanimous), "share_same_sign_after": float(sign_agree)}


def committee_signals(scores0: pd.DataFrame, debated: pd.DataFrame) -> pd.DataFrame:
    """Per (date, cik): each persona's round-0 score, the vote mean, and the debate mean.

    For firms that were not debated, the debate mean equals the vote mean.
    """
    wide = scores0.pivot_table(index=["date", "cik"], columns="persona", values="score")
    wide.columns = [f"p_{c}" for c in wide.columns]
    out = wide.copy()
    out["vote_mean"] = scores0.groupby(["date", "cik"])["score"].mean()
    out["debate_mean"] = out["vote_mean"]
    if not debated.empty:
        last = debated[debated["round"] == debated["round"].max()]
        dm = last.groupby(["date", "cik"])["score"].mean()
        out.loc[dm.index, "debate_mean"] = dm
        out["debated"] = out.index.isin(dm.index)
    else:
        out["debated"] = False
    return out.reset_index()


__all__ = ["SCHEMA", "COLUMNS", "RuleBackend", "independent_round", "debate", "disputed",
           "debate_diagnostics", "committee_signals", "system_prompt"]
