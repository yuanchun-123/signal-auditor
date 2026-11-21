"""Offline tests for research/agent_committee (no network, no API key)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "research" / "agent_committee"))

import committee as C  # noqa: E402
from cards import COLUMNS, chunks, firm_id, prepare, render_table  # noqa: E402
from personas import PERSONAS, Persona, twin_scores  # noqa: E402


@pytest.fixture(scope="module")
def features():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-03-31", periods=6, freq="QE")
    rows = []
    for d in dates:
        for cik in range(1000, 1090):
            r = {"date": d, "cik": cik, "sector": "Technology", "log_mcap": rng.normal(23, 1),
                 "mcap": 1e10, "fwd_ret_3m": rng.normal(0.02, 0.1)}
            for c, _, k in COLUMNS:
                if k != "text" and c != "size_decile":
                    r[c] = rng.normal(0.1, 0.05)
            rows.append(r)
    return pd.DataFrame(rows)


def test_cards_hide_identity(features):
    cards = prepare(features.assign(name="ACME CORP", ticker="ACME"))
    table = render_table(cards.head(30))
    for leak in ["ACME", "1000", "2020", "cik"]:
        assert leak not in table
    assert table.splitlines()[0].startswith("id | sector")
    assert len(table.splitlines()) == 31


def test_firm_ids_change_across_dates():
    a = firm_id(pd.Timestamp("2020-03-31"), 1000)
    b = firm_id(pd.Timestamp("2020-06-30"), 1000)
    assert a != b and a == firm_id(pd.Timestamp("2020-03-31"), 1000)


def test_chunks_cover_each_date_once(features):
    cards = prepare(features)
    parts = chunks(cards, size=40)
    allrows = pd.concat(parts)
    assert len(allrows) == len(cards) and not allrows.duplicated(["date", "cik"]).any()
    assert all(p["date"].nunique() == 1 for p in parts)


def test_twin_scores_on_integer_scale(features):
    cards = prepare(features)
    for p in PERSONAS.values():
        s = twin_scores(cards, p)
        assert s.between(-3, 3).all() and len(s) == len(cards)


def test_skill_md_loader(tmp_path):
    f = tmp_path / "value-skill" / "SKILL.md"
    f.parent.mkdir()
    f.write_text("---\nname: value\ndescription: x\n---\nBuy cheap, safe companies.\n")
    p = Persona.from_skill_md(f)
    assert p.philosophy == "Buy cheap, safe companies." and p.key == "value-skill"


def test_rule_committee_end_to_end(features):
    cards = prepare(features)
    personas = {k: PERSONAS[k] for k in ["deep_value", "trend_follower", "forensic_accountant"]}
    backend = C.RuleBackend(personas)
    s0 = C.independent_round(cards, personas, backend, chunk_size=40)
    assert s0["score"].notna().all() and set(s0["persona"]) == set(personas)
    deb = C.debate(cards, s0, personas, backend, n_disputed=5, rounds=2)
    diag = C.debate_diagnostics(s0, deb)
    assert diag["dispersion_after"] <= diag["dispersion_before"]  # simulated revisions converge
    sig = C.committee_signals(s0, deb)
    assert {"vote_mean", "debate_mean", "debated"} <= set(sig.columns)
    assert sig["debated"].any()


def test_collect_drops_unknown_ids_and_clips():
    chunk = pd.DataFrame({"date": [pd.Timestamp("2020-03-31")] * 2, "cik": [1, 2],
                          "firm_id": ["FAAAAA", "FBBBBB"]})
    ans = {"scores": [{"id": "FAAAAA", "score": 9, "reason": "x"},
                      {"id": "FZZZZZ", "score": 1, "reason": "hallucinated id"}]}
    out = C._collect(ans, chunk).set_index("firm_id")
    assert out.loc["FAAAAA", "score"] == 3 and np.isnan(out.loc["FBBBBB", "score"])
