"""Investor personas and their rule-based twins.

Each persona has

* a written philosophy, used as the LLM agent's system prompt, and
* a *rule-based twin*: a transparent formula over the same fact-sheet columns
  that encodes the persona's stated rules.

The twin is the key control of the design. If an LLM persona's scores are
explained by its own twin, the LLM added nothing beyond a factor formula; the
audit's characteristic-neutral check (neutralising the LLM score on the twin)
measures exactly that.

Personas are *styles inspired by* well-known public investing philosophies, not
simulations of the people. External persona files (e.g. Claude Code SKILL.md
files from GitHub) can be loaded with :func:`Persona.from_skill_md`; read any such
file before using it -- it becomes a system prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


def z(s: pd.Series) -> pd.Series:
    """Cross-sectional z-score, winsorised at 1%/99%, missing -> 0 (neutral)."""
    s = s.astype(float)
    lo, hi = s.quantile(0.01), s.quantile(0.99)
    s = s.clip(lo, hi)
    sd = s.std()
    out = (s - s.mean()) / sd if sd and sd > 0 else s * 0.0
    return out.fillna(0.0)


@dataclass(frozen=True)
class Persona:
    key: str
    label: str
    philosophy: str
    twin: Callable[[pd.DataFrame], pd.Series] | None = None

    @classmethod
    def from_skill_md(cls, path: str | Path, key: str | None = None) -> "Persona":
        """Use the body of a SKILL.md file (YAML front matter stripped) as the philosophy."""
        text = Path(path).read_text(encoding="utf-8")
        body = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.S).strip()
        name = key or Path(path).parent.name
        return cls(key=name, label=f"{name} (from {Path(path).name})", philosophy=body)


PERSONAS: dict[str, Persona] = {}


def _add(p: Persona) -> None:
    PERSONAS[p.key] = p


_add(Persona(
    key="quality_compounder",
    label="Quality at a fair price (Buffett-style)",
    philosophy=(
        "You invest like a patient owner of businesses. You want durable, high-return "
        "businesses: consistently high return on equity achieved without heavy leverage, "
        "strong and stable operating margins, and free cash flow that is real rather than "
        "accounting profit. You will pay a fair price for an excellent business but not an "
        "excessive one. You dislike high debt, capital-hungry businesses whose cash flow "
        "lags earnings, and anything you cannot judge from durable economics."),
    twin=lambda d: z(d["roe"]) + z(d["op_margin"]) + z(d["fcf_yield"]) - z(d["debt_to_equity"]),
))
_add(Persona(
    key="deep_value",
    label="Deep value with a margin of safety (Graham-style)",
    philosophy=(
        "You are a defensive value investor. You buy statistically cheap stocks: high "
        "earnings yield, high book-to-market, and a strong balance sheet (current ratio "
        "comfortably above 1, modest debt). You insist on a margin of safety and ignore "
        "stories, growth narratives and recent price momentum."),
    twin=lambda d: z(d["earnings_yield"]) + z(d["book_to_market"]) + z(d["current_ratio"])
    - z(d["debt_to_equity"]),
))
_add(Persona(
    key="garp",
    label="Growth at a reasonable price (Lynch-style)",
    philosophy=(
        "You look for companies growing sales and earnings steadily, bought at a price that "
        "does not already assume perfection: growth should be high relative to the valuation "
        "(think of a low price-to-earnings-to-growth ratio). You avoid shrinking businesses "
        "and expensive glamour stocks whose growth is already priced in."),
    twin=lambda d: z(d["revenue_growth"]) + z(d["net_income_growth"]) + z(d["earnings_yield"]),
))
_add(Persona(
    key="magic_formula",
    label="Cheap and good (Greenblatt-style)",
    philosophy=(
        "You rank businesses on two things only: how cheap they are (earnings yield) and how "
        "good they are (return on capital). The best ideas are both cheap and good. You are "
        "disciplined and unemotional and apply the same logic to every company."),
    twin=lambda d: d["earnings_yield"].rank(pct=True).fillna(0.5)
    + d["roa"].rank(pct=True).fillna(0.5),
))
_add(Persona(
    key="trend_follower",
    label="Momentum trader",
    philosophy=(
        "You are a systematic trend follower. Stocks that have outperformed over the past "
        "year (excluding the most recent month) tend to keep outperforming; recent one-month "
        "moves often reverse. You prefer strong, steady trends over volatile ones and you do "
        "not care about valuation or fundamentals."),
    twin=lambda d: z(d["mom_12_1"]) - 0.5 * z(d["vol_12m"]) - 0.25 * z(d["ret_1m"]),
))
_add(Persona(
    key="forensic_accountant",
    label="Forensic accountant / short-seller",
    philosophy=(
        "You are a forensic accountant who hunts for low-quality or manipulated earnings, the "
        "way activist short sellers do. Red flags: large accruals (earnings well above "
        "operating cash flow), receivables growing faster than sales, rapid asset growth, "
        "aggressive leverage. Clean companies whose cash flow backs up their earnings deserve "
        "high scores; companies whose reported profits look fragile deserve low scores."),
    twin=lambda d: -z(d["accruals_to_assets"]) + z(d["cfo_to_net_income"].clip(-5, 5))
    - z(d["receivables_index"]) - z(d["asset_growth"]),
))


def twin_scores(cards: pd.DataFrame, persona: Persona, levels: int = 7) -> pd.Series:
    """Twin score mapped to the LLM's integer scale (-3..3) by within-date quantiles."""
    if persona.twin is None:
        raise ValueError(f"persona {persona.key} has no rule-based twin")
    raw = pd.concat([persona.twin(g) for _, g in cards.groupby("date")]).reindex(cards.index)
    pct = raw.groupby(cards["date"]).rank(pct=True).fillna(0.5)
    half = (levels - 1) // 2
    return (np.ceil(pct * levels) - 1 - half).clip(-half, half).astype(int)
