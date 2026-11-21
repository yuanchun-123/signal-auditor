"""Anonymised fact sheets shown to the agents.

Look-ahead control: an LLM that recognises a company can "remember" what its
stock did next. So the agents never see a name, ticker, CIK, calendar date or
exact size. Each firm gets an ID that is random per rebalance date (so an agent
cannot follow a firm across quarters), size is shown only as a decile within the
universe, and every other field is a scale-free ratio rounded to two significant
digits. :func:`identity_probe_prompt` lets you measure how often a model can still
name the company.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

COLUMNS = [  # (feature column, header shown to the agent, kind)
    ("sector", "sector", "text"),
    ("size_decile", "size decile (10=largest)", "int"),
    ("earnings_yield", "earnings yield", "pct"),
    ("book_to_market", "book/market", "x"),
    ("fcf_yield", "FCF yield", "pct"),
    ("sales_to_price", "sales/price", "x"),
    ("roe", "ROE", "pct"),
    ("roa", "ROA", "pct"),
    ("gross_margin", "gross margin", "pct"),
    ("op_margin", "operating margin", "pct"),
    ("revenue_growth", "revenue growth", "pct"),
    ("net_income_growth", "net income growth", "pct"),
    ("accruals_to_assets", "accruals/assets", "pct"),
    ("cfo_to_net_income", "CFO/net income", "x"),
    ("receivables_index", "receivables-to-sales index", "x"),
    ("asset_growth", "asset growth", "pct"),
    ("debt_to_equity", "LT debt/equity", "x"),
    ("liabilities_to_assets", "liabilities/assets", "pct"),
    ("current_ratio", "current ratio", "x"),
    ("rnd_to_sales", "R&D/sales", "pct"),
    ("shareholder_yield", "dividends+buybacks yield", "pct"),
    ("mom_12_1", "return t-12..t-1 months", "pct"),
    ("ret_1m", "return last month", "pct"),
    ("vol_12m", "volatility (annualised)", "pct"),
]


def firm_id(date: pd.Timestamp, cik: int, salt: str = "committee-v1") -> str:
    h = hashlib.sha256(f"{salt}|{pd.Timestamp(date).date()}|{cik}".encode()).hexdigest()
    return "F" + h[:5].upper()


def _fmt(v, kind: str) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    if kind == "text":
        return str(v)
    if kind == "int":
        return str(int(v))
    if kind == "pct":
        return f"{float(f'{v * 100:.2g}'):g}%"
    return f"{float(f'{v:.2g}'):g}"


def prepare(features: pd.DataFrame) -> pd.DataFrame:
    """Add firm IDs and size deciles; drop everything that could identify a firm."""
    f = features.copy()
    f["date"] = pd.to_datetime(f["date"])
    f["firm_id"] = [firm_id(d, c) for d, c in zip(f["date"], f["cik"])]
    f["size_decile"] = f.groupby("date")["log_mcap"].transform(
        lambda s: np.ceil(s.rank(pct=True) * 10).clip(1, 10))
    return f


def render_table(chunk: pd.DataFrame) -> str:
    """Pipe-separated table, one firm per row. Row order is shuffled by firm ID."""
    chunk = chunk.sort_values("firm_id")
    header = "id | " + " | ".join(h for _, h, _ in COLUMNS)
    rows = [header]
    for r in chunk.itertuples(index=False):
        cells = [_fmt(getattr(r, c), k) for c, _, k in COLUMNS]
        rows.append(f"{r.firm_id} | " + " | ".join(cells))
    return "\n".join(rows)


def chunks(cards: pd.DataFrame, size: int = 40, seed: int = 0) -> list[pd.DataFrame]:
    """Split each date's cross-section into random chunks of about ``size`` firms."""
    out = []
    for d, g in cards.groupby("date"):
        g = g.sample(frac=1.0, random_state=seed + int(pd.Timestamp(d).strftime("%Y%m%d")))
        n = max(1, round(len(g) / size))
        edges = np.linspace(0, len(g), n + 1).astype(int)
        out.extend(g.iloc[a:b] for a, b in zip(edges[:-1], edges[1:]))
    return out


def identity_probe_prompt(row: pd.Series) -> str:
    """Ask a model to name the company from its anonymised card (memorisation probe)."""
    table = render_table(row.to_frame().T)
    return ("Here is an anonymised fact sheet of a US-listed company. If you can tell which "
            "company it is and roughly which year the figures are from, name them; otherwise "
            "answer 'unknown'.\n\n" + table)
