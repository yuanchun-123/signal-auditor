"""Step 2: year-over-year 10-K text similarity (Cohen, Malloy & Nguyen, 2020).

For each firm, download each 10-K's primary document in filing order, reduce it to
a bag of words, and compare it with the firm's previous 10-K:

* sim_cosine  -- cosine similarity of term-frequency vectors (the paper's main measure)
* sim_jaccard -- Jaccard similarity of the word sets

Low similarity = the firm "changed" its report. Only consecutive filings 9-18
months apart are compared. Raw documents are not kept; cleaned word counts are
cached as gzip JSON so other measures can be computed later without re-downloading.

Resumable: firms already in data/similarity.csv are skipped.
"""

from __future__ import annotations

import gzip
import html
import json
import math
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from sec import get

DATA = Path(__file__).parent / "data"
CACHE = DATA / "bow"
OUT = DATA / "similarity.csv"

_IX_HEADER = re.compile(r"<ix:header>.*?</ix:header>", re.S | re.I)
_HIDDEN = re.compile(r"<div[^>]*display:\s*none[^>]*>.*?</div>", re.S | re.I)
_SCRIPT = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.S | re.I)
_TAG = re.compile(r"<[^>]+>")
_WORD = re.compile(r"[a-z]{2,}")


def bag_of_words(raw: str) -> Counter:
    t = _IX_HEADER.sub(" ", raw)
    t = _HIDDEN.sub(" ", t)
    t = _SCRIPT.sub(" ", t)
    t = html.unescape(_TAG.sub(" ", t)).lower()
    return Counter(_WORD.findall(t))


def cosine(a: Counter, b: Counter) -> float:
    if len(a) > len(b):
        a, b = b, a
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else float("nan")


def jaccard(a: Counter, b: Counter) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else float("nan")


def doc_url(row) -> str:
    acc = row.accessionNumber.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{int(row.cik)}/{acc}/{row.primaryDocument}"


def load_bow(row) -> Counter:
    path = CACHE / f"{row.accessionNumber}.json.gz"
    if path.exists():
        with gzip.open(path, "rt") as f:
            return Counter(json.load(f))
    r = get(doc_url(row))
    bow = bag_of_words(r.content.decode("utf-8", errors="ignore"))
    with gzip.open(path, "wt") as f:
        json.dump(bow, f)
    return bow


def firm_similarities(cik: int, rows: pd.DataFrame) -> list[dict]:
    rows = rows.sort_values("filingDate")
    out, prev, prev_date = [], None, None
    for row in rows.itertuples():
        try:
            bow = load_bow(row)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {row.accessionNumber}: {e}", file=sys.stderr)
            prev, prev_date = None, None
            continue
        date = pd.Timestamp(row.filingDate)
        rec = {"cik": cik, "accessionNumber": row.accessionNumber, "filingDate": row.filingDate,
               "acceptanceDateTime": row.acceptanceDateTime, "n_words": sum(bow.values()),
               "sim_cosine": float("nan"), "sim_jaccard": float("nan")}
        if prev is not None and 270 <= (date - prev_date).days <= 550 and rec["n_words"] > 2000:
            rec["sim_cosine"] = cosine(bow, prev)
            rec["sim_jaccard"] = jaccard(bow, prev)
        out.append(rec)
        prev, prev_date = bow, date
    return out


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    filings = pd.read_csv(DATA / "filings.csv")
    filings = filings[filings["primaryDocument"].notna()]
    done = set(pd.read_csv(OUT)["cik"]) if OUT.exists() else set()
    todo = [(c, g) for c, g in filings.groupby("cik") if c not in done]
    print(f"{len(todo)} firms to process ({len(done)} already done)", flush=True)
    header = not OUT.exists()
    with ThreadPoolExecutor(6) as ex:
        futs = {ex.submit(firm_similarities, c, g): c for c, g in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            recs = fut.result()
            if recs:
                pd.DataFrame(recs).to_csv(OUT, mode="a", header=header, index=False)
                header = False
            if i % 10 == 0:
                print(f"{i}/{len(todo)} firms", flush=True)


if __name__ == "__main__":
    main()
