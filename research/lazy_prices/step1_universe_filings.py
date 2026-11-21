"""Step 1: universe and 10-K filing list.

Universe = the 500 largest SEC registrants by public float as of mid-2011
(dei:EntityPublicFloat, XBRL frame CY2011Q2I). Choosing firms on 2011 information --
not on today's index membership -- avoids selecting on survival to 2026. Firms
that later delisted stay in the filing sample; they can only drop out of the
*price* sample, and step 3 reports how many do.

Outputs data/universe.csv and data/filings.csv.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from sec import get

DATA = Path(__file__).parent / "data"
N_FIRMS = 500
FORMS = {"10-K", "10-K405"}
START, END = "2010-01-01", "2026-11-01"


def universe() -> pd.DataFrame:
    js = get("https://data.sec.gov/api/xbrl/frames/dei/EntityPublicFloat/USD/CY2011Q2I.json").json()
    df = pd.DataFrame(js["data"]).rename(columns={"val": "public_float_2011"})
    df = df.sort_values("public_float_2011", ascending=False).drop_duplicates("cik")
    return df.head(N_FIRMS)[["cik", "entityName", "public_float_2011"]].reset_index(drop=True)


def _rows(block: dict) -> pd.DataFrame:
    keep = ["accessionNumber", "filingDate", "reportDate", "acceptanceDateTime", "form", "primaryDocument"]
    return pd.DataFrame({k: block.get(k, []) for k in keep})


def filings_for(cik: int) -> tuple[pd.DataFrame, dict]:
    js = get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()
    frames = [_rows(js["filings"]["recent"])]
    for f in js["filings"].get("files", []):
        if f.get("filingTo", "9999") >= START:
            frames.append(_rows(get(f"https://data.sec.gov/submissions/{f['name']}").json()))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["form"].isin(FORMS) & df["filingDate"].between(START, END)].copy()
    df["cik"] = cik
    meta = {"cik": cik, "tickers": " ".join(js.get("tickers", [])),
            "exchanges": " ".join(x or "" for x in js.get("exchanges", [])),
            "sic": js.get("sic"), "name_now": js.get("name")}
    return df, meta


def main() -> None:
    DATA.mkdir(exist_ok=True)
    uni = universe()
    with ThreadPoolExecutor(4) as ex:
        out = list(ex.map(filings_for, uni["cik"]))
    filings = pd.concat([o[0] for o in out], ignore_index=True)
    meta = pd.DataFrame([o[1] for o in out])
    uni = uni.merge(meta, on="cik", how="left")
    uni.to_csv(DATA / "universe.csv", index=False)
    filings.to_csv(DATA / "filings.csv", index=False)
    print(f"{len(uni)} firms, {len(filings)} 10-K filings, "
          f"{(uni['tickers'].fillna('') == '').sum()} firms with no current ticker")


if __name__ == "__main__":
    main()
