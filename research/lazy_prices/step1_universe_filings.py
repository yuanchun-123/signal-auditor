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


def _frame(concept: str, period: str) -> pd.Series:
    js = get(f"https://data.sec.gov/api/xbrl/frames/{concept}/USD/{period}.json").json()
    d = pd.DataFrame(js["data"]).drop_duplicates("cik").set_index("cik")
    return d["val"].rename(period)


def universe() -> pd.DataFrame:
    """Largest firms by public float, with guards against XBRL scaling errors.

    Self-reported dei:EntityPublicFloat contains typos (e.g. 1e18, or values
    1000x too large). We take the smaller of the 2010 and 2011 values -- a
    one-year typo cannot inflate it -- and drop firms whose float exceeds 20x
    total assets.
    """
    def year(concept: str, y: int) -> pd.Series:
        # firms measure float on different dates (fiscal years differ): pool all quarters
        out = pd.Series(dtype=float)
        for q in (2, 1, 3, 4):
            out = out.combine_first(_frame(concept, f"CY{y}Q{q}I"))
        return out

    f11 = year("dei/EntityPublicFloat", 2011).rename("f11")
    f10 = year("dei/EntityPublicFloat", 2010).rename("f10")
    assets = year("us-gaap/Assets", 2011).rename("assets")
    names = pd.concat([pd.DataFrame(get(f"https://data.sec.gov/api/xbrl/frames/dei/EntityPublicFloat/"
                                        f"USD/CY2011Q{q}I.json").json()["data"]) for q in (1, 2, 3, 4)])
    names = names.drop_duplicates("cik").set_index("cik")
    df = pd.concat([f11, f10, assets, names["entityName"]], axis=1, join="outer")
    df.index.name = "cik"
    df = df[df["f11"].notna()]
    df["public_float_2011"] = df[["f11", "f10"]].min(axis=1)
    ok = df["assets"].isna() | (df["public_float_2011"] <= 20 * df["assets"])
    df = df[ok].sort_values("public_float_2011", ascending=False).head(N_FIRMS)
    return df.reset_index()[["cik", "entityName", "public_float_2011", "f10", "f11", "assets"]]


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
