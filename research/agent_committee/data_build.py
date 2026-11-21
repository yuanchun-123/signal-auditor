"""Point-in-time firm features for the agent committee.

For every quarter-end rebalance date t and firm, every number is computed only
from information public at t:

* Fundamentals: annual (10-K) XBRL facts from SEC companyfacts, using the latest
  fiscal year whose 10-K was *filed* on or before t (and the year before it, for
  growth). Amendments filed later never overwrite what was known at t.
* Market value: split-unadjusted month-end close x latest cover-page shares
  outstanding filed on or before t.
* Price features: 12-1 momentum, 12-month volatility, 1-month return.

Outputs data/features.csv: one row per (date, cik).

Universe: the Lazy Prices universe (500 largest by 2010-11 public float), firms
with a current ticker -- survivorship caveat as documented there.
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "lazy_prices"))
from sec import get  # noqa: E402

DATA = HERE / "data"
LP_DATA = HERE.parent / "lazy_prices" / "data"
START, END = "2014-12-31", "2026-06-30"

CONCEPTS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "gross_profit": ["GrossProfit"],
    "op_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "rnd": ["ResearchAndDevelopmentExpense"],
    "dividends": ["PaymentsOfDividendsCommonStock", "PaymentsOfDividends"],
    "buybacks": ["PaymentsForRepurchaseOfCommonStock"],
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "equity": ["StockholdersEquity",
               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "cur_assets": ["AssetsCurrent"],
    "cur_liabilities": ["LiabilitiesCurrent"],
    "lt_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
    "receivables": ["AccountsReceivableNetCurrent"],
}
FLOWS = {"revenue", "gross_profit", "op_income", "net_income", "cfo", "capex", "rnd",
         "dividends", "buybacks"}


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def _facts_for(cik: int) -> pd.DataFrame:
    js = get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json").json()
    rows = []
    gaap = js.get("facts", {}).get("us-gaap", {})
    for item, tags in CONCEPTS.items():
        for rank, tag in enumerate(tags):
            for unit, obs in gaap.get(tag, {}).get("units", {}).items():
                if unit != "USD":
                    continue
                for o in obs:
                    if o.get("form") not in ("10-K", "10-K/A", "10-K405"):
                        continue
                    rows.append((cik, item, rank, o.get("start"), o["end"], o["val"],
                                 o["filed"], o.get("fy"), o.get("fp"), o.get("form")))
    for o in js.get("facts", {}).get("dei", {}).get("EntityCommonStockSharesOutstanding",
                                                     {}).get("units", {}).get("shares", []):
        rows.append((cik, "shares_out", 0, None, o["end"], o["val"], o["filed"],
                     o.get("fy"), o.get("fp"), o.get("form")))
    return pd.DataFrame(rows, columns=["cik", "item", "rank", "start", "end", "val", "filed",
                                       "fy", "fp", "form"])


def download_facts(ciks: list[int]) -> pd.DataFrame:
    path = DATA / "facts.csv.gz"
    if path.exists():
        return pd.read_csv(path)

    def safe(c):
        try:
            return _facts_for(c)
        except Exception as e:  # noqa: BLE001
            print(f"facts {c}: {e}")
            return pd.DataFrame()

    with ThreadPoolExecutor(4) as ex:
        out = pd.concat(list(ex.map(safe, ciks)), ignore_index=True)
    out.to_csv(path, index=False)
    return out


def _yahoo(ticker: str) -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker.replace('.', '-')}"
    params = {"interval": "1mo", "period1": 1262304000, "period2": int(time.time()),
              "events": "split"}
    r = requests.get(url, params=params, headers={"User-Agent": "signal-auditor"}, timeout=30)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    idx = pd.to_datetime(res["timestamp"], unit="s").to_period("M").to_timestamp("M")
    q = res["indicators"]
    df = pd.DataFrame({"close": q["quote"][0]["close"], "adjclose": q["adjclose"][0]["adjclose"]},
                      index=idx).groupby(level=0).last()
    # Yahoo's close is split-adjusted; undo splits to get the price actually quoted.
    splits = res.get("events", {}).get("splits", {})
    factor = pd.Series(1.0, index=df.index)
    for s in splits.values():
        when = pd.Timestamp(s["date"], unit="s").to_period("M").to_timestamp("M")
        ratio = s["numerator"] / s["denominator"]
        factor[factor.index < when] *= ratio
    df["close_unadj"] = df["close"] * factor
    return df


def download_prices(uni: pd.DataFrame) -> pd.DataFrame:
    path = DATA / "prices_split.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])
    out = []
    for row in uni.itertuples():
        tick = row.tickers.split()[0]
        try:
            d = _yahoo(tick).reset_index(names="date")
            d["cik"] = row.cik
            out.append(d)
        except Exception as e:  # noqa: BLE001
            print(f"prices {tick}: {e}")
        time.sleep(0.25)
    prices = pd.concat(out, ignore_index=True)
    this_month = pd.Timestamp.today().to_period("M").to_timestamp("M")
    prices = prices[prices["date"] < this_month]
    prices.to_csv(path, index=False)
    return prices


# --------------------------------------------------------------------------- #
# Point-in-time features
# --------------------------------------------------------------------------- #
def annual_table(facts: pd.DataFrame) -> pd.DataFrame:
    """One row per (cik, fiscal-year end, filed): first-reported annual values."""
    f = facts[facts["item"] != "shares_out"].copy()
    f["end"] = pd.to_datetime(f["end"])
    f["filed"] = pd.to_datetime(f["filed"])
    f["start"] = pd.to_datetime(f["start"])
    is_flow = f["item"].isin(FLOWS)
    dur = (f["end"] - f["start"]).dt.days
    f = f[~is_flow | dur.between(350, 380)]          # flows: full fiscal years only
    # as originally filed: earliest filing reporting that (item, period end)
    f = f.sort_values(["cik", "item", "end", "filed", "rank"])
    f = f.drop_duplicates(["cik", "item", "end"], keep="first")
    return f[["cik", "item", "end", "filed", "val"]]


def features_at(date: pd.Timestamp, ann: pd.DataFrame, shares: pd.DataFrame,
                px: pd.DataFrame) -> pd.DataFrame:
    known = ann[ann["filed"] <= date]
    # latest fiscal year end per firm, among 10-Ks already filed
    fy_end = known.groupby("cik")["end"].max().rename("fy_end")
    known = known.merge(fy_end, on="cik")
    known = known[known["end"] >= known["fy_end"] - pd.Timedelta(days=400)]
    known["lag"] = np.where(known["end"] >= known["fy_end"] - pd.Timedelta(days=30), 0, 1)
    wide = known.pivot_table(index="cik", columns=["item", "lag"], values="val", aggfunc="last")
    cur = wide.xs(0, axis=1, level="lag") if 0 in wide.columns.get_level_values(1) else None
    prev = wide.xs(1, axis=1, level="lag") if 1 in wide.columns.get_level_values(1) else None
    if cur is None:
        return pd.DataFrame()
    out = pd.DataFrame(index=cur.index)
    out["fy_end"] = fy_end
    g = cur.get

    sh = shares[shares["filed"] <= date].sort_values("filed").groupby("cik")["val"].last()
    p = px[px["date"] == date].set_index("cik")
    out["mcap"] = p["close_unadj"].reindex(out.index) * sh.reindex(out.index)

    def ratio(a, b):
        a = a if a is not None else pd.Series(np.nan, index=out.index)
        b = b if b is not None else pd.Series(np.nan, index=out.index)
        return (a / b.where(b.abs() > 0)).reindex(out.index)

    out["earnings_yield"] = ratio(g("net_income"), out["mcap"])
    out["book_to_market"] = ratio(g("equity"), out["mcap"])
    fcf = (g("cfo") - g("capex").fillna(0)) if g("cfo") is not None else None
    out["fcf_yield"] = ratio(fcf, out["mcap"])
    out["sales_to_price"] = ratio(g("revenue"), out["mcap"])
    out["roe"] = ratio(g("net_income"), g("equity"))
    out["roa"] = ratio(g("net_income"), g("assets"))
    out["gross_margin"] = ratio(g("gross_profit"), g("revenue"))
    out["op_margin"] = ratio(g("op_income"), g("revenue"))
    out["accruals_to_assets"] = ratio(g("net_income") - g("cfo") if g("cfo") is not None else None,
                                      g("assets"))
    out["cfo_to_net_income"] = ratio(g("cfo"), g("net_income"))
    out["debt_to_equity"] = ratio(g("lt_debt"), g("equity"))
    out["liabilities_to_assets"] = ratio(g("liabilities"), g("assets"))
    out["current_ratio"] = ratio(g("cur_assets"), g("cur_liabilities"))
    out["rnd_to_sales"] = ratio(g("rnd"), g("revenue"))
    payout = (g("dividends").fillna(0) if g("dividends") is not None else 0) + \
             (g("buybacks").fillna(0) if g("buybacks") is not None else 0)
    out["shareholder_yield"] = ratio(payout if not isinstance(payout, int) else None, out["mcap"])
    if prev is not None:
        pg = prev.get
        out["revenue_growth"] = ratio(g("revenue"), pg("revenue")) - 1
        out["net_income_growth"] = ratio(g("net_income") - pg("net_income"),
                                         pg("net_income").abs() if pg("net_income") is not None else None)
        out["asset_growth"] = ratio(g("assets"), pg("assets")) - 1
        if g("receivables") is not None and pg("receivables") is not None and pg("revenue") is not None:
            # Beneish DSRI: receivables-to-sales this year vs last year
            out["receivables_index"] = ratio(g("receivables") / g("revenue"),
                                             pg("receivables") / pg("revenue"))
    out["log_mcap"] = np.log(out["mcap"].where(out["mcap"] > 0))

    hist = px[px["date"] <= date].pivot(index="date", columns="cik", values="adjclose").sort_index()
    if len(hist) >= 13:
        out["mom_12_1"] = (hist.iloc[-2] / hist.iloc[-13] - 1).reindex(out.index)
        out["ret_1m"] = (hist.iloc[-1] / hist.iloc[-2] - 1).reindex(out.index)
        out["vol_12m"] = hist.pct_change(fill_method=None).iloc[-12:].std().reindex(out.index) * np.sqrt(12)
    out["months_since_fy_end"] = ((date - out["fy_end"]).dt.days / 30.4).round(1)
    out.insert(0, "date", date)
    return out.reset_index()


def forward_quarter_returns(px: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    w = px.pivot(index="date", columns="cik", values="adjclose").sort_index()
    fwd = (w.shift(-3) / w - 1).reindex(dates)
    return fwd.stack().rename("fwd_ret_3m").rename_axis(["date", "cik"]).reset_index()


SIC_SECTORS = [(100, 999, "Agriculture"), (1000, 1499, "Mining & Energy"),
               (1500, 1799, "Construction"), (2000, 3999, "Manufacturing"),
               (4000, 4899, "Transportation & Communications"), (4900, 4999, "Utilities"),
               (5000, 5199, "Wholesale"), (5200, 5999, "Retail"),
               (6000, 6799, "Finance, Insurance & Real Estate"), (7000, 8999, "Services"),
               (9100, 9999, "Public Administration")]


def sector(sic) -> str:
    try:
        s = int(sic)
    except (TypeError, ValueError):
        return "Unknown"
    if 2830 <= s <= 2836:
        return "Pharmaceuticals"
    if 3570 <= s <= 3579 or 3670 <= s <= 3679 or 7370 <= s <= 7379:
        return "Technology"
    for lo, hi, name in SIC_SECTORS:
        if lo <= s <= hi:
            return name
    return "Unknown"


def main() -> None:
    DATA.mkdir(exist_ok=True)
    uni = pd.read_csv(LP_DATA / "universe.csv")
    uni = uni[uni["tickers"].fillna("") != ""]
    facts = download_facts(uni["cik"].tolist())
    print(f"facts: {len(facts):,} rows for {facts['cik'].nunique()} firms")
    prices = download_prices(uni)
    print(f"prices: {prices['cik'].nunique()} firms")

    ann = annual_table(facts)
    shares = facts[facts["item"] == "shares_out"].assign(filed=lambda d: pd.to_datetime(d["filed"]))
    dates = pd.date_range(START, END, freq="QE")
    feats = pd.concat([features_at(d, ann, shares, prices) for d in dates], ignore_index=True)
    feats = feats.merge(forward_quarter_returns(prices, dates), on=["date", "cik"], how="left")
    feats["sector"] = feats["cik"].map(uni.set_index("cik")["sic"].map(sector))
    feats = feats[feats["mcap"].notna() & (feats["months_since_fy_end"] <= 18)]
    feats.to_csv(DATA / "features.csv", index=False)
    print(f"features: {len(feats):,} firm-quarters, {feats['cik'].nunique()} firms, "
          f"{feats['date'].nunique()} dates")
    print(feats.drop(columns=["cik"]).describe().T[["count", "50%"]].round(3).to_string())


if __name__ == "__main__":
    main()
