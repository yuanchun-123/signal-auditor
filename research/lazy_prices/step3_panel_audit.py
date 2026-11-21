"""Step 3: monthly panel, prices, controls -- then audit.

Signal at month-end t = similarity of the firm's most recent 10-K accepted by SEC
on or before t (and within the past 12 months). High similarity ("non-changers")
is predicted to earn higher future returns than low similarity ("changers").

Two versions are audited:

* ``correct`` -- the signal is dated by SEC acceptance time (``available_at``).
* ``naive``   -- the signal is dated by the fiscal period end (``reportDate``),
  a common mistake: 10-Ks are filed 60-90 days after fiscal year-end, so the
  timestamp check should FAIL.

Controls for the characteristic-neutral check: log public float (lagged so it is
already reported), 12-1 momentum and 12-month volatility. Factor spanning uses
Fama-French 5 + momentum.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))

from sec import get  # noqa: E402

from signal_auditor import audit  # noqa: E402
from signal_auditor.factors import load_french  # noqa: E402

DATA = HERE / "data"
REPORTS = HERE / "reports"
START, END = "2013-01-31", "2026-09-30"


# --------------------------------------------------------------------------- #
# Prices
# --------------------------------------------------------------------------- #
def yahoo_monthly(ticker: str) -> pd.Series:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker.replace('.', '-')}"
    params = {"interval": "1mo", "period1": 1262304000, "period2": int(time.time())}
    r = requests.get(url, params=params, headers={"User-Agent": "signal-auditor"}, timeout=30)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    idx = pd.to_datetime(res["timestamp"], unit="s").to_period("M").to_timestamp("M")
    px = res["indicators"]["adjclose"][0]["adjclose"]
    return pd.Series(px, index=idx, dtype=float).groupby(level=0).last()


def load_prices(uni: pd.DataFrame) -> pd.DataFrame:
    cache = DATA / "prices_monthly.csv"
    if cache.exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    cols = {}
    for row in uni.itertuples():
        tick = str(row.tickers).split()[0] if isinstance(row.tickers, str) and row.tickers else None
        if not tick:
            continue
        try:
            cols[row.cik] = yahoo_monthly(tick)
        except Exception as e:  # noqa: BLE001
            print(f"no prices for {tick}: {e}")
        time.sleep(0.25)
    prices = pd.DataFrame(cols).sort_index()
    this_month = pd.Timestamp.today().to_period("M").to_timestamp("M")
    prices = prices[prices.index < this_month]
    prices.columns = prices.columns.astype(str)
    prices.to_csv(cache)
    return prices


# --------------------------------------------------------------------------- #
# Size: public float by year, from XBRL frames
# --------------------------------------------------------------------------- #
def public_float_by_year(years: range) -> pd.DataFrame:
    cache = DATA / "public_float.csv"
    if cache.exists():
        return pd.read_csv(cache)
    rows = []
    for y in years:
        js = get(f"https://data.sec.gov/api/xbrl/frames/dei/EntityPublicFloat/USD/CY{y}Q2I.json").json()
        d = pd.DataFrame(js["data"])[["cik", "val"]].drop_duplicates("cik")
        d["year"] = y
        rows.append(d)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(cache, index=False)
    return out


# --------------------------------------------------------------------------- #
# Panel
# --------------------------------------------------------------------------- #
def signal_panel(sims: pd.DataFrame, dates: pd.DatetimeIndex, date_col: str,
                 measure: str) -> pd.DataFrame:
    """For each month-end, the latest filing dated (by ``date_col``) on or before it."""
    s = sims.dropna(subset=[measure]).copy()
    # acceptanceDateTime is UTC (EDGAR accepts 06:00-22:00 US Eastern; the UTC hours
    # cluster at 20-22, i.e. just after the 16:00 ET close). Convert to Eastern time.
    # Panel dates are month-end closes, so "available" means tradeable at that day's
    # close: a filing accepted at or after 16:00 ET becomes usable the next day.
    acc = (pd.to_datetime(s["acceptanceDateTime"], utc=True)
           .dt.tz_convert("America/New_York").dt.tz_localize(None))
    tradeable = acc.dt.normalize() + pd.to_timedelta((acc.dt.hour >= 16).astype(int), unit="D")
    s["available_at"] = tradeable
    if date_col == "acceptanceDateTime":
        s["known"] = tradeable
    else:
        s["known"] = pd.to_datetime(s[date_col]).dt.normalize()
    s = s.sort_values("known")
    grid = pd.DataFrame({"date": dates}).sort_values("date")
    out = []
    for cik, g in s.groupby("cik"):
        m = pd.merge_asof(grid, g[["known", "available_at", measure]], left_on="date",
                          right_on="known", direction="backward",
                          tolerance=pd.Timedelta(days=366))
        m["asset"] = str(cik)
        out.append(m.dropna(subset=[measure]))
    p = pd.concat(out, ignore_index=True).rename(columns={measure: "signal"})
    return p[["date", "asset", "signal", "available_at"]]


def add_returns_and_controls(panel: pd.DataFrame, prices: pd.DataFrame,
                             pf: pd.DataFrame) -> pd.DataFrame:
    rets = prices.pct_change(fill_method=None)
    fwd = rets.shift(-1)
    mom = prices.shift(1) / prices.shift(12) - 1
    vol = rets.rolling(12, min_periods=10).std()

    def stack(w, name):
        return w.stack().rename(name).rename_axis(["date", "asset"]).reset_index()

    out = panel.merge(stack(fwd, "fwd_ret"), on=["date", "asset"], how="inner")
    out = out.merge(stack(mom, "mom_12_1"), on=["date", "asset"], how="left")
    out = out.merge(stack(vol, "vol_12m"), on=["date", "asset"], how="left")
    # float measured mid-year y is reported in the 10-K filed early in y+1:
    # use year t-1 from April onwards, year t-2 before that
    out["float_year"] = np.where(out["date"].dt.month >= 4, out["date"].dt.year - 1,
                                 out["date"].dt.year - 2)
    pf = pf.assign(asset=pf["cik"].astype(str), log_float=np.log(pf["val"].clip(lower=1)))
    out = out.merge(pf[["asset", "year", "log_float"]], left_on=["asset", "float_year"],
                    right_on=["asset", "year"], how="left").drop(columns=["year", "float_year"])
    return out


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    uni = pd.read_csv(DATA / "universe.csv")
    sims = pd.read_csv(DATA / "similarity.csv")
    sims = sims.merge(pd.read_csv(DATA / "filings.csv")[["accessionNumber", "reportDate"]],
                      on="accessionNumber", how="left")
    prices = load_prices(uni)
    pf = public_float_by_year(range(2010, 2026))
    factors = load_french("M", cache_dir=DATA).drop(columns="RF")
    dates = pd.date_range(START, END, freq="ME")

    n_sim_firms = sims.dropna(subset=["sim_cosine"])["cik"].nunique()
    n_priced = prices.columns.isin(sims["cik"].astype(str)).sum()
    print(f"universe {len(uni)} firms | with similarity {n_sim_firms} | with prices {n_priced} "
          f"| no current ticker {(uni['tickers'].fillna('') == '').sum()}")

    controls = ["log_float", "mom_12_1", "vol_12m"]
    runs = [("correct", "acceptanceDateTime", "sim_cosine", 2),
            ("correct", "acceptanceDateTime", "sim_jaccard", 2),
            ("naive", "reportDate", "sim_cosine", 2)]
    summary = []
    for label, date_col, measure, trials in runs:
        panel = add_returns_and_controls(signal_panel(sims, dates, date_col, measure), prices, pf)
        panel = panel.dropna(subset=controls)
        name = f"Lazy Prices: {measure} ({label} dating)"
        rep = audit(panel, name=name, horizon=1, periods_per_year=12, n_trials=trials,
                    cost_bps=10, factors=factors, neutralize=controls, intended_assets=len(uni),
                    universe_note="top 500 by 2011 public float; firms without a current "
                                  "ticker lack prices")
        path = REPORTS / f"lazy_prices_{measure}_{label}.html"
        rep.to_html(str(path))
        print(rep.to_markdown())
        summary.append({"run": name, "verdict": rep.verdict, **rep.counts(),
                        "firms": panel["asset"].nunique(), "months": panel["date"].nunique()})
    pd.DataFrame(summary).to_csv(REPORTS / "summary.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
