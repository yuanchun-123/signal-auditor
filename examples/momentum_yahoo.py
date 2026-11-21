"""Real-data example: 12-1 month momentum on a hand-picked list of large US stocks.

    python examples/momentum_yahoo.py

The universe is *today's* large caps, chosen with hindsight, so the auditor should
flag survivorship bias -- that is the point of the example. Prices come from
Yahoo Finance's public chart endpoint (adjusted close, monthly) and are cached in
``examples/data/``.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

from signal_auditor import audit, forward_returns, to_long
from signal_auditor.factors import load_french

TICKERS = """AAPL MSFT AMZN GOOGL META NVDA JPM BAC WFC C GS MS XOM CVX COP JNJ PFE MRK ABT
LLY UNH PG KO PEP WMT HD LOW MCD NKE SBUX DIS CMCSA VZ T INTC CSCO ORCL IBM QCOM TXN
ADBE CRM AMGN GILD BMY MDT HON GE CAT DE MMM BA LMT RTX UPS FDX UNP NEE DUK SO AMT""".split()

HERE = Path(__file__).parent
CACHE = HERE / "data" / "monthly_adjclose.csv"


def fetch(ticker: str) -> pd.Series:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"interval": "1mo", "period1": 946684800, "period2": int(time.time())}
    r = requests.get(url, params=params, headers={"User-Agent": "signal-auditor"}, timeout=30)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    idx = pd.to_datetime(res["timestamp"], unit="s").to_period("M").to_timestamp("M")
    px = res["indicators"]["adjclose"][0]["adjclose"]
    return pd.Series(px, index=idx, name=ticker).groupby(level=0).last()


def load_prices() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_csv(CACHE, index_col=0, parse_dates=True)
    CACHE.parent.mkdir(exist_ok=True)
    cols = []
    for t in TICKERS:
        try:
            cols.append(fetch(t))
        except Exception as e:  # noqa: BLE001
            print(f"skip {t}: {e}")
        time.sleep(0.3)
    prices = pd.concat(cols, axis=1).sort_index()
    prices = prices[prices.index < pd.Timestamp.today().to_period("M").to_timestamp("M")]
    prices.to_csv(CACHE)
    return prices


def main() -> None:
    prices = load_prices()
    # 12-1 momentum: return from t-12 to t-1, skipping the most recent month
    mom = prices.shift(1) / prices.shift(12) - 1.0
    panel = to_long(mom, forward_returns(prices, 1))
    factors = load_french("M", cache_dir=HERE / "data").drop(columns="RF")
    rep = audit(panel, name="12-1 momentum, 61 current US large caps", horizon=1,
                periods_per_year=12, n_trials=1, cost_bps=5, min_assets=20,
                universe_note="hand-picked current large caps (hindsight universe)",
                factors=factors)
    out = HERE / "reports" / "momentum_yahoo.html"
    out.parent.mkdir(exist_ok=True)
    rep.to_html(str(out))
    print(rep.to_markdown())
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
