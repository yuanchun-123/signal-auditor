"""Known-factor data and alignment helpers.

The spanning check needs factor returns on the same calendar as the panel.
:func:`load_french` downloads the Fama-French 5 factors plus momentum from
Kenneth French's data library and returns decimal returns indexed by
month-end (monthly) or trading day (daily).
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FILES = {
    "M": ("F-F_Research_Data_5_Factors_2x3_CSV.zip", "F-F_Momentum_Factor_CSV.zip"),
    "D": ("F-F_Research_Data_5_Factors_2x3_daily_CSV.zip", "F-F_Momentum_Factor_daily_CSV.zip"),
}


def _parse_french_csv(text: str, freq: str) -> pd.DataFrame:
    """Parse the first (monthly or daily) table of a French data-library CSV."""
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines)
                 if ln.strip() and ln.split(",")[0].strip() == "" and "," in ln)
    header = [c.strip() for c in lines[start].split(",")]
    rows = []
    for ln in lines[start + 1:]:
        parts = [p.strip() for p in ln.split(",")]
        if not parts[0].isdigit():
            break
        rows.append(parts)
    df = pd.DataFrame(rows, columns=["date"] + header[1:])
    fmt = "%Y%m" if freq == "M" else "%Y%m%d"
    idx = pd.to_datetime(df.pop("date"), format=fmt)
    if freq == "M":
        idx = idx + pd.offsets.MonthEnd(0)
    df.index = idx
    df = df.apply(pd.to_numeric, errors="coerce") / 100.0
    return df.replace([-99.99 / 100, -999 / 100], np.nan)


def load_french(freq: str = "M", cache_dir: str | Path | None = None) -> pd.DataFrame:
    """Fama-French 5 factors + momentum (``Mkt-RF, SMB, HML, RMW, CMA, Mom, RF``)."""
    import requests

    freq = freq.upper()
    if freq not in FILES:
        raise ValueError("freq must be 'M' or 'D'")
    tables = []
    for fname in FILES[freq]:
        path = Path(cache_dir) / fname if cache_dir else None
        if path is not None and path.exists():
            blob = path.read_bytes()
        else:
            r = requests.get(BASE + fname, timeout=60, headers={"User-Agent": "signal-auditor"})
            r.raise_for_status()
            blob = r.content
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(blob)
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            text = z.read(z.namelist()[0]).decode("latin-1")
        tables.append(_parse_french_csv(text, freq))
    out = tables[0].join(tables[1], how="inner")
    out.columns = [c.strip() for c in out.columns]
    return out.rename(columns={"MOM": "Mom"})


def align_to_rebalances(factors: pd.DataFrame, calendar: pd.DatetimeIndex,
                        rebalance_dates: pd.DatetimeIndex, horizon: int) -> pd.DataFrame:
    """Factor returns over each holding period.

    A position formed at panel date ``D[i]`` earns the forward return over
    ``(D[i], D[i+horizon]]``, so it is matched with factor returns stamped
    ``D[i+1] .. D[i+horizon]`` (summed). ``factors`` must be indexed by the
    panel's own calendar (e.g. month-ends for a monthly panel).
    """
    cal = pd.DatetimeIndex(sorted(calendar))
    f = factors.reindex(cal)
    pos = cal.get_indexer(rebalance_dates)
    rows = []
    for p in pos:
        window = f.iloc[p + 1: p + 1 + horizon]
        rows.append(window.sum(min_count=horizon) if len(window) == horizon
                    else pd.Series(np.nan, index=f.columns))
    return pd.DataFrame(rows, index=rebalance_dates)
