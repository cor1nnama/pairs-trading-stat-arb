from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd

def _load_csv_series(path: str | Path, price_col: str = "Adj Close") -> pd.Series:
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    if price_col not in df.columns:
        raise ValueError(f"CSV {path} missing column '{price_col}'. Columns: {list(df.columns)}")
    return df[price_col]

def get_prices(data: Dict) -> pd.DataFrame:
    """
    Accepts CONFIG['data'] dict and returns 2-col DataFrame with columns ['A','B'].
    Provide either tickers or csv paths (not both).
    """
    import yfinance as yf

    t1, t2 = data.get("ticker1"), data.get("ticker2")
    c1, c2 = data.get("csv1"), data.get("csv2")
    start, end = data.get("start"), data.get("end")
    price_col = data.get("price_col", "Adj Close")
    freq = data.get("freq", "B")

    use_tickers = bool(t1) and bool(t2)
    use_csvs    = bool(c1) and bool(c2)
    if use_tickers == use_csvs:   # both True or both False -> invalid
        raise ValueError("Provide either both tickers or both csv paths (not both).")

    if use_tickers:
        px_all = yf.download([t1, t2], start=start, end=end)  # auto_adjust=True
        if isinstance(px_all.columns, pd.MultiIndex):
            lvl0 = px_all.columns.get_level_values(0)
            panel = "Adj Close" if "Adj Close" in lvl0 else "Close"
            px = px_all[panel]
        else:
            panel = "Adj Close" if "Adj Close" in px_all.columns else "Close"
            px = px_all[panel]
        df = px.rename(columns={t1: "A", t2: "B"})
    else:
        s1 = _load_csv_series(c1, price_col=price_col).rename("A")
        s2 = _load_csv_series(c2, price_col=price_col).rename("B")
        df = pd.concat([s1, s2], axis=1)

    return df.sort_index().asfreq(freq).ffill().dropna()[["A","B"]]

