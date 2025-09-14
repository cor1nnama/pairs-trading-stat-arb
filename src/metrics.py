# Sharpe, drawdown, win rate
from __future__ import annotations
import numpy as np
import pandas as pd

def _ret_series(pnl: pd.Series, capital: float) -> pd.Series:
    return pnl.fillna(0.0) / float(capital)

def sharpe_ratio(pnl: pd.Series, capital: float = 1_000_000.0, periods_per_year: int = 252) -> float:
    r = _ret_series(pnl, capital)
    mu = r.mean() * periods_per_year
    sd = r.std(ddof=1) * np.sqrt(periods_per_year)
    return float(mu / sd) if sd > 0 else np.nan

def sortino_ratio(pnl: pd.Series, capital: float = 1_000_000.0, periods_per_year: int = 252) -> float:
    r = _ret_series(pnl, capital)
    downside = r[r < 0]
    dd = downside.std(ddof=1) * np.sqrt(periods_per_year)
    ann = r.mean() * periods_per_year
    return float(ann / dd) if dd > 0 else np.nan

def max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = equity - roll_max
    return float(dd.min())

def annual_return(pnl: pd.Series, capital: float = 1_000_000.0, periods_per_year: int = 252) -> float:
    r = _ret_series(pnl, capital)
    return float(r.mean() * periods_per_year)

def hit_rate(pnl: pd.Series) -> float:
    r = pnl.dropna()
    return float((r > 0).mean()) if len(r) else np.nan
