# Engle–Granger / ADF test
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

@dataclass
class CointResult:
    beta: float
    pval: float
    adf_stat: float
    r2: float
    resid_std: float

def hedge_ratio_ols(df: pd.DataFrame) -> float:
    """
    Regress A on B (A ~ alpha + beta*B); return beta as hedge ratio.
    """
    A = df["A"].values
    B = sm.add_constant(df["B"].values)
    res = sm.OLS(A, B).fit()
    return float(res.params[1])

def engle_granger(df: pd.DataFrame) -> CointResult:
    """
    2-step Engle–Granger:
    1) OLS A~B -> residuals
    2) ADF test on residuals
    """
    A = df["A"]
    B = sm.add_constant(df["B"])
    ols = sm.OLS(A, B).fit()
    resid = ols.resid
    adf_stat, pval, *_ = adfuller(resid, autolag="AIC")
    return CointResult(
        beta=float(ols.params[1]),
        pval=float(pval),
        adf_stat=float(adf_stat),
        r2=float(ols.rsquared),
        resid_std=float(resid.std(ddof=1)),
    )

def scan_pairs_for_coint(prices_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Given a wide DataFrame of prices with ticker columns, compute EG p-values
    for every pair (i,j). Returns sorted DataFrame (best=lowest p-value).
    """
    cols = list(prices_wide.columns)
    out = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            Aname, Bname = cols[i], cols[j]
            df = prices_wide[[Aname, Bname]].dropna().rename(columns={Aname: "A", Bname: "B"})
            if len(df) < 50:
                continue
            res = engle_granger(df)
            out.append({
                "A": Aname, "B": Bname,
                "pval": res.pval, "beta": res.beta, "r2": res.r2, "adf_stat": res.adf_stat
            })
    return pd.DataFrame(out).sort_values("pval", ascending=True).reset_index(drop=True)
