# Z-score signals, bands
from __future__ import annotations
import numpy as np
import pandas as pd

def compute_spread(df: pd.DataFrame, beta: float) -> pd.Series:
    return (df["A"] - beta * df["B"]).rename("spread")

def zscore(series: pd.Series, lookback: int = 60) -> pd.Series:
    m = series.rolling(lookback).mean()
    s = series.rolling(lookback).std()
    z = (series - m) / s
    return z.rename("z")

def generate_signals(
    z: pd.Series,
    entry: float = 2.0,
    exit: float = 0.0,
    max_abs_z: float | None = None,
    cooldown: int = 0,
) -> pd.Series:
    """
    Returns {-1,0,+1} with hysteresis. Optional stop-loss via max_abs_z and cooldown days after exit.
    +1 = long spread (long A, short B), -1 = short spread.
    """
    z = z.copy()
    pos = np.zeros(len(z), dtype=float)
    cd = 0  # cooldown counter

    for i in range(1, len(z)):
        prev = pos[i - 1]

        # stop trading during cooldown
        if cd > 0:
            pos[i] = 0.0
            cd -= 1
            continue

        zi = z.iloc[i]
        # guard NaNs
        if np.isnan(zi):
            pos[i] = 0.0
            continue

        # optional hard stop-loss if z blows out
        if max_abs_z is not None and abs(zi) >= max_abs_z:
            pos[i] = 0.0
            cd = cooldown
            continue

        if prev == 0:
            if zi <= -entry:
                pos[i] = +1
            elif zi >= +entry:
                pos[i] = -1
            else:
                pos[i] = 0
        elif prev == +1:
            pos[i] = 0 if zi >= exit else +1
        elif prev == -1:
            pos[i] = 0 if zi <= -exit else -1

        # enter cooldown after a fresh exit
        if prev != 0 and pos[i] == 0 and cooldown > 0:
            cd = cooldown

    return pd.Series(pos, index=z.index, name="signal")
