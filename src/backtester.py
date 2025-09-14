# Trade simulation with PnL & costs
from __future__ import annotations
from typing import Dict, Optional
import numpy as np
import pandas as pd

class PairsBacktester:
    def __init__(
        self,
        tc_bps: float = 1.0,
        slippage_bps: float = 0.5,
        short_borrow_apr: float = 0.02,
        capital: float = 1_000_000.0,
        signal_delay: int = 1,
        periods_per_year: int = 252,
        max_gross: Optional[float] = None,  # cap on gross exposure in dollars
    ):
        self.tc_bps = float(tc_bps)
        self.slippage_bps = float(slippage_bps)
        self.short_borrow_apr = float(short_borrow_apr)
        self.capital = float(capital)
        self.signal_delay = int(signal_delay)
        self.ppy = int(periods_per_year)
        self.max_gross = max_gross

    def _shares(self, prices: pd.DataFrame, signal: pd.Series, beta: float) -> pd.DataFrame:
        prices = prices[["A", "B"]].copy()
        sig = signal.reindex(prices.index).fillna(0.0)
        if self.signal_delay > 0:
            sig = sig.shift(self.signal_delay).fillna(0.0)

        gross_leg = self.capital * 0.5  # dollars per leg
        A_shares = (gross_leg / prices["A"]) * sig
        B_shares = -(gross_leg / prices["B"]) * beta * sig

        shares = pd.DataFrame({"A_shares": A_shares, "B_shares": B_shares})
        # optional gross cap
        if self.max_gross is not None:
            gross_now = (shares.abs() * prices.to_frame()).sum(axis=1)
            scale = np.minimum(1.0, self.max_gross / np.maximum(gross_now, 1e-9))
            shares = shares.mul(scale, axis=0)
        return shares

    def simulate(self, prices: pd.DataFrame, signal: pd.Series, beta: float) -> Dict[str, pd.Series]:
        prices = prices.dropna().copy()
        shares = self._shares(prices, signal, beta)
        trades = shares.diff().fillna(0.0).rename(columns={"A_shares": "A_trades", "B_shares": "B_trades"})

        dA = prices["A"].diff().fillna(0.0)
        dB = prices["B"].diff().fillna(0.0)

        # position PnL uses previous-day shares
        pos_prev = shares.shift(1).fillna(0.0)
        pnl_pos = pos_prev["A_shares"] * dA + pos_prev["B_shares"] * dB

        # trading costs on traded notional
        traded_notional = (trades.abs().assign(
            A=lambda x: x["A_trades"] * prices["A"],
            B=lambda x: x["B_trades"] * prices["B"]
        )[["A", "B"]].sum(axis=1))
        cost_rate = (self.tc_bps + self.slippage_bps) / 10_000.0
        costs = traded_notional * cost_rate

        # borrow fee on short leg notional (assume B is short when signal>0)
        short_notional = (pos_prev["B_shares"].clip(upper=0).abs() * prices["B"])
        borrow_daily = short_notional * (self.short_borrow_apr / self.ppy)

        pnl = pnl_pos - costs - borrow_daily
        equity = pnl.cumsum()

        out = {
            "pnl": pnl.rename("pnl"),
            "equity": equity.rename("equity"),
            "costs": costs.rename("costs"),
            "borrow": borrow_daily.rename("borrow_fee"),
            "A_shares": shares["A_shares"],
            "B_shares": shares["B_shares"],
            "A_trades": trades["A_trades"],
            "B_trades": trades["B_trades"],
        }
        return out
