import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backtester import PairsBacktester

def test_backtest_shapes():
    """Test that backtester returns expected data structures."""
    idx = pd.date_range("2022-01-01", periods=252, freq="B")
    A = 100 + np.cumsum(np.random.normal(0, 1, len(idx)))
    B = 100 + np.cumsum(np.random.normal(0, 1, len(idx)))
    prices = pd.DataFrame({"A": A, "B": B}, index=idx)

    sig = pd.Series(0, index=idx)
    sig.iloc[50:100] = 1  # some trades

    bt = PairsBacktester().simulate(prices, sig, beta=1.0)
    
    # Check required keys exist
    required_keys = {"pnl", "equity", "costs", "borrow", "A_shares", "B_shares", "A_trades", "B_trades"}
    assert required_keys.issubset(bt.keys())
    
    # Check all series have same length
    for key, series in bt.items():
        assert len(series) == len(idx), f"{key} has wrong length"

def test_backtest_costs():
    """Test that costs are properly calculated."""
    idx = pd.date_range("2022-01-01", periods=10, freq="B")
    prices = pd.DataFrame({"A": [100] * 10, "B": [100] * 10}, index=idx)
    
    # Signal that trades every day
    sig = pd.Series([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], index=idx)
    
    bt = PairsBacktester(tc_bps=10.0, slippage_bps=5.0).simulate(prices, sig, beta=1.0)
    
    # Should have costs on trading days
    assert bt["costs"].sum() > 0, "No costs calculated"

def test_signal_delay():
    """Test that signal delay is properly applied."""
    idx = pd.date_range("2022-01-01", periods=5, freq="B")
    prices = pd.DataFrame({"A": [100] * 5, "B": [100] * 5}, index=idx)
    
    # Signal on day 1
    sig = pd.Series([0, 1, 0, 0, 0], index=idx)
    
    # With delay=1, should start trading on day 2
    bt = PairsBacktester(signal_delay=1).simulate(prices, sig, beta=1.0)
    
    # First day should have no position
    assert bt["A_shares"].iloc[0] == 0
    # Second day should have no position (signal from day 1, applied on day 2, but signal is 0)
    assert bt["A_shares"].iloc[1] == 0
    # Third day should have position (signal from day 2, applied on day 3, signal is 1)
    assert bt["A_shares"].iloc[2] != 0
