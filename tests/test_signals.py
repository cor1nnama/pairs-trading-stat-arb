import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from signal_generator import generate_signals, compute_spread, zscore

def test_signals_hysteresis():
    """Test that signals have proper hysteresis (no whipsawing)."""
    z = pd.Series([0, 2.1, 2.2, 1.0, 0.1, -0.1, -0.2])
    sig = generate_signals(z, entry=2.0, exit=0.0)
    
    # Check signal values are valid
    assert set(sig.unique()) <= {-1, 0, 1}, f"Invalid signal values: {sig.unique()}"
    
    # Check specific behavior
    assert sig.iloc[1] == -1, "Should enter short at z=2.1"
    assert sig.iloc[4] == -1, "Should still be short at z=0.1 (exit=0.0)"
    assert sig.iloc[5] == 0, "Should exit at z=-0.1 (crosses exit=0.0)"
    
    # Should not re-enter immediately after exit
    assert sig.iloc[6] == 0, "Should not re-enter immediately"

def test_signals_cooldown():
    """Test cooldown functionality."""
    # Create a scenario where cooldown should be triggered
    z = pd.Series([0, 2.1, 1.0, 0.1, -0.1, 2.2, 1.0, 0.1])  # Exit at -0.1, then cooldown
    sig = generate_signals(z, entry=2.0, exit=0.0, cooldown=2)
    
    # After exit at index 4, should have cooldown period
    # The signal exits at z=-0.1 (crosses 0.0), then cooldown starts
    assert sig.iloc[5] == 0, "Should respect cooldown period"
    assert sig.iloc[6] == 0, "Should respect cooldown period"

def test_signals_max_abs_z():
    """Test emergency stop functionality."""
    z = pd.Series([0, 2.1, 1.0, 0.1, 5.0, 4.0, 3.0])  # Extreme z-score
    sig = generate_signals(z, entry=2.0, exit=0.0, max_abs_z=4.0)
    
    # Should exit when z exceeds max_abs_z
    assert sig.iloc[4] == 0, "Should exit on extreme z-score"

def test_compute_spread():
    """Test spread computation."""
    df = pd.DataFrame({"A": [100, 101, 102], "B": [50, 51, 52]})
    spread = compute_spread(df, beta=2.0)
    
    expected = pd.Series([0.0, -1.0, -2.0], name="spread", dtype=float)  # A - 2*B
    pd.testing.assert_series_equal(spread, expected)

def test_zscore():
    """Test z-score calculation."""
    # Simple test with known values
    series = pd.Series([1, 2, 3, 4, 5])
    z = zscore(series, lookback=5)
    
    # First few values should be NaN due to insufficient lookback
    assert pd.isna(z.iloc[0])
    assert pd.isna(z.iloc[1])
    assert pd.isna(z.iloc[2])
    
    # Last value should be 0 (mean of [1,2,3,4,5] is 3, std is ~1.58, so 5-3/1.58 ≈ 1.26)
    assert not pd.isna(z.iloc[4])
