import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from coint_test import hedge_ratio_ols, engle_granger, scan_pairs_for_coint

def test_beta_close_to_true():
    """Test that OLS hedge ratio is close to true relationship."""
    rng = np.random.default_rng(0)
    B = np.linspace(100, 120, 500)
    A = 2.0 * B + rng.normal(scale=0.5, size=B.size)
    df = pd.DataFrame({"A": A, "B": B})
    beta = hedge_ratio_ols(df)
    assert 1.9 < beta < 2.1, f"Beta {beta} not close to true value 2.0"

def test_eg_pval_low_on_coint():
    """Test that Engle-Granger detects cointegration."""
    rng = np.random.default_rng(0)
    B = np.cumsum(rng.normal(size=1000))
    A = 1.5 * B + rng.normal(scale=0.2, size=1000)
    df = pd.DataFrame({"A": A, "B": B})
    res = engle_granger(df)
    assert res.pval < 0.05, f"P-value {res.pval} should be < 0.05 for cointegrated series"
    assert 1.4 < res.beta < 1.6, f"Beta {res.beta} not close to true value 1.5"

def test_eg_pval_high_on_non_coint():
    """Test that Engle-Granger rejects non-cointegrated series."""
    rng = np.random.default_rng(0)
    A = np.cumsum(rng.normal(size=1000))
    B = np.cumsum(rng.normal(size=1000))  # Independent random walks
    df = pd.DataFrame({"A": A, "B": B})
    res = engle_granger(df)
    assert res.pval > 0.05, f"P-value {res.pval} should be > 0.05 for non-cointegrated series"

def test_scan_pairs():
    """Test pair scanning functionality."""
    # Create test data with one cointegrated pair
    rng = np.random.default_rng(0)
    n = 500
    
    # Cointegrated pair
    B1 = np.cumsum(rng.normal(size=n))
    A1 = 1.5 * B1 + rng.normal(scale=0.1, size=n)
    
    # Independent series
    A2 = np.cumsum(rng.normal(size=n))
    B2 = np.cumsum(rng.normal(size=n))
    
    df = pd.DataFrame({
        "STOCK_A": A1,
        "STOCK_B": B1, 
        "STOCK_C": A2,
        "STOCK_D": B2
    })
    
    results = scan_pairs_for_coint(df)
    
    # Should find the cointegrated pair first (lowest p-value)
    assert len(results) > 0, "No pairs found"
    assert results.iloc[0]["A"] == "STOCK_A" or results.iloc[0]["B"] == "STOCK_A", "Cointegrated pair not found first"
    assert results.iloc[0]["pval"] < 0.05, "Best pair should be cointegrated"
