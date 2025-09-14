# 📈 Pairs Trading: Cointegration & Mean Reversion

This project implements a **statistical arbitrage (stat-arb) strategy** based on *pairs trading*.  
The system identifies cointegrated asset pairs, models their spread as a mean-reverting process,  
and backtests entry/exit rules with realistic costs and risk management.

## Features
- 📊 Fetch historical price data with [Yahoo Finance](https://pypi.org/project/yfinance/)
- 🔬 Cointegration testing using Engle–Granger and ADF
- 📈 Spread construction and z-score normalization
- 🎯 Signal generation with entry/exit thresholds
- 💰 Backtesting with transaction costs and risk controls
- 📊 Performance metrics: Sharpe ratio, drawdown, PnL distribution
- 🧪 Comprehensive test suite

## Quick Start

### 1. Installation
```bash
git clone https://github.com/yourusername/pairs-trading-stat-arb.git
cd pairs-trading-stat-arb
pip install -r requirements.txt
```

### 2. Run the Strategy
```bash
# Using default configuration (XOM vs CVX)
python src/main.py

# Using custom configuration
python src/main.py --config configs/example.yaml
```

### 3. Explore the Notebook
```bash
jupyter notebook notebooks/PairsTradingAnalysis.ipynb
```

## Configuration

The strategy can be configured via YAML files. See `configs/example.yaml`:

```yaml
data:
  source: yfinance    
  ticker1: XOM
  ticker2: CVX
  start: "2018-01-01"
  end: null

strategy:
  lookback: 60
  entry: 2.0
  exit: 0.0
  max_abs_z: 4.0
  cooldown: 2

execution:
  tc_bps: 1.0
  slippage_bps: 0.5
  short_borrow_apr: 0.02
  capital: 1000000
  signal_delay: 1
```

## Testing
```bash
python -m pytest tests/
```

## Example Results
```
Engle–Granger p-value: 0.0297 (ADF=-3.060, beta=0.830, R^2=0.923)
Sharpe: 0.51
Sortino: 0.49
Annual return: 2.61%
Max drawdown (USD): -108355
Hit rate: 22.15%
```

## Methodology

### 1. Cointegration Testing
- **Engle-Granger 2-step procedure**: OLS regression → ADF test on residuals
- **Hedge ratio estimation**: β from `A_t = α + βB_t + ε_t`
- **Stationarity check**: ADF p-value < 0.05 suggests cointegration

### 2. Signal Generation
- **Spread construction**: `S_t = A_t - βB_t`
- **Z-score normalization**: `Z_t = (S_t - μ) / σ` (rolling window)
- **Trading rules**: Enter when |Z| > entry threshold, exit when |Z| < exit threshold

### 3. Backtesting
- **Dollar-neutral positioning**: 50% capital per leg
- **Realistic costs**: Transaction costs + slippage + borrow fees
- **Signal delay**: Trade on next bar to avoid look-ahead bias

## ⚠️ Disclaimer

**This project is for educational purposes only.**  
It is not financial advice and should not be used in live trading.
