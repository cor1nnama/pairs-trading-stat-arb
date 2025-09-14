# Run end-to-end workflow
from __future__ import annotations
import argparse, sys
import yaml
import pandas as pd
import matplotlib.pyplot as plt

# Handle both relative and absolute imports
try:
    from .data_loader import get_prices
    from .coint_test import engle_granger
    from .signal_generator import compute_spread, zscore, generate_signals
    from .backtester import PairsBacktester
    from .metrics import sharpe_ratio, sortino_ratio, max_drawdown, annual_return, hit_rate
except ImportError:
    from data_loader import get_prices
    from coint_test import engle_granger
    from signal_generator import compute_spread, zscore, generate_signals
    from backtester import PairsBacktester
    from metrics import sharpe_ratio, sortino_ratio, max_drawdown, annual_return, hit_rate

def _load_config(path: str | None) -> dict:
    if path is None:
        # sensible defaults if you don't pass a config
        return {
            "data": {"ticker1": "XOM", "ticker2": "CVX", "start": "2018-01-01", "end": None, "freq": "B"},
            "strategy": {"lookback": 60, "entry": 2.0, "exit": 0.0, "max_abs_z": 4.0, "cooldown": 2},
            "execution": {"tc_bps": 1.0, "slippage_bps": 0.5, "short_borrow_apr": 0.02, "capital": 1_000_000, "signal_delay": 1},
        }
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser(description="Pairs Trading Strategy Backtester")
    ap.add_argument("--config", required=False, help="Path to YAML config (optional)")
    args = ap.parse_args()

    cfg = _load_config(args.config)

    # Handle different config formats
    data_config = cfg["data"]
    if "source" in data_config:
        # New format with source field
        if data_config["source"] == "yfinance":
            df = get_prices(
                ticker1=data_config["ticker1"],
                ticker2=data_config["ticker2"],
                start=data_config["start"],
                end=data_config["end"],
                freq=data_config.get("freq", "B")
            )
        elif data_config["source"] == "csv":
            df = get_prices(
                csv1=data_config["csv1"],
                csv2=data_config["csv2"],
                start=data_config["start"],
                end=data_config["end"],
                price_col=data_config.get("price_col", "Adj Close"),
                freq=data_config.get("freq", "B")
            )
        else:
            raise ValueError(f"Unknown data source: {data_config['source']}")
    else:
        # Legacy format - assume yfinance
        df = get_prices(**data_config)

    eg = engle_granger(df)
    spread = compute_spread(df, eg.beta)
    z = zscore(spread, cfg["strategy"]["lookback"])
    sig = generate_signals(
        z,
        entry=cfg["strategy"]["entry"],
        exit=cfg["strategy"]["exit"],
        max_abs_z=cfg["strategy"].get("max_abs_z"),
        cooldown=cfg["strategy"].get("cooldown", 0),
    )

    bt = PairsBacktester(**cfg["execution"]).simulate(df, sig, eg.beta)

    cap = float(cfg["execution"]["capital"])
    print(f"Engle–Granger p-value: {eg.pval:.4f} (ADF={eg.adf_stat:.3f}, beta={eg.beta:.3f}, R^2={eg.r2:.3f})")
    print(f"Sharpe: {sharpe_ratio(bt['pnl'], capital=cap):.2f}")
    print(f"Sortino: {sortino_ratio(bt['pnl'], capital=cap):.2f}")
    print(f"Annual return: {annual_return(bt['pnl'], capital=cap):.2%}")
    print(f"Max drawdown (USD): {max_drawdown(bt['equity']):.0f}")
    print(f"Hit rate: {hit_rate(bt['pnl']):.2%}")

    # quick plots
    try:
        ax = bt["equity"].plot(figsize=(10, 4), title="Equity Curve")
        ax.set_xlabel("Date"); ax.set_ylabel("PnL (cumulative)")
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    sys.exit(main())
