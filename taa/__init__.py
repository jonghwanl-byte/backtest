"""Hybrid-Hysteresis-TAA 검증 프레임워크."""
from .config import (
    INCEPTION, SCALAR_RULES, TICKERS, WEIGHT_SCENARIOS,
    ExecConfig, GridSpec, StrategyConfig,
)
from .data import load_cash_rate, load_prices, synthetic_prices
from .engine import buy_and_hold, run_backtest, simple_ma_filter
from .fastscan import GridScanner, grid_distribution, surface
from .metrics import annual_table, deflated_sharpe, perf_stats, vol_target_scale
from .periods import IS_OOS, MARKET_TYPES, REGIMES

__all__ = [n for n in dir() if not n.startswith("_")]
