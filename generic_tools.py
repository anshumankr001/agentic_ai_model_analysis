import json
from typing import Any, Dict
from random_pnl_generator import generate_random_cumulative_pnl

import numpy as np
import pandas as pd

Number = float | int
JSONType = dict[str, str | Number | dict]

TRADING_DAYS_PER_YEAR = 261

def _ensure_pnl_data() -> pd.DataFrame:
    return generate_random_cumulative_pnl(
        num_days=2609,
        daily_pnl_pct_mean=0.5,
        daily_pnl_pct_std=4.0,
        start_date="2015-01-01",
        seed=43,
    )

def generate_top_level_summary(
    pnl_data: pd.DataFrame | None = None,
    pnl_column: str = "PnL",
    initial_capital: float = 100_000.0,
    risk_free_rate: float = 0.02,
) -> JSONType:

    if pnl_data is None:
        pnl_data = _ensure_pnl_data()

    pnl_series = pnl_data[pnl_column].astype(float).sort_index()

    trading_days: int = len(pnl_series)
    if trading_days == 0:
        raise ValueError("PnL series is empty.")

    years: float = trading_days / TRADING_DAYS_PER_YEAR

    # Index is DatetimeIndex from generate_random_cumulative_pnl
    start_date: str = str(pd.Timestamp(pnl_series.index[0]).date())  # type: ignore[arg-type]
    end_date: str = str(pd.Timestamp(pnl_series.index[-1]).date())  # type: ignore[arg-type]

    total_return_pct: float = float(pnl_series.iloc[-1] - pnl_series.iloc[0])
    annualized_return_pct: float = (
        total_return_pct / years if years > 0 else float("nan")
    )

    daily_pct = pnl_series.diff().fillna(pnl_series.iloc[0])
    daily_values = daily_pct.to_numpy(dtype=float)

    annualized_vol_pct: float = float(
        daily_values.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    )

    excess_daily = daily_values - (risk_free_rate / TRADING_DAYS_PER_YEAR * 100.0)
    sharpe_ratio: float = (
        float(excess_daily.mean() / excess_daily.std(ddof=1))
        if excess_daily.std(ddof=1) > 0
        else float("nan")
    )

    running_max = pnl_series.cummax()
    drawdowns = pnl_series - running_max
    max_drawdown_pct: float = float(drawdowns.min())

    wins: int = int((daily_values > 0).sum())
    win_rate_pct: float = float(wins / trading_days * 100.0)

    best_day_pct: float = float(daily_pct.max())
    worst_day_pct: float = float(daily_pct.min())

    skewness: float = float(daily_pct.skew())
    kurtosis: float = float(daily_pct.kurtosis())

    final_value: float = float(initial_capital * (1.0 + total_return_pct / 100.0))
    total_pnl_amount: float = float(final_value - initial_capital)

    return {
        "backtest_period": {
            "trading_days": trading_days,
            "years": round(years, 2),
            "start_date": start_date,
            "end_date": end_date,
        },
        "capital": {
            "initial_capital": initial_capital,
            "final_value": round(final_value, 2),
            "total_pnl_amount": round(total_pnl_amount, 2),
            "total_return_pct": round(total_return_pct, 2),
        },
        "performance": {
            "annualized_return_pct": round(annualized_return_pct, 2),
            "annualized_volatility_pct": round(annualized_vol_pct, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
        },
        "risk": {
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "win_rate_pct": round(win_rate_pct, 2),
        },
        "daily_pnl": {
            "best_day_pct": round(best_day_pct, 2),
            "worst_day_pct": round(worst_day_pct, 2),
        },
        "distribution": {
            "skewness": round(skewness, 2),
            "kurtosis": round(kurtosis, 2),
        },
    }

# Example usage
if __name__ == "__main__":
    print(_ensure_pnl_data())
    print(generate_top_level_summary())