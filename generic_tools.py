import json
import math
import pandas as pd
from random_pnl_generator import generate_random_cumulative_pnl
from scipy.stats import kurtosis, skew

INITIAL_CAPITAL = 100_000.0
TRADING_DAYS_PER_YEAR = 252

def _ensure_pnl_data() -> pd.DataFrame:
    return generate_random_cumulative_pnl(
        num_days=2609,
        daily_pnl_pct_mean=0.05,
        daily_pnl_pct_std=2.0,
        start_date="2015-01-01",
        seed=43,
    )


def generate_top_level_summary(
    pnl_data: pd.DataFrame | None = None,
    pnl_column: str = "PnL",
    initial_capital: float = INITIAL_CAPITAL,
    risk_free_rate: float = 0.02,
) -> str:
    if pnl_data is None:
        pnl_data = _ensure_pnl_data()

    if pnl_column not in pnl_data.columns:
        raise ValueError(f"Column '{pnl_column}' not found in pnl_data.")

    cumulative_returns = pnl_data[pnl_column].astype(float)
    trading_days = int(len(cumulative_returns))
    if trading_days == 0:
        raise ValueError("pnl_data is empty.")

    daily_returns = cumulative_returns.diff().fillna(cumulative_returns.iloc[0])
    years = trading_days / 252.0

    total_return_pct = float(cumulative_returns.iloc[-1] * 100.0)

    portfolio_values: pd.Series = (initial_capital * (1.0 + cumulative_returns))  # type: ignore[assignment]
    final_value = float(portfolio_values.iloc[-1])
    total_pnl = float(final_value - initial_capital)

    mean_daily_return = float(daily_returns.mean())
    vol_daily = float(daily_returns.std(ddof=1))
    annualized_return = float(((1.0 + mean_daily_return) ** 252.0 - 1.0) * 100.0)
    annualized_vol = float(vol_daily * (252.0 ** 0.5) * 100.0)

    rf_daily = risk_free_rate / 252.0
    excess_daily = daily_returns - rf_daily
    excess_vol = float(excess_daily.std(ddof=1))
    sharpe_ratio = float(
        (excess_daily.mean() / excess_vol * (252.0 ** 0.5)) if excess_vol > 0 else 0.0
    )

    running_max = portfolio_values.cummax()
    drawdowns = portfolio_values / running_max - 1.0
    max_drawdown_pct = float(drawdowns.min() * 100.0)

    positive_days = int((daily_returns > 0.0).sum())
    win_rate_pct = float(positive_days / trading_days * 100.0)

    best_day_pct = float(daily_returns.max() * 100.0)
    worst_day_pct = float(daily_returns.min() * 100.0)

    skew_value = float(daily_returns.skew())
    kurtosis_value = float(daily_returns.kurtosis())

    # Index is DatetimeIndex from generate_random_cumulative_pnl
    start_date = str(pd.Timestamp(pnl_data.index[0]).date())  # type: ignore[arg-type]
    end_date = str(pd.Timestamp(pnl_data.index[-1]).date())  # type: ignore[arg-type]

    stats: dict[str, object] = {
        "backtest_period": {
            "trading_days": trading_days,
            "years": round(years, 2),
            "start_date": start_date,
            "end_date": end_date,
        },
        "capital": {
            "initial_capital": initial_capital,
            "final_value": round(final_value, 2),
            "total_pnl": round(total_pnl, 2),
        },
        "performance": {
            "total_return_pct": round(total_return_pct, 2),
            "annualized_return_pct": round(annualized_return, 2),
            "annualized_volatility_pct": round(annualized_vol, 2),
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
            "skewness": round(skew_value, 2),
            "kurtosis": round(kurtosis_value, 2),
        },
    }

    return json.dumps(stats, indent=2)

def get_periodic_performance_summary(
    pnl_data: pd.DataFrame | None = None,
    pnl_column: str = "PnL",
    period: str = "YE",
    initial_capital: float = INITIAL_CAPITAL,
    risk_free_rate: float = 0.02,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
) -> str:
    """
    Generates a detailed periodic performance summary and returns it as a JSON string.
    """
    if pnl_data is None:
        pnl_data = _ensure_pnl_data()

    if pnl_column not in pnl_data.columns:
        raise ValueError(f"Column '{pnl_column}' not found in pnl_data.")

    pnl_series = pnl_data[pnl_column].astype(float)
    daily_returns = pnl_series.diff().fillna(pnl_series.iloc[0])

    grouped = daily_returns.resample(period)
    periods: list[dict[str, object]] = []

    for _, period_returns in grouped:
        if period_returns.empty:
            continue

        trading_days = int(period_returns.size)
        if trading_days == 0:
            continue

        years = trading_days / float(trading_days_per_year) if trading_days_per_year > 0 else 0.0

        cumulative_returns = period_returns.cumsum()
        total_return = float(cumulative_returns.iloc[-1])

        daily_mean = float(period_returns.mean())
        daily_vol = float(period_returns.std(ddof=1)) if trading_days > 1 else 0.0

        annualized_return = daily_mean * float(trading_days_per_year)
        annualized_vol = daily_vol * math.sqrt(float(trading_days_per_year)) if trading_days_per_year > 0 else 0.0

        sharpe_ratio = (
            (annualized_return - risk_free_rate) / annualized_vol if annualized_vol > 0.0 else 0.0
        )

        running_max = cumulative_returns.cummax()
        drawdown = cumulative_returns - running_max
        max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

        win_rate = float((period_returns > 0.0).sum()) / float(trading_days)

        best_day_pnl = float(period_returns.max())
        worst_day_pnl = float(period_returns.min())

        skew_value = float(skew(period_returns, bias=False)) if trading_days > 2 else 0.0
        kurtosis_value = float(kurtosis(period_returns, fisher=True, bias=False)) if trading_days > 3 else 0.0

        period_stats: dict[str, object] = {
            "backtest_period": {
                "trading_days": trading_days,
                "years": round(years, 4),
                "start_date": str(pd.Timestamp(period_returns.index[0]).date()),  # type: ignore[arg-type]
                "end_date": str(pd.Timestamp(period_returns.index[-1]).date()),  # type: ignore[arg-type]
            },
            "capital": {
                "initial_capital": initial_capital,
                "total_pnl": round(total_return, 6),
            },
            "performance": {
                "total_return": round(total_return, 6),
                "annualized_return": round(annualized_return, 6),
                "annualized_volatility": round(annualized_vol, 6),
                "sharpe_ratio": round(sharpe_ratio, 4),
            },
            "risk": {
                "max_drawdown": round(max_drawdown, 6),
                "win_rate": round(win_rate, 4),
            },
            "daily_pnl": {
                "best_day_pnl": round(best_day_pnl, 6),
                "worst_day_pnl": round(worst_day_pnl, 6),
            },
            "distribution": {
                "skewness": round(skew_value, 4),
                "kurtosis": round(kurtosis_value, 4),
            },
        }

        periods.append(period_stats)

    result: dict[str, object] = {
        "resample_period": period,
        "periods": periods,
    }

    return json.dumps(result, indent=4)

# Example usage
if __name__ == "__main__":
    print(get_periodic_performance_summary())