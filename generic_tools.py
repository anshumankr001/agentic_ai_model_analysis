import json
import pandas as pd
from random_pnl_generator import generate_random_cumulative_pnl

INITIAL_CAPITAL = 100_000.0

def _ensure_pnl_data() -> pd.DataFrame:
    return generate_random_cumulative_pnl(
        num_days=252 * 3,
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

# Example usage
if __name__ == "__main__":
    print(_ensure_pnl_data())
    print(generate_top_level_summary())