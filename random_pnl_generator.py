import numpy as np
import pandas as pd
from numpy.typing import NDArray
from numpy.random import Generator


def generate_random_cumulative_pnl(
    num_days: int,
    daily_pnl_pct_mean: float,
    daily_pnl_pct_std: float,
    start_date: str = "2025-01-01",
    seed: int | None = 0,
) -> pd.DataFrame:
    """
    Generate a random cumulative PnL percentage time series.

    Returns
    -------
    pd.DataFrame
        Index:   DatetimeIndex of business days, named "Date"
        Columns: 
            - "PnL": cumulative percentage return (float64)
    """
    rng: Generator = np.random.default_rng(seed)

    # Daily percentage returns from normal distribution (e.g. 0.05 => 0.05%)
    daily_pnl_pct: NDArray[np.floating] = (
        rng.normal(
            loc=daily_pnl_pct_mean,
            scale=daily_pnl_pct_std,
            size=num_days,
        )
        / 100.0
    )

    # Business day index
    dates: pd.DatetimeIndex = pd.bdate_range(
        start=start_date,
        periods=num_days,
        freq="B",
    )

    # Build DataFrame: cumulative sum of daily pct
    pnl_df: pd.DataFrame = pd.DataFrame(
        data={"PnL": daily_pnl_pct.cumsum()},
        index=dates,
    ).astype("float64")

    pnl_df.index.name = "Date"

    return pnl_df


# Example usage
if __name__ == "__main__":
    pnl = generate_random_cumulative_pnl(
        num_days=2609,
        daily_pnl_pct_mean=0.5,
        daily_pnl_pct_std=3.0,
        start_date="2015-01-01",
        seed=43,
    )

    print(pnl)
    print(type(pnl))
    print("\nFinal cumulative PnL %:", f"{pnl['PnL'].iloc[-1]:.2%}")
