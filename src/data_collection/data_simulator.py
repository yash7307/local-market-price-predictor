"""
Data Simulator
--------------
Generates a realistic synthetic dataset for 2 years of daily sales.
Use this when real historical data is unavailable.
Run this first to create data/raw/sales_data.csv.
"""

import numpy as np
import pandas as pd
from pathlib import Path

OUTPUT_PATH = Path(__file__).parents[2] / "data" / "raw" / "sales_data.csv"


def simulate_sales(
    start: str = "2022-01-01",
    periods: int = 730,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate daily sales data with realistic price-demand dynamics.

    Features generated:
        date, price, competitor_price, is_weekend, is_festival,
        inventory, temperature, demand (target)

    TODO:
        - Add multi-product support (product_id column).
        - Introduce stockout days (demand > inventory → clipped sales).
        - Add weather API integration for real temperature data.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=periods, freq="D")

    price             = rng.uniform(80, 150, periods)
    competitor_price  = rng.uniform(75, 160, periods)
    inventory         = rng.integers(10, 200, periods)
    is_weekend        = (dates.dayofweek >= 5).astype(int)
    is_festival       = rng.choice([0, 1], periods, p=[0.93, 0.07])
    temperature       = 25 + 10 * np.sin(2 * np.pi * dates.dayofyear / 365) \
                        + rng.normal(0, 2, periods)   # seasonal temp curve

    # Demand model: base − price sensitivity + competitor lift + weekend/festival boost
    demand = (
        100
        - 0.5 * price
        + 0.3 * competitor_price
        + 10 * is_weekend
        + 25 * is_festival
        + rng.normal(0, 10, periods)
    ).clip(min=0)

    df = pd.DataFrame({
        "date":             dates,
        "price":            price.round(2),
        "competitor_price": competitor_price.round(2),
        "is_weekend":       is_weekend,
        "is_festival":      is_festival,
        "inventory":        inventory,
        "temperature":      temperature.round(1),
        "demand":           demand.round(0).astype(int),
    })

    return df


if __name__ == "__main__":
    df = simulate_sales()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[OK] Simulated {len(df)} rows -> {OUTPUT_PATH}")
    print(df.head())
