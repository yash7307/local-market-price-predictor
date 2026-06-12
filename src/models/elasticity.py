"""
Price Elasticity Estimator
---------------------------
Measures how sensitive demand is to price changes.

Price Elasticity = (% change in demand) / (% change in price)

Interpretation:
    elasticity < -1   → Price-elastic   (demand drops sharply when price rises)
    -1 < elasticity < 0 → Price-inelastic (demand is relatively stable)
    elasticity = -1   → Unitary elastic
    elasticity > 0    → Giffen good (rare — demand rises with price)
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def point_elasticity(
    price_old: float,
    price_new: float,
    demand_old: float,
    demand_new: float,
) -> float:
    """
    Calculate point price elasticity of demand.

    Args:
        price_old:  Original price (₹)
        price_new:  New price (₹)
        demand_old: Demand at original price (units)
        demand_new: Demand at new price (units)

    Returns:
        Elasticity coefficient (float). Returns 0.0 if division by zero.
    """
    if demand_old == 0 or price_old == 0:
        return 0.0
    pct_demand = (demand_new - demand_old) / demand_old
    pct_price  = (price_new  - price_old)  / price_old
    if pct_price == 0:
        return 0.0
    return round(pct_demand / pct_price, 4)


def arc_elasticity(
    price_old: float,
    price_new: float,
    demand_old: float,
    demand_new: float,
) -> float:
    """
    Calculate arc (midpoint) price elasticity — more stable for large changes.

    Uses the midpoint formula:
        E = [(Q2-Q1)/((Q1+Q2)/2)] / [(P2-P1)/((P1+P2)/2)]
    """
    mid_q = (demand_old + demand_new) / 2
    mid_p = (price_old  + price_new)  / 2
    if mid_q == 0 or mid_p == 0:
        return 0.0
    delta_q = (demand_new - demand_old) / mid_q
    delta_p = (price_new  - price_old)  / mid_p
    if delta_p == 0:
        return 0.0
    return round(delta_q / delta_p, 4)


def classify_elasticity(elasticity: float) -> str:
    """
    Human-readable classification of an elasticity coefficient.

    Returns one of:
        "Highly Elastic", "Elastic", "Unitary Elastic",
        "Inelastic", "Perfectly Inelastic", "Giffen / Positive"
    """
    e = abs(elasticity)
    if elasticity > 0:
        return "Giffen / Positive"
    elif e > 2.0:
        return "Highly Elastic"
    elif e > 1.0:
        return "Elastic"
    elif abs(e - 1.0) < 0.05:
        return "Unitary Elastic"
    elif e > 0.0:
        return "Inelastic"
    else:
        return "Perfectly Inelastic"


def compute_elasticity_from_curve(
    curve_df: pd.DataFrame,
) -> pd.Series:
    """
    Compute point elasticity at each row of a price–demand curve dataframe.

    Args:
        curve_df: DataFrame with columns ['price', 'demand'] sorted by price.

    Returns:
        pd.Series of elasticity values (aligned with curve_df index).
    """
    prices  = curve_df["price"].values
    demands = curve_df["demand"].values
    elasticities = [0.0]  # first point has no previous
    for i in range(1, len(prices)):
        e = point_elasticity(prices[i-1], prices[i], demands[i-1], demands[i])
        elasticities.append(e)
    return pd.Series(elasticities, index=curve_df.index)


def recommend_strategy(elasticity: float, inventory: int) -> str:
    """
    Suggest a pricing action based on elasticity + inventory level.

    Returns a brief actionable recommendation string.
    """
    classification = classify_elasticity(elasticity)

    if inventory > 150:
        return "🔴 High stock — consider lowering price to clear inventory faster."

    if "Highly Elastic" in classification or "Elastic" in classification:
        return (
            "⚡ Demand is price-sensitive. "
            "Small price drops can significantly boost volume. "
            "Stay at or slightly below competitor price."
        )
    elif "Inelastic" in classification:
        return (
            "💰 Demand is stable. "
            "You can raise price modestly without losing many customers. "
            "Margin improvement opportunity."
        )
    elif "Unitary" in classification:
        return (
            "⚖️ Balanced elasticity. "
            "Revenue is relatively flat across price changes — "
            "focus on cost reduction for profit gains."
        )
    else:
        return "📊 Insufficient data to make a strong recommendation."


if __name__ == "__main__":
    # Quick demo
    e = point_elasticity(100, 110, 200, 180)
    print(f"Elasticity    : {e}")
    print(f"Classification: {classify_elasticity(e)}")
    print(f"Strategy      : {recommend_strategy(e, inventory=80)}")
