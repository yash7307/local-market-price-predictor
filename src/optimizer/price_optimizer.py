"""
Price Optimizer
----------------
Finds the revenue-maximizing price using Scipy bounded optimization.
Depends on a trained demand model from src/models/demand_model.py.
"""

from scipy.optimize import minimize_scalar
import pandas as pd


def get_optimal_price(
    model,
    context: dict,
    price_min: float = 50.0,
    price_max: float = 500.0,
) -> dict:
    """
    Find the price that maximizes expected revenue.

    Args:
        model     : Trained XGBRegressor (demand model).
        context   : dict with all features EXCEPT price, e.g.:
                    {competitor_price, is_weekend, is_festival,
                     inventory, month, day_of_week, temperature}
        price_min : Lower bound for price search.
        price_max : Upper bound for price search.

    Returns:
        dict with keys:
            optimal_price   — suggested price (₹)
            expected_demand — predicted units at optimal price
            expected_revenue— optimal_price × expected_demand
            confidence      — placeholder for confidence score

    TODO:
        - Add inventory constraint: don't over-price when stock is high.
        - Add competitor threshold: never exceed competitor_price by > 10%.
        - Replace Scipy optimizer with a gradient-based RL policy for
          sequential decision making.
    """
    from src.models.demand_model import FEATURES

    def neg_revenue(price: float) -> float:
        row = {**context, "price": price}
        X   = pd.DataFrame([row])[FEATURES]
        demand = max(0, float(model.predict(X)[0]))
        return -(price * demand)  # minimize negative = maximize revenue

    result = minimize_scalar(neg_revenue, bounds=(price_min, price_max),
                             method="bounded")

    optimal_price    = round(result.x, 2)
    expected_demand  = round(-result.fun / optimal_price, 1) if optimal_price > 0 else 0
    expected_revenue = round(-result.fun, 2)

    return {
        "optimal_price":    optimal_price,
        "expected_demand":  expected_demand,
        "expected_revenue": expected_revenue,
        "confidence":       "medium",   # TODO: derive from model uncertainty
    }


if __name__ == "__main__":
    from src.models.demand_model import load_model

    model = load_model()
    context = {
        "competitor_price": 190,
        "is_weekend":       1,
        "is_festival":      0,
        "inventory":        80,
        "month":            5,
        "day_of_week":      6,
        "temperature":      32.0,
    }
    result = get_optimal_price(model, context)
    print(f"Optimal Price   : ₹{result['optimal_price']}")
    print(f"Expected Demand : {result['expected_demand']} units")
    print(f"Expected Revenue: ₹{result['expected_revenue']}")
