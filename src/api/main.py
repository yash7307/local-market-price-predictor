"""
Dynamic Pricing Engine — FastAPI Backend
-----------------------------------------
Run: uvicorn src.api.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

app = FastAPI(
    title="PriceIQ — Dynamic Pricing Engine API",
    description="ML-powered price optimization for small businesses.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_demand_model = None


def get_model():
    global _demand_model
    if _demand_model is None:
        from src.models.demand_model import load_model
        _demand_model = load_model()
    return _demand_model


# ── Schemas ───────────────────────────────────────────────────────────────────

class PricingRequest(BaseModel):
    competitor_price: float = Field(..., example=190.0)
    inventory:        int   = Field(..., example=80)
    is_weekend:       int   = Field(0,   example=1)
    is_festival:      int   = Field(0,   example=0)
    month:            int   = Field(..., example=5)
    day_of_week:      int   = Field(..., example=6)
    temperature:      float = Field(30.0, example=32.0)
    price_min:        float = Field(50.0)
    price_max:        float = Field(500.0)


class PricingResponse(BaseModel):
    optimal_price:    float
    expected_demand:  float
    expected_revenue: float
    confidence:       str


class ElasticityResponse(BaseModel):
    elasticity:      float
    classification:  str
    recommendation:  str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "ok",
        "model_loaded": _demand_model is not None,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/price/optimize", response_model=PricingResponse, tags=["Pricing"])
def optimize_price(req: PricingRequest):
    """Returns the revenue-maximizing suggested price."""
    try:
        from src.optimizer.price_optimizer import get_optimal_price
        model  = get_model()
        result = get_optimal_price(
            model,
            context={
                "competitor_price": req.competitor_price,
                "is_weekend":       req.is_weekend,
                "is_festival":      req.is_festival,
                "inventory":        req.inventory,
                "month":            req.month,
                "day_of_week":      req.day_of_week,
                "temperature":      req.temperature,
            },
            price_min=req.price_min,
            price_max=req.price_max,
        )
        return result
    except FileNotFoundError:
        raise HTTPException(503, "Model not trained. Run train_pipeline.py first.")


@app.get("/price/revenue-curve", tags=["Pricing"])
def revenue_curve(
    competitor_price: float = 190.0,
    inventory:        int   = 100,
    is_weekend:       int   = 0,
    is_festival:      int   = 0,
    month:            int   = 6,
    day_of_week:      int   = 0,
    temperature:      float = 30.0,
    price_min:        float = 50.0,
    price_max:        float = 500.0,
    points:           int   = 60,
):
    """Returns price vs revenue curve data for chart rendering."""
    try:
        from src.models.demand_model import FEATURES
        model = get_model()
        ctx = dict(competitor_price=competitor_price, is_weekend=is_weekend,
                   is_festival=is_festival, inventory=inventory, month=month,
                   day_of_week=day_of_week, temperature=temperature)
        rows = []
        for p in np.linspace(price_min, price_max, points):
            row = {**ctx, "price": p}
            d   = max(0, float(model.predict(pd.DataFrame([row])[FEATURES])[0]))
            rows.append({"price": round(p, 2), "demand": round(d, 1), "revenue": round(p*d, 2)})
        return {"curve": rows}
    except FileNotFoundError:
        raise HTTPException(503, "Model not trained.")


@app.get("/elasticity/classify", response_model=ElasticityResponse, tags=["Analysis"])
def classify_elasticity(
    price_old:  float,
    price_new:  float,
    demand_old: float,
    demand_new: float,
    inventory:  int = 100,
):
    """Returns elasticity coefficient, classification, and recommendation."""
    from src.models.elasticity import point_elasticity, classify_elasticity, recommend_strategy
    e   = point_elasticity(price_old, price_new, demand_old, demand_new)
    cls = classify_elasticity(e)
    rec = recommend_strategy(e, inventory)
    return {"elasticity": round(e, 4), "classification": cls, "recommendation": rec}


@app.get("/history", tags=["Data"])
def get_history(limit: int = 90):
    """Returns the last N days of simulated sales history."""
    path = Path(__file__).parents[2] / "data" / "raw" / "sales_data.csv"
    if not path.exists():
        raise HTTPException(404, "No sales data found. Run train_pipeline.py first.")
    df = pd.read_csv(path).tail(limit)
    df["revenue"] = (df["price"] * df["demand"]).round(2)
    return {"data": df.to_dict(orient="records"), "rows": len(df)}


@app.get("/competitor/prices", tags=["Market"])
def competitor_prices(product: str = "fmcg", num_competitors: int = 4):
    """Returns simulated competitor price data for a product category."""
    from src.data_collection.scraper import get_competitor_price
    return get_competitor_price(product, num_competitors=num_competitors)
