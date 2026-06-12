"""
Demand Forecasting Model
-------------------------
Uses XGBoost to predict units sold given pricing & contextual features.
Train this model first — the price optimizer depends on it.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

MODEL_PATH = Path(__file__).parents[2] / "data" / "processed" / "demand_model.pkl"

FEATURES = [
    "price", "competitor_price", "is_weekend",
    "is_festival", "inventory", "month", "day_of_week", "temperature",
]
TARGET = "demand"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-derived features to the dataframe."""
    df = df.copy()
    df["date"]        = pd.to_datetime(df["date"])
    df["month"]       = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    return df


def train(df: pd.DataFrame) -> XGBRegressor:
    """
    Train an XGBoost demand forecasting model.

    Steps:
        1. Feature engineering
        2. Chronological train/test split (no data leakage)
        3. Model training
        4. Evaluation (MAE, R²)
        5. Save model to disk

    TODO:
        - Add hyperparameter tuning with Optuna.
        - Add LightGBM as a competing model and compare.
        - Add SHAP feature importance analysis.
    """
    df = build_features(df)

    # Chronological split — never shuffle time series data
    split_idx = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test,  y_test  = test_df[FEATURES],  test_df[TARGET]

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    preds = model.predict(X_test)
    print(f"MAE : {mean_absolute_error(y_test, preds):.2f}")
    print(f"R²  : {r2_score(y_test, preds):.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"[OK] Model saved -> {MODEL_PATH}")
    return model


def load_model() -> XGBRegressor:
    """Load the trained model from disk."""
    return joblib.load(MODEL_PATH)


def predict_demand(model, input_row: dict) -> float:
    """
    Predict demand for a single pricing scenario.

    Args:
        model: Trained XGBRegressor.
        input_row: dict with keys matching FEATURES.

    Returns:
        Predicted demand (units).
    """
    X = pd.DataFrame([input_row])[FEATURES]
    return float(model.predict(X)[0])


if __name__ == "__main__":
    from src.data_collection.data_simulator import simulate_sales
    df = simulate_sales()
    train(df)
