"""
Training Pipeline
------------------
One-command setup: simulates data → trains demand model → prints metrics.

Run from project root:
    python train_pipeline.py

Outputs:
    data/raw/sales_data.csv
    data/processed/demand_model.pkl
"""

import sys
from pathlib import Path

# Allow src imports
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 55)
print("  Dynamic Pricing Engine - Training Pipeline")
print("=" * 55)

# Step 1: Simulate Data
print("\n[1/2] Generating synthetic sales data...")
from src.data_collection.data_simulator import simulate_sales, OUTPUT_PATH

df = simulate_sales()
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)
print(f"  [OK] {len(df)} rows -> {OUTPUT_PATH}")
print(f"  Sample:\n{df.head(3).to_string(index=False)}\n")

# Step 2: Train Demand Model
print("[2/2] Training XGBoost demand model...")
from src.models.demand_model import train

model = train(df)
print("  [OK] Model saved to data/processed/demand_model.pkl")

print("\n" + "=" * 55)
print("  [DONE] Pipeline complete! Ready to run dashboard:")
print("         streamlit run dashboard/app.py")
print("=" * 55)
