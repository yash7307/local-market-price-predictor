# Dynamic Pricing Engine 🏷️

> ML-powered price optimization for small businesses (kirana stores, local e-commerce sellers, food stalls)

---

## Project Structure

```
dynamic-pricing-engine/
│
├── data/
│   ├── raw/              ← Simulated or real CSV sales data
│   └── processed/        ← Trained model (.pkl) saved here
│
├── notebooks/
│   ├── 01_EDA.ipynb                  ← Exploratory data analysis
│   ├── 02_demand_forecasting.ipynb   ← XGBoost model training & eval
│   ├── 03_price_elasticity.ipynb     ← Elasticity analysis
│   └── 04_rl_pricing_agent.ipynb     ← (Optional) RL agent
│
├── src/
│   ├── data_collection/
│   │   ├── scraper.py          ← Competitor price scraper
│   │   └── data_simulator.py   ← Synthetic dataset generator
│   ├── models/
│   │   ├── demand_model.py     ← XGBoost demand forecaster
│   │   ├── elasticity.py       ← Price elasticity estimator
│   │   └── rl_agent.py         ← RL pricing agent (stub)
│   ├── optimizer/
│   │   └── price_optimizer.py  ← Scipy revenue maximizer
│   └── api/
│       └── main.py             ← FastAPI service
│
├── dashboard/
│   └── app.py                  ← Streamlit UI
│
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Set up environment
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Generate training data
```bash
python -m src.data_collection.data_simulator
# → Creates data/raw/sales_data.csv
```

### 3. Train demand model
```bash
python -m src.models.demand_model
# → Saves data/processed/demand_model.pkl
```

### 4. Run the dashboard
```bash
streamlit run dashboard/app.py
```

### 5. Run the API (optional)
```bash
uvicorn src.api.main:app --reload --port 8000
# Docs → http://localhost:8000/docs
```

---

## Tech Stack

| Layer        | Tools                              |
|--------------|------------------------------------|
| ML Models    | XGBoost, LightGBM, PyTorch (LSTM)  |
| Optimization | Scipy, Stable-Baselines3 (RL)      |
| Backend API  | FastAPI + Uvicorn                  |
| Dashboard    | Streamlit + Plotly                 |
| Data         | Pandas, NumPy                      |
| Scraping     | BeautifulSoup, Requests            |

---

## Interview One-Liner

> "I built a dynamic pricing engine that uses XGBoost for demand forecasting and Scipy optimization to find revenue-maximizing prices in real time. The system takes competitor prices, inventory levels, and seasonal signals as inputs and outputs an optimal price with expected revenue impact, served via a FastAPI backend and Streamlit dashboard."
