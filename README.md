---
title: PriceIQ - Dynamic Pricing Engine
emoji: 🏷️
colorFrom: purple
colorTo: indigo
sdk: streamlit
sdk_version: "1.35.0"
app_file: dashboard/app.py
pinned: false
license: mit
short_description: ML-powered real-time price optimization for small businesses
---

# Dynamic Pricing Engine 🏷️

> ML-powered price optimization for small businesses (kirana stores, local e-commerce, food stalls)

**Live Demo:** [Hugging Face Space](https://huggingface.co/spaces/yash7307/local-market-price-predictor)

---

## Features

- **XGBoost Demand Forecaster** — predicts units sold from pricing + context signals
- **Scipy Revenue Optimizer** — finds the revenue-maximizing price via bounded optimization
- **Price Elasticity Estimator** — classifies demand sensitivity with actionable strategy
- **Rule-based RL Agent** — State/Action/Reward environment (SB3 PPO-ready)
- **Competitor Price Scraper** — BeautifulSoup + simulated fallback
- **5-Tab Premium Dashboard** — glassmorphism dark UI with live Plotly charts

---

## Project Structure

```
dynamic-pricing-engine/
│
├── data/
│   ├── raw/              ← Simulated or real CSV sales data
│   └── processed/        ← Trained model (.pkl) saved here
│
├── dashboard/
│   └── app.py            ← Streamlit UI (5 tabs, glassmorphism theme)
│
├── src/
│   ├── data_collection/
│   │   ├── data_simulator.py   ← Synthetic dataset generator
│   │   └── scraper.py          ← Competitor price scraper
│   ├── models/
│   │   ├── demand_model.py     ← XGBoost demand forecaster
│   │   ├── elasticity.py       ← Price elasticity estimator
│   │   └── rl_agent.py         ← RL pricing agent
│   ├── optimizer/
│   │   └── price_optimizer.py  ← Scipy revenue maximizer
│   └── api/
│       └── main.py             ← FastAPI service
│
├── train_pipeline.py     ← One-command setup
└── requirements.txt
```

---

## Quick Start (Local)

```bash
# 1. Clone & setup
git clone https://github.com/yash7307/local-market-price-predictor.git
cd local-market-price-predictor
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt

# 2. Train model + generate data
python train_pipeline.py

# 3. Launch dashboard
streamlit run dashboard/app.py

# 4. (Optional) FastAPI backend
uvicorn src.api.main:app --reload --port 8000
# → Docs: http://localhost:8000/docs
```

> **Hugging Face Spaces:** The model auto-trains on first launch (~10 seconds). No setup needed.

---

## Tech Stack

| Layer        | Tools                              |
|--------------|-------------------------------------|
| ML Models    | XGBoost, Scipy optimization         |
| RL Agent     | Rule-based (Stable-Baselines3 ready)|
| Backend API  | FastAPI + Uvicorn + CORS            |
| Dashboard    | Streamlit + Plotly                  |
| Data         | Pandas, NumPy                       |
| Scraping     | BeautifulSoup, Requests             |

---

## Model Performance

| Metric | Value |
|--------|-------|
| MAE    | 10.18 units |
| R²     | 0.60  |
| Train  | 584 days |
| Test   | 146 days |

---

## Interview One-Liner

> *"I built a dynamic pricing engine using XGBoost for demand forecasting and Scipy optimization to find revenue-maximizing prices in real time. The system ingests competitor prices, inventory levels, seasonal signals, and weather data, then outputs an optimal price with expected revenue impact and price-elasticity classification. It's served via FastAPI and visualized in a multi-tab Streamlit dashboard deployed on Hugging Face Spaces."*
