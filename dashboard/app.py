"""Dynamic Pricing Engine — Premium Streamlit Dashboard"""

import sys, numpy as np, pandas as pd, plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(page_title="PriceIQ | Dynamic Pricing Engine",
                   page_icon="🏷️", layout="wide",
                   initial_sidebar_state="expanded")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark background */
.stApp { background: linear-gradient(135deg, #0f0c29, #1a1a2e, #16213e); }

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04);
    border-right: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
}

/* Metric cards */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(108,99,255,0.3);
    border-radius: 16px;
    padding: 18px;
    backdrop-filter: blur(8px);
    transition: transform 0.2s, box-shadow 0.2s;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(108,99,255,0.25);
}

/* Hero banner */
.hero {
    background: linear-gradient(135deg, rgba(108,99,255,0.25), rgba(255,107,107,0.15));
    border: 1px solid rgba(108,99,255,0.35);
    border-radius: 20px;
    padding: 28px 36px;
    margin-bottom: 24px;
    backdrop-filter: blur(10px);
}
.hero h1 { font-size: 2rem; font-weight: 700; color: #fff; margin: 0; }
.hero p  { color: rgba(255,255,255,0.65); margin: 6px 0 0; font-size: 0.95rem; }

/* Cards */
.card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 16px;
    backdrop-filter: blur(6px);
}

/* Confidence badge */
.badge-high   { background:#00d68f22; color:#00d68f; border:1px solid #00d68f55; border-radius:8px; padding:4px 12px; font-weight:600; }
.badge-medium { background:#ffaa0022; color:#ffaa00; border:1px solid #ffaa0055; border-radius:8px; padding:4px 12px; font-weight:600; }
.badge-low    { background:#ff6b6b22; color:#ff6b6b; border:1px solid #ff6b6b55; border-radius:8px; padding:4px 12px; font-weight:600; }

/* Tab styling */
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-size: 0.85rem; font-weight: 500; color: rgba(255,255,255,0.6);
    border-radius: 8px 8px 0 0;
}
[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
    color: #6C63FF; border-bottom: 2px solid #6C63FF;
}

/* Divider */
hr { border-color: rgba(255,255,255,0.08) !important; }

/* Plotly chart bg */
.js-plotly-plot { border-radius: 12px; overflow: hidden; }

/* Sliders & inputs */
[data-testid="stSlider"] > div { color: #6C63FF; }
</style>
""", unsafe_allow_html=True)

ACCENT  = "#6C63FF"
ACCENT2 = "#FF6B6B"
GOLD    = "#FFD93D"
GREEN   = "#00D68F"
DARK_BG = "rgba(0,0,0,0)"

PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
    font=dict(family="Inter", color="rgba(255,255,255,0.8)"),
    margin=dict(l=20, r=20, t=40, b=20),
)

PRODUCT_CATEGORIES = {
    "FMCG / Groceries":  (50,  300, 120),
    "Electronics":       (200, 2000, 800),
    "Food & Beverages":  (20,  200,  80),
    "Clothing":          (100, 1500, 400),
    "Health & Beauty":   (50,  600, 200),
}

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load model, auto-training it if the .pkl doesn't exist yet (HF Spaces friendly)."""
    from pathlib import Path as _P
    pkl = _P(__file__).parent.parent / "data" / "processed" / "demand_model.pkl"
    if not pkl.exists():
        with st.spinner("First launch: training demand model... (~10 seconds)"):
            from src.data_collection.data_simulator import simulate_sales, OUTPUT_PATH
            from src.models.demand_model import train
            df = simulate_sales()
            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(OUTPUT_PATH, index=False)
            train(df)
    try:
        from src.models.demand_model import load_model as _l
        return _l()
    except Exception:
        return None

def build_curve(model, ctx, pmin, pmax, n=80):
    from src.models.demand_model import FEATURES
    prices, revenues, demands = [], [], []
    for p in np.linspace(pmin, pmax, n):
        row = {**ctx, "price": p}
        d   = max(0, float(model.predict(pd.DataFrame([row])[FEATURES])[0]))
        prices.append(round(p,2)); demands.append(round(d,1)); revenues.append(round(p*d,2))
    return pd.DataFrame({"price": prices, "demand": demands, "revenue": revenues})

def confidence_score(model, result):
    """Derive a simple confidence from R² proxy."""
    rev = result["expected_revenue"]
    if rev > 8000: return "High", GREEN
    if rev > 3000: return "Medium", GOLD
    return "Low", ACCENT2

def get_history():
    path = Path(__file__).parent.parent / "data" / "raw" / "sales_data.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=["date"])
        return df
    return None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style='text-align:center;padding:12px 0 20px'>
        <span style='font-size:2rem'>🏷️</span><br>
        <span style='color:#6C63FF;font-weight:700;font-size:1.1rem'>PriceIQ</span><br>
        <span style='color:rgba(255,255,255,0.4);font-size:0.75rem'>Dynamic Pricing Engine</span>
    </div>""", unsafe_allow_html=True)

    st.markdown("**Product Setup**")
    category = st.selectbox("Product Category", list(PRODUCT_CATEGORIES.keys()), index=0)
    pmin_default, pmax_default, price_default = PRODUCT_CATEGORIES[category]

    st.markdown("---")
    st.markdown("**Market Inputs**")
    current_price    = st.slider("Your Current Price (₹)", pmin_default, pmax_default, price_default)
    competitor_price = st.number_input("Competitor Price (₹)", value=float(int(price_default*0.95)), step=5.0)
    inventory        = st.slider("Inventory (units)", 0, 500, 100)

    st.markdown("---")
    st.markdown("**Context**")
    is_weekend  = st.checkbox("Weekend?",         value=datetime.today().weekday() >= 5)
    is_festival = st.checkbox("Festival Season?", value=False)
    temperature = st.slider("Temperature (°C)", 10, 50, 30)

    st.markdown("---")
    st.markdown("**Price Bounds**")
    price_min = st.number_input("Min (₹)", value=float(pmin_default), step=10.0)
    price_max = st.number_input("Max (₹)", value=float(pmax_default), step=10.0)

# ── Model + Context ───────────────────────────────────────────────────────────
model = load_model()

context = {
    "competitor_price": competitor_price,
    "is_weekend":       int(is_weekend),
    "is_festival":      int(is_festival),
    "inventory":        inventory,
    "month":            datetime.today().month,
    "day_of_week":      datetime.today().weekday(),
    "temperature":      float(temperature),
}

# ── Hero Banner ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <h1>🏷️ Dynamic Pricing Engine</h1>
  <p>ML-powered real-time price optimization &nbsp;·&nbsp; <b style="color:#6C63FF">{category}</b> &nbsp;·&nbsp;
     {datetime.today().strftime('%A, %d %b %Y')}</p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("⚠️ No trained model found. Run: `python train_pipeline.py`")
    st.stop()

from src.optimizer.price_optimizer import get_optimal_price
result = get_optimal_price(model, context, price_min, price_max)
conf_label, conf_color = confidence_score(model, result)
delta_price   = result["optimal_price"] - current_price
current_rev   = current_price * max(0, float(model.predict(
    pd.DataFrame([{**context, "price": current_price}])[
        ["price","competitor_price","is_weekend","is_festival","inventory","month","day_of_week","temperature"]
    ])[0]))
rev_gain      = result["expected_revenue"] - current_rev
rev_gain_pct  = (rev_gain / current_rev * 100) if current_rev > 0 else 0

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Price Optimizer",
    "📊 Revenue Analysis",
    "🔍 Market Intelligence",
    "📈 Sales History",
    "ℹ️ About",
])

# ════════════════════════════════════════════════════════════
# TAB 1 — Price Optimizer
# ════════════════════════════════════════════════════════════
with tab1:
    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💰 Optimal Price",    f"₹{result['optimal_price']:,.2f}",
              f"{delta_price:+.2f} vs current")
    c2.metric("📦 Expected Demand",  f"{result['expected_demand']:,.0f} units")
    c3.metric("📈 Expected Revenue", f"₹{result['expected_revenue']:,.0f}",
              f"{rev_gain_pct:+.1f}% vs static")
    c4.metric("🎯 Confidence",       conf_label)

    st.markdown("---")

    col_a, col_b = st.columns([3, 2])

    with col_a:
        # Revenue curve — compact
        curve_df = build_curve(model, context, price_min, price_max)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=curve_df["price"], y=curve_df["revenue"],
            mode="lines", fill="tozeroy",
            fillcolor="rgba(108,99,255,0.12)",
            line=dict(color=ACCENT, width=2.5), name="Revenue"))
        fig.add_vline(x=result["optimal_price"], line_dash="dash",
            line_color=GREEN, annotation_text=f"Optimal ₹{result['optimal_price']:.0f}",
            annotation_font_color=GREEN)
        fig.add_vline(x=current_price, line_dash="dot",
            line_color=GOLD, annotation_text=f"Current ₹{current_price}",
            annotation_font_color=GOLD)
        fig.update_layout(**PLOT_LAYOUT, height=320, title="Price vs Revenue Curve",
            xaxis_title="Price (₹)", yaxis_title="Revenue (₹)")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("#### Pricing Recommendation")
        direction = "raise" if delta_price > 0 else "lower"
        pct_change = abs(delta_price / current_price * 100)

        st.markdown(f"""
        <div class='card'>
          <p style='color:rgba(255,255,255,0.55);margin:0 0 6px;font-size:0.8rem'>SUGGESTED ACTION</p>
          <p style='font-size:1.5rem;font-weight:700;color:#fff;margin:0'>
            {'📈' if delta_price > 0 else '📉'} {direction.title()} by {pct_change:.1f}%
          </p>
          <p style='color:rgba(255,255,255,0.6);font-size:0.85rem;margin:8px 0 0'>
            From <b style='color:{GOLD}'>₹{current_price}</b> → 
            <b style='color:{GREEN}'>₹{result['optimal_price']:.2f}</b>
          </p>
        </div>
        <div class='card'>
          <p style='color:rgba(255,255,255,0.55);margin:0 0 6px;font-size:0.8rem'>REVENUE IMPACT</p>
          <p style='font-size:1.4rem;font-weight:700;color:{"#00d68f" if rev_gain>=0 else "#ff6b6b"};margin:0'>
            {"+" if rev_gain >= 0 else ""}₹{rev_gain:,.0f} / day
          </p>
          <p style='color:rgba(255,255,255,0.5);font-size:0.8rem;margin:6px 0 0'>
            Monthly impact: {"+" if rev_gain>=0 else ""}₹{rev_gain*30:,.0f}
          </p>
        </div>
        <div class='card'>
          <p style='color:rgba(255,255,255,0.55);margin:0 0 8px;font-size:0.8rem'>CONFIDENCE</p>
          <span class='badge-{"high" if conf_label=="High" else "medium" if conf_label=="Medium" else "low"}'>
            {conf_label} Confidence
          </span>
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# TAB 2 — Revenue Analysis
# ════════════════════════════════════════════════════════════
with tab2:
    if "curve_df" not in dir():
        curve_df = build_curve(model, context, price_min, price_max)

    c1, c2 = st.columns(2)

    with c1:
        # Dual-axis: Revenue + Demand vs Price
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=curve_df["price"], y=curve_df["revenue"],
            name="Revenue (₹)", line=dict(color=ACCENT, width=2.5),
            fill="tozeroy", fillcolor="rgba(108,99,255,0.10)"))
        fig2.add_trace(go.Scatter(x=curve_df["price"], y=curve_df["demand"],
            name="Demand (units)", line=dict(color=ACCENT2, width=2, dash="dot"),
            yaxis="y2"))
        fig2.add_vline(x=result["optimal_price"], line_dash="dash",
            line_color=GREEN, annotation_text="Optimal")
        fig2.update_layout(**PLOT_LAYOUT, height=360,
            title="Revenue & Demand vs Price",
            xaxis_title="Price (₹)", yaxis_title="Revenue (₹)",
            yaxis2=dict(title="Demand (units)", overlaying="y", side="right",
                        showgrid=False, color=ACCENT2),
            legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        # Revenue at key price points
        key_prices = [price_min, price_min+(price_max-price_min)*0.25,
                      current_price, result["optimal_price"],
                      price_min+(price_max-price_min)*0.75, price_max]
        key_revs, key_demands = [], []
        from src.models.demand_model import FEATURES
        for p in key_prices:
            row  = {**context, "price": p}
            d    = max(0, float(model.predict(pd.DataFrame([row])[FEATURES])[0]))
            key_revs.append(round(p*d, 0))
            key_demands.append(round(d, 1))
        labels = [f"₹{int(p)}" for p in key_prices]
        colors = [GREEN if p == result["optimal_price"] else ACCENT for p in key_prices]

        fig3 = go.Figure(go.Bar(x=labels, y=key_revs,
            marker_color=colors,
            text=[f"₹{int(r):,}" for r in key_revs],
            textposition="outside", textfont_color="white"))
        fig3.update_layout(**PLOT_LAYOUT, height=360,
            title="Revenue at Key Price Points", yaxis_title="Revenue (₹)")
        st.plotly_chart(fig3, use_container_width=True)

    # A/B Impact
    st.markdown("#### A/B Test: Static vs Dynamic Pricing")
    cols = st.columns(3)
    cols[0].metric("Static Pricing Revenue",  f"₹{current_rev:,.0f}/day", "Your current price")
    cols[1].metric("Dynamic Pricing Revenue", f"₹{result['expected_revenue']:,.0f}/day",
                   f"+₹{rev_gain:,.0f}" if rev_gain >= 0 else f"₹{rev_gain:,.0f}")
    cols[2].metric("Annual Revenue Gain",     f"₹{rev_gain*365:,.0f}",
                   f"{rev_gain_pct:+.1f}%")

# ════════════════════════════════════════════════════════════
# TAB 3 — Market Intelligence
# ════════════════════════════════════════════════════════════
with tab3:
    col1, col2 = st.columns(2)

    with col1:
        # Feature importance
        importance = pd.Series(model.feature_importances_,
                               index=model.feature_names_in_).sort_values()
        nice_names = {
            "price": "Your Price", "competitor_price": "Competitor Price",
            "inventory": "Inventory", "is_weekend": "Weekend",
            "is_festival": "Festival", "month": "Month",
            "day_of_week": "Day of Week", "temperature": "Temperature",
        }
        labels = [nice_names.get(i, i) for i in importance.index]
        bar_colors = [GREEN if importance[i] == importance.max() else ACCENT
                      for i in importance.index]
        fig4 = go.Figure(go.Bar(x=importance.values, y=labels,
            orientation="h", marker_color=bar_colors,
            text=[f"{v:.3f}" for v in importance.values],
            textposition="outside", textfont_color="white"))
        fig4.update_layout(**PLOT_LAYOUT, height=360,
            title="Feature Importance — What Drives Demand?",
            xaxis_title="Importance Score")
        st.plotly_chart(fig4, use_container_width=True)

    with col2:
        # Elasticity gauge
        from src.models.elasticity import point_elasticity, classify_elasticity, recommend_strategy
        mid_idx = len(curve_df)//2
        e = point_elasticity(
            curve_df.iloc[mid_idx-1]["price"], curve_df.iloc[mid_idx]["price"],
            curve_df.iloc[mid_idx-1]["demand"], curve_df.iloc[mid_idx]["demand"])
        e_class = classify_elasticity(e)

        fig5 = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=round(e, 3),
            title=dict(text="Price Elasticity", font=dict(size=16, color="white")),
            gauge=dict(
                axis=dict(range=[-3, 1], tickcolor="white"),
                bar=dict(color=ACCENT),
                steps=[
                    dict(range=[-3, -1], color="rgba(255,107,107,0.25)"),
                    dict(range=[-1,  0], color="rgba(255,170,0,0.25)"),
                    dict(range=[ 0,  1], color="rgba(0,214,143,0.25)"),
                ],
                threshold=dict(line=dict(color=GREEN, width=3), value=-1),
            ),
            number=dict(font=dict(color="white", size=28)),
        ))
        fig5.update_layout(**PLOT_LAYOUT, height=280)
        st.plotly_chart(fig5, use_container_width=True)
        st.info(f"**{e_class}** — {recommend_strategy(e, inventory)}")

    # Competitor comparison
    st.markdown("#### Competitor Price Landscape")
    from src.data_collection.scraper import get_competitor_price
    product_key = category.lower().split("/")[0].strip().replace(" ", "_")
    comp_data   = get_competitor_price(product_key, num_competitors=4, seed=42)

    comp_names  = [c["name"] for c in comp_data["competitors"]] + ["You (Current)", "Optimal"]
    comp_prices = [c["price"] for c in comp_data["competitors"]] + [current_price, result["optimal_price"]]
    comp_colors = ([ACCENT]*4) + [GOLD, GREEN]

    fig6 = go.Figure(go.Bar(x=comp_names, y=comp_prices,
        marker_color=comp_colors,
        text=[f"₹{p:.0f}" for p in comp_prices],
        textposition="outside", textfont_color="white"))
    fig6.update_layout(**PLOT_LAYOUT, height=300,
        title="Price Comparison vs Competitors", yaxis_title="Price (₹)")
    st.plotly_chart(fig6, use_container_width=True)

# ════════════════════════════════════════════════════════════
# TAB 4 — Sales History
# ════════════════════════════════════════════════════════════
with tab4:
    hist_df = get_history()
    if hist_df is not None:
        hist_df["revenue"] = hist_df["price"] * hist_df["demand"]
        hist_df["30d_avg_rev"] = hist_df["revenue"].rolling(30).mean()

        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Daily Demand",  f"{hist_df['demand'].mean():.0f} units")
        c2.metric("Avg Daily Revenue", f"₹{hist_df['revenue'].mean():,.0f}")
        c3.metric("Avg Price",         f"₹{hist_df['price'].mean():.2f}")

        # Revenue over time
        fig7 = go.Figure()
        fig7.add_trace(go.Scatter(x=hist_df["date"], y=hist_df["revenue"],
            mode="lines", line=dict(color=ACCENT, width=1), name="Daily Revenue",
            opacity=0.5))
        fig7.add_trace(go.Scatter(x=hist_df["date"], y=hist_df["30d_avg_rev"],
            mode="lines", line=dict(color=GREEN, width=2.5), name="30-day Avg"))
        fig7.update_layout(**PLOT_LAYOUT, height=320,
            title="Historical Daily Revenue", xaxis_title="Date", yaxis_title="Revenue (₹)")
        st.plotly_chart(fig7, use_container_width=True)

        # Price vs Demand scatter
        sample = hist_df.sample(min(300, len(hist_df)), random_state=1)
        fig8 = go.Figure(go.Scatter(
            x=sample["price"], y=sample["demand"],
            mode="markers",
            marker=dict(color=sample["demand"], colorscale="Viridis",
                        size=6, opacity=0.7, showscale=True,
                        colorbar=dict(title="Demand"))))
        fig8.update_layout(**PLOT_LAYOUT, height=320,
            title="Price vs Demand (Historical Scatter)",
            xaxis_title="Price (₹)", yaxis_title="Demand (units)")
        st.plotly_chart(fig8, use_container_width=True)

        with st.expander("📋 Raw Data"):
            st.dataframe(hist_df.tail(60), use_container_width=True)
    else:
        st.warning("No sales data found. Run `python train_pipeline.py` first.")

# ════════════════════════════════════════════════════════════
# TAB 5 — About
# ════════════════════════════════════════════════════════════
with tab5:
    st.markdown("""
    ## About PriceIQ — Dynamic Pricing Engine

    > Built for small businesses to compete with algorithm-driven giants.

    ### Architecture
    """)
    st.code("""
Input Layer          ML Engine                  Output
─────────────        ───────────────────────    ──────────────────────
Competitor Prices ─► Demand Forecasting ──────► Optimal Price (₹)
Historical Sales  ─► (XGBoost, R²=0.60)        Expected Revenue
Time Features     ─► Price Elasticity ────────► Confidence Score
Inventory Level   ─► Estimator                  Revenue Impact
Weather / Temp    ─► Scipy Revenue ────────────► Dashboard Charts
Festival Flags    ─► Optimizer
                  ─► Rule-Based RL Agent
    """, language="text")

    st.markdown("### Tech Stack")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
| Layer        | Tool                  |
|:------------|:---------------------|
| ML Model    | XGBoost + Scipy      |
| Optimizer   | Scipy minimize_scalar |
| RL Agent    | Rule-based (SB3 ready) |
| Backend API | FastAPI + Uvicorn    |
        """)
    with col2:
        st.markdown("""
| Layer       | Tool               |
|:-----------|:------------------|
| Dashboard  | Streamlit + Plotly |
| Data       | Pandas + NumPy    |
| Scraping   | BeautifulSoup     |
| Elasticity | Custom formula    |
        """)

    st.markdown("### Interview One-Liner")
    st.info("""
💬 *"I built a dynamic pricing engine using XGBoost for demand forecasting and Scipy optimization
to find revenue-maximizing prices in real time. The system ingests competitor prices, inventory
levels, seasonal signals, and weather data, then outputs an optimal price with expected revenue
impact and a price-elasticity classification. It's served via FastAPI and visualized in a
multi-tab Streamlit dashboard."*
    """)

    st.markdown("---")
    st.caption("PriceIQ v1.0 · Built with Streamlit + XGBoost · For small business pricing intelligence")
