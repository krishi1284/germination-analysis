# ================================================================
# SECTION 1: IMPORTS & PAGE SETUP
# This runs first every time the app loads.
# ================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

# Set page config — must be the very first Streamlit command
st.set_page_config(
    page_title="Seed Germination Analysis",
    page_icon="🌱",
    layout="wide"
)

# ================================================================
# SECTION 2: LOAD DATA & MODELS
# @st.cache_data means: load once, reuse every time.
# Without this, it would reload the CSV on every user interaction.
# ================================================================

@st.cache_data
def load_data():
    df = pd.read_csv('seed.info2.csv')
    df.columns = (df.columns
                  .str.replace('øC', 'C', regex=False)
                  .str.replace('(', '', regex=False)
                  .str.replace(')', '', regex=False)
                  .str.replace(' ', '_')
                  .str.strip())
    df['Seed_name'] = df['Seed_name'].str.strip().str.title()
    return df

@st.cache_resource
def load_models():
    best_model    = joblib.load('best_model.pkl')
    seed_models   = joblib.load('seed_models.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    features      = joblib.load('features.pkl')
    return best_model, seed_models, label_encoder, features

df                                        = load_data()
best_model, seed_models, le, FEATURES     = load_models()
SEEDS                                     = sorted(df['Seed_name'].unique().tolist())


# ================================================================
# SECTION 3: SIDEBAR
# Sidebar controls filter data across all tabs.
# Changing seed or day range updates every chart automatically.
# ================================================================

st.sidebar.image("https://em-content.zobj.net/source/apple/391/seedling_1f331.png", width=80)
st.sidebar.title("🌱 Germination Dashboard")
st.sidebar.markdown("---")

selected_seed = st.sidebar.selectbox(
    "Select Seed",
    options=SEEDS,
    index=0
)

day_range = st.sidebar.slider(
    "Select Day Range",
    min_value=int(df['Day'].min()),
    max_value=int(df['Day'].max()),
    value=(int(df['Day'].min()), int(df['Day'].max()))
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Info**")
st.sidebar.success("✅ XGBoost (Global)")
st.sidebar.info("✅ Per-Seed Models")

# Filter dataframe based on sidebar selections
filtered_df = df[
    (df['Seed_name'] == selected_seed) &
    (df['Day'] >= day_range[0]) &
    (df['Day'] <= day_range[1])
]


# ================================================================
# SECTION 4: HEADER & QUICK METRICS
# Shows key numbers at the top of the page.
# These update automatically when seed or day range changes.
# ================================================================

st.title("🌱 Seed Germination Analysis Dashboard")
st.markdown("Real ML-powered predictions using XGBoost & Random Forest")
st.markdown("---")

# Quick metric cards across the top
col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_growth = filtered_df['Growthmm'].mean()
    st.metric("Avg Growth", f"{avg_growth:.1f} mm")

with col2:
    max_growth = filtered_df['Growthmm'].max()
    st.metric("Max Growth", f"{max_growth:.1f} mm")

with col3:
    avg_temp = ((filtered_df['Temp_MorningC'] + filtered_df['Temp_AfternoonC']) / 2).mean()
    st.metric("Avg Temperature", f"{avg_temp:.1f} °C")

with col4:
    avg_water = filtered_df['Waterml'].mean()
    st.metric("Avg Water", f"{avg_water:.1f} ml")

st.markdown("---")


# ================================================================
# SECTION 5: TABS
# Each tab is one analysis view.
# Tabs keep the dashboard clean — user picks what they want to see.
# ================================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Growth Over Time",
    "🌡️ Growth vs Temperature",
    "💧 Growth vs Water",
    "🔥 Heatmap",
    "🤖 AI Prediction",
    "📊 Model Insights"
])

# ================================================================
# TAB 1: ANIMATED GROWTH OVER TIME
# Shows how each seed grows across days.
# Dropdown selects one seed; animated chart shows progression.
# ================================================================

with tab1:
    st.subheader(f"📈 Growth Over Days — {selected_seed}")

    fig1 = px.line(
        filtered_df,
        x='Day',
        y='Growthmm',
        markers=True,
        line_shape='spline',
        title=f"Growth Trend: {selected_seed} (Day {day_range[0]}–{day_range[1]})"
    )
    fig1.update_traces(line_color='#2ecc71', line_width=3, marker_size=8)
    fig1.update_layout(
        xaxis_title="Day",
        yaxis_title="Growth (mm)",
        template="plotly_white"
    )
    st.plotly_chart(fig1, use_container_width=True)

    # All seeds comparison below
    st.subheader("📊 All Seeds Comparison")
    fig_all = px.line(
        df[df['Day'].between(day_range[0], day_range[1])],
        x='Day',
        y='Growthmm',
        color='Seed_name',
        markers=True,
        line_shape='spline',
        title="All Seeds Growth Comparison"
    )
    fig_all.update_layout(template="plotly_white")
    st.plotly_chart(fig_all, use_container_width=True)


# ================================================================
# TAB 2: GROWTH vs TEMPERATURE
# Scatter plot shows relationship between temperature and growth.
# Bubble size = water amount. Color = day number.
# ================================================================

with tab2:
    st.subheader(f"🌡️ Growth vs Temperature — {selected_seed}")

    fig2 = px.scatter(
        filtered_df,
        x='Temp_AfternoonC',
        y='Growthmm',
        size='Waterml',
        color='Day',
        hover_data=['Moisture1-5', 'Sunlight_type'],
        title=f"Temperature vs Growth ({selected_seed})",
        color_continuous_scale='Viridis'
    )
    fig2.update_layout(template="plotly_white",
                       xaxis_title="Afternoon Temperature (°C)",
                       yaxis_title="Growth (mm)")
    st.plotly_chart(fig2, use_container_width=True)

    # Insight box
    corr = filtered_df['Temp_AfternoonC'].corr(filtered_df['Growthmm'])
    if abs(corr) > 0.5:
        st.info(f"📌 Temperature has a **{'positive' if corr > 0 else 'negative'} correlation** of `{corr:.2f}` with growth for {selected_seed}.")
    else:
        st.info(f"📌 Temperature shows **weak correlation** (`{corr:.2f}`) with growth for {selected_seed}.")


# ================================================================
# TAB 3: GROWTH vs WATER
# ================================================================

with tab3:
    st.subheader(f"💧 Growth vs Water — {selected_seed}")

    fig3 = px.scatter(
        filtered_df,
        x='Waterml',
        y='Growthmm',
        size='Moisture1-5',
        color='Day',
        hover_data=['Temp_AfternoonC', 'Sunlight_type'],
        title=f"Water vs Growth ({selected_seed})",
        color_continuous_scale='Blues'
    )
    fig3.update_layout(template="plotly_white",
                       xaxis_title="Water (ml)",
                       yaxis_title="Growth (mm)")
    st.plotly_chart(fig3, use_container_width=True)

    corr_w = filtered_df['Waterml'].corr(filtered_df['Growthmm'])
    st.info(f"📌 Water correlation with growth: `{corr_w:.2f}` for {selected_seed}.")


# ================================================================
# TAB 4: GROWTH HEATMAP
# Shows growth intensity across all seeds and days.
# Darker = more growth. Instantly shows which seed grew fastest.
# ================================================================

with tab4:
    st.subheader("🔥 Growth Heatmap — All Seeds")

    pivot = df[df['Day'].between(day_range[0], day_range[1])].pivot_table(
        index='Day',
        columns='Seed_name',
        values='Growthmm',
        aggfunc='mean'
    )
    pivot.index = pivot.index.astype(int)

    fig4 = px.imshow(
        pivot,
        labels=dict(x="Seed", y="Day", color="Growth (mm)"),
        color_continuous_scale='YlGn',
        title="Growth Heatmap (mm) by Seed & Day",
        aspect='auto'
    )
    fig4.update_layout(template="plotly_white")
    st.plotly_chart(fig4, use_container_width=True)

    # Sunlight bar chart below
    st.subheader("🌞 Average Growth by Sunlight Type")
    sun_df = df.groupby('Sunlight_type')['Growthmm'].mean().reset_index()
    fig_sun = px.bar(
        sun_df,
        x='Sunlight_type',
        y='Growthmm',
        text_auto='.1f',
        color='Growthmm',
        color_continuous_scale='Oranges',
        title="Avg Growth by Sunlight Type"
    )
    fig_sun.update_layout(template="plotly_white",
                          xaxis_title="Sunlight Type",
                          yaxis_title="Avg Growth (mm)")
    st.plotly_chart(fig_sun, use_container_width=True)

# ================================================================
# TAB 5: AI PREDICTION SIMULATOR
# THIS is the real model — not dummy like before.
# User adjusts sliders → model predicts growth in real time.
# We also apply the same feature engineering from training.
# ================================================================

with tab5:
    st.subheader("🤖 AI Growth Prediction Simulator")
    st.markdown("Adjust the environmental conditions below and get a **real ML prediction.**")

    model_choice = st.radio(
        "Choose Model",
        ["🌍 Global XGBoost (all seeds)", "🌱 Per-Seed Model (more accurate)"],
        horizontal=True
    )

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        pred_seed    = st.selectbox("Seed", SEEDS)
        pred_day     = st.slider("Day", 1, 10, 5)
        temp_morning = st.slider("Temp Morning (°C)", 15.0, 40.0, 25.0, 0.5)
        temp_aft     = st.slider("Temp Afternoon (°C)", 15.0, 45.0, 28.0, 0.5)

    with col_b:
        water        = st.slider("Water (ml)", 1, 60, 15)
        moisture     = st.slider("Moisture (1-5)", 1, 5, 3)
        sunlight     = st.slider("Sunlight Type (1-5)", 1, 5, 2)

    # ── Feature engineering (must match training exactly) ──
    temp_diff        = temp_aft - temp_morning
    avg_temp         = (temp_morning + temp_aft) / 2
    water_per_moist  = water / moisture
    seed_encoded     = le.transform([pred_seed])[0]

    input_data = pd.DataFrame([[
        temp_morning, temp_aft, water, sunlight, moisture,
        pred_day, temp_diff, avg_temp, water_per_moist, seed_encoded
    ]], columns=FEATURES)

    st.markdown("---")

    if st.button("🔮 Predict Growth", use_container_width=True):
        if "Global" in model_choice:
            prediction = best_model.predict(input_data)[0]
            model_used = "Global XGBoost"
        else:
            if pred_seed in seed_models:
                prediction = seed_models[pred_seed].predict(input_data)[0]
                model_used = f"Per-Seed ({pred_seed})"
            else:
                prediction = best_model.predict(input_data)[0]
                model_used = "Global XGBoost (fallback)"

        prediction = max(0, prediction)  # growth can't be negative

        st.success(f"🌱 Predicted Growth: **{prediction:.2f} mm**")
        st.caption(f"Model used: {model_used}")

        # Visual gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            number={'suffix': " mm"},
            gauge={
                'axis': {'range': [0, 75]},
                'bar': {'color': "#2ecc71"},
                'steps': [
                    {'range': [0, 20],  'color': '#ffeaa7'},
                    {'range': [20, 45], 'color': '#81ecec'},
                    {'range': [45, 75], 'color': '#55efc4'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': prediction
                }
            },
            title={'text': f"Predicted Growth for {pred_seed}"}
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

# ================================================================
# TAB 6: MODEL INSIGHTS
# Shows feature importance chart + per-seed model scores.
# This is the "explain your model" section — great for portfolio.
# ================================================================

with tab6:
    st.subheader("📊 Feature Importance")
    st.markdown("Which environmental factor drives plant growth the most?")

    importances = best_model.feature_importances_
    feat_df = pd.DataFrame({
        'Feature': FEATURES,
        'Importance': importances
    }).sort_values('Importance', ascending=True)

    fig6 = px.bar(
        feat_df,
        x='Importance',
        y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale='Greens',
        title="Feature Importance — XGBoost Global Model",
        text_auto='.3f'
    )
    fig6.update_layout(template="plotly_white", showlegend=False)
    st.plotly_chart(fig6, use_container_width=True)

    st.markdown("---")
    st.subheader("🌱 Per-Seed Model Performance")

    metrics_data = {
        'Seed': ['Green Gram (Large)', 'Green Gram (Small)', 'Black Gram', 'Brown Lentils', 'Chickpeas'],
        'R²':   [0.984, 0.556, 0.897, 0.792, 0.995],
        'MAE (mm)': [1.84, 9.34, 2.17, 5.02, 0.82]
    }
    metrics_df = pd.DataFrame(metrics_data)

    # Color R² column — green = good, red = weak
    def color_r2(val):
        color = '#2ecc71' if val >= 0.85 else ('#f39c12' if val >= 0.7 else '#e74c3c')
        return f'background-color: {color}; color: white; font-weight: bold'

    st.dataframe(
        metrics_df.style.applymap(color_r2, subset=['R²']),
        use_container_width=True
    )

    st.caption("🔴 Green Gram (Small) has the lowest R² (0.556) — its growth pattern is the most variable and hardest to predict with current features.")

