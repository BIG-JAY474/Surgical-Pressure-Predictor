from datetime import datetime
import streamlit as st
import pandas as pd
import joblib
import pydeck as pdk
import numpy as np

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG & CUSTOM STYLING (BEAUTIFICATION)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Toronto Bike Share | Surgical Predictor",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished cards, typography, and clean contrast
st.markdown("""
<style>
    /* Global styling overrides */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header hero section */
    .hero-container {
        background: linear-gradient(135deg, #0e1117 0%, #1a1c23 100%);
        border: 1px solid #2d3139;
        padding: 24px;
        border-radius: 12px;
        margin-bottom: 25px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Result card container */
    .result-card {
        padding: 20px;
        border-radius: 10px;
        border-left: 6px solid;
        margin-top: 10px;
        margin-bottom: 20px;
    }
    
    .result-alert {
        background-color: #2b1d1d;
        border-color: #ff4b4b;
        color: #ff8c8c;
    }
    
    .result-ok {
        background-color: #1b2e23;
        border-color: #21c35e;
        color: #8ce4ad;
    }
    
    /* Subheaders and accents */
    .section-head {
        font-size: 1.15rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        color: #f0f2f6;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. MODEL & DATA LOADING
# -----------------------------------------------------------------------------
@st.cache_resource
def load_assets():
    model = joblib.load('commuter_pulse_model.pkl')
    model_features = joblib.load('model_features.pkl')
    return model, model_features

model, model_features = load_assets()

# Extract list of available districts
districts = [f.replace('dist_', '') for f in model_features if f.startswith('dist_')]
districts = sorted(districts) if districts else ["Financial District", "Waterfront", "Annex", "Entertainment District"]

# -----------------------------------------------------------------------------
# 3. SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/bicycle.png", width=64)
    st.title("Control Panel")
    st.caption("Adjust real-time conditions")
    st.divider()

    st.markdown('<p class="section-head">📍 Location</p>', unsafe_allow_html=True)
    selected_district = st.selectbox("Select District", districts)

    st.markdown('<p class="section-head">🕒 Temporal & Weather</p>', unsafe_allow_html=True)
    selected_hour = st.slider("Hour of Day (0 - 23h)", 0, 23, 17)
    
    day_map = {"Monday":0, "Tuesday":1, "Wednesday":2, "Thursday":3, "Friday":4, "Saturday":5, "Sunday":6}
    selected_day = st.selectbox("Day of Week", list(day_map.keys()))

    col_w1, col_w2 = st.columns(2)
    with col_w1:
        temp = st.number_input("Temp (°C)", value=12.0, step=1.0)
        precip = st.number_input("Precip (mm)", value=5.0, step=0.5)
    with col_w2:
        wind = st.number_input("Wind (km/h)", value=15.0, step=1.0)

    st.markdown('<p class="section-head">📊 Station Activity</p>', unsafe_allow_html=True)
    flow_1h = st.number_input("Net Flow (Last 1h)", value=5)
    flow_3h = st.number_input("Net Flow (Last 3h)", value=15)
    capacity = st.number_input("Station Capacity", value=30)

# -----------------------------------------------------------------------------
# 4. FEATURE ENGINE PREPARATION
# -----------------------------------------------------------------------------
def prepare_input_vector():
    input_dict = {feat: 0 for feat in model_features}

    # Core features
    if 'hour' in input_dict: input_dict['hour'] = selected_hour
    if 'day_of_week' in input_dict: input_dict['day_of_week'] = day_map[selected_day]
    if 'is_weekend' in input_dict: input_dict['is_weekend'] = 1 if day_map[selected_day] >= 5 else 0
    if 'is_rush_hour' in input_dict: input_dict['is_rush_hour'] = 1 if selected_hour in [7,8,9,16,17,18] else 0
    if 'hour_sin' in input_dict: input_dict['hour_sin'] = np.sin(2 * np.pi * selected_hour / 24)
    if 'hour_cos' in input_dict: input_dict['hour_cos'] = np.cos(2 * np.pi * selected_hour / 24)
    if 'temp' in input_dict: input_dict['temp'] = temp
    if 'precip' in input_dict: input_dict['precip'] = precip
    if 'wind_spd' in input_dict: input_dict['wind_spd'] = wind
    if 'temp_rush_interaction' in input_dict: input_dict['temp_rush_interaction'] = temp * (1 if selected_hour in [7,8,9,16,17,18] else 0)
    if 'flow_1h' in input_dict: input_dict['flow_1h'] = flow_1h
    if 'flow_3h_rolling' in input_dict: input_dict['flow_3h_rolling'] = flow_3h
    if 'flow_trend' in input_dict: input_dict['flow_trend'] = flow_1h - (flow_3h / 3)
    if 'cap_feature' in input_dict: input_dict['cap_feature'] = capacity

    # One-hot district encoding
    dist_col = f"dist_{selected_district}"
    if dist_col in input_dict:
        input_dict[dist_col] = 1

    return pd.DataFrame([input_dict])[model_features]

# -----------------------------------------------------------------------------
# 5. MAIN CONTENT DISPLAY
# -----------------------------------------------------------------------------

# Hero Banner
st.markdown("""
<div class="hero-container">
    <h1 style="margin:0; font-size: 2.2rem;">🚲 Toronto Bike Share</h1>
    <p style="color: #8b949e; margin-top: 6px; font-size: 1.05rem;">
        September Surgical Pressure Predictor & Real-time Rebalancing Dashboard
    </p>
</div>
""", unsafe_allow_html=True)

input_df = prepare_input_vector()

# Model Execution
prediction = model.predict(input_df)[0]
proba = model.predict_proba(input_df)[0] if hasattr(model, "predict_proba") else [0.5, 0.5]
confidence = proba[int(prediction)] if len(proba) > 1 else 1.0

# Layout: Summary KPIs & Results
col_res, col_kpi = st.columns([1.2, 1], gap="large")

with col_res:
    st.subheader("🎯 Prediction & Action")
    if prediction != 0:
        st.markdown(f"""
        <div class="result-card result-alert">
            <h3 style="margin:0; color:#ff4b4b;">⚠️ HIGH PRESSURE DETECTED</h3>
            <p style="margin-top:8px; font-size: 1rem; color: #e6e6e6;">
                <b>Surgical Action:</b> Dispatch crew to rebalance bikes to this station immediately.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card result-ok">
            <h3 style="margin:0; color:#21c35e;">✅ NORMAL CAPACITY</h3>
            <p style="margin-top:8px; font-size: 1rem; color: #e6e6e6;">
                <b>Surgical Action:</b> No immediate intervention needed.
            </p>
        </div>
        """, unsafe_allow_html=True)

with col_kpi:
    st.subheader("📈 Model Metrics")
    m1, m2 = st.columns(2)
    m1.metric(
        label="Model Confidence", 
        value=f"{confidence:.1%}",
        delta="High Precision" if confidence > 0.75 else "Moderate",
        delta_color="normal"
    )
    m2.metric(
        label="Target State", 
        value="Critical" if prediction != 0 else "Stable",
        delta_color="inverse" if prediction != 0 else "normal"
    )

st.divider()

# -----------------------------------------------------------------------------
# 6. GEOSPATIAL VISUALIZATION
# -----------------------------------------------------------------------------
st.subheader(f"📍 Geographic Focus: {selected_district}")

coords = {
    "Financial District": [43.648, -79.381],
    "Waterfront": [43.639, -79.380],
    "Annex": [43.666, -79.403],
    "Entertainment District": [43.645, -79.390]
}
target_coord = coords.get(selected_district, [43.653, -79.383])

view_state = pdk.ViewState(
    latitude=target_coord[0],
    longitude=target_coord[1],
    zoom=14,
    pitch=45
)

# Marker color shifts dynamically based on status
marker_color = [255, 75, 75, 200] if prediction != 0 else [33, 195, 94, 200]

layer = pdk.Layer(
    "ScatterplotLayer",
    data=[{"lat": target_coord[0], "lon": target_coord[1]}],
    get_position="[lon, lat]",
    get_color=marker_color,
    get_radius=220,
    pickable=True
)

st.pydeck_chart(pdk.Deck(
    map_style="mapbox://styles/mapbox/dark-v9",
    initial_view_state=view_state, 
    layers=[layer]
))
