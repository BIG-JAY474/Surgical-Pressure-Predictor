from datetime import datetime
import streamlit as st
import pandas as pd
import joblib
import pydeck as pdk
import numpy as np

# 1. LOAD THE BRAIN (The 41-Feature Version)
model = joblib.load('commuter_pulse_model.pkl')
model_features = joblib.load('model_features.pkl')

st.set_page_config(page_title="Toronto Bike Share Predictor", layout="centered")

st.title("🚲 Toronto Bike Share: September Surgical Predictor")
st.markdown("Early-warning system optimized for **High Recall** on station pressure.")

# 2. SIDEBAR INPUTS
with st.sidebar:
    st.header("📍 Station Location")
    # Dynamically pull the 27 cleaned districts from your model features
    districts = [f.replace('dist_', '') for f in model_features if f.startswith('dist_')]
    selected_district = st.selectbox("Select District", sorted(districts))
    
    st.header("🕒 Time & Weather")
    selected_hour = st.slider("Hour of Day (0-23)", 0, 23, 17)
    day_map = {"Monday":0, "Tuesday":1, "Wednesday":2, "Thursday":3, "Friday":4, "Saturday":5, "Sunday":6}
    selected_day = st.selectbox("Day of Week", list(day_map.keys()))
    
    temp = st.number_input("Temperature (°C)", value=12)
    precip = st.number_input("Precipitation (mm)", value=5.0)
    wind = st.number_input("Wind Speed (km/h)", value=15)

    st.header("📊 Current Flow")
    flow_1h = st.number_input("Net Flow (Last 1h)", value=5)
    flow_3h = st.number_input("Net Flow (Last 3h)", value=15)
    capacity = st.number_input("Station Capacity", value=30)

# 3. FEATURE ENGINE (The 41-Feature Surgical Logic)
def prepare_input_vector():
    # Initialize all 41 features to 0
    input_dict = {feat: 0 for feat in model_features}
    
    # Core 14 Features
    input_dict['hour'] = selected_hour
    input_dict['day_of_week'] = day_map[selected_day]
    input_dict['is_weekend'] = 1 if day_map[selected_day] >= 5 else 0
    input_dict['is_rush_hour'] = 1 if selected_hour in [7,8,9,16,17,18] else 0
    input_dict['hour_sin'] = np.sin(2 * np.pi * selected_hour / 24)
    input_dict['hour_cos'] = np.cos(2 * np.pi * selected_hour / 24)
    input_dict['temp'] = temp
    input_dict['precip'] = precip
    input_dict['wind_spd'] = wind
    input_dict['temp_rush_interaction'] = temp * input_dict['is_rush_hour']
    input_dict['flow_1h'] = flow_1h
    input_dict['flow_3h_rolling'] = flow_3h
    input_dict['flow_trend'] = flow_1h - (flow_3h / 3)
    input_dict['cap_feature'] = capacity
    
    # One-Hot District (The remaining 27 features)
    dist_col = f"dist_{selected_district}"
    if dist_col in input_dict:
        input_dict[dist_col] = 1
        
    return pd.DataFrame([input_dict])[model_features]

# 4. PREDICTION & OUTPUT (Indented correctly to avoid NameError)
if st.button("Generate Prediction"):
    # Born inside the button block
    input_df = prepare_input_vector()
    
    prediction = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0]
    
    labels = {0: "Low Supply (Emptying)", 1: "Normal", 2: "High Supply (Filling)"}
    result = labels[prediction]
    conf = probs[prediction]
    
    st.divider()
    if prediction == 2:
        st.error(f"### Result: {result}")
        st.write("🚚 **Surgical Action:** Dispatch clear-out van within 60 minutes.")
    elif prediction == 0:
        st.warning(f"### Result: {result}")
        st.write("🚲 **Surgical Action:** Rebalance bikes to this station.")
    else:
        st.success(f"### Result: {result}")
        st.write("✅ **Surgical Action:** No intervention needed.")
        
    st.metric("Model Confidence", f"{conf:.2%}")

    # 5. SURGICAL MAP (Inside the block so it has access to selected_district)
    st.write(f"### 📍 Surgical Focus: {selected_district}")
    
    # Coordinates for 'Street View' context
    coords = {
        "Financial District": [43.648, -79.381],
        "Waterfront": [43.639, -79.380],
        "Annex": [43.666, -79.403],
        "Entertainment District": [43.645, -79.390]
    }
    
    # Default to center of Toronto if district coords not mapped yet
    target_coord = coords.get(selected_district, [43.653, -79.383])
    
    view_state = pdk.ViewState(
        latitude=target_coord[0], 
        longitude=target_coord[1], 
        zoom=14, 
        pitch=40
    )
    
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=[{"lat": target_coord[0], "lon": target_coord[1]}],
        get_position='[lon, lat]',
        get_color='[200, 30, 0, 160]' if prediction != 1 else '[0, 200, 0, 160]',
        get_radius=250,
    )
    
    st.pydeck_chart(pdk.Deck(initial_view_state=view_state, layers=[layer]))
