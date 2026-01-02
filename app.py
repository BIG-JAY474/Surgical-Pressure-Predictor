from datetime import datetime
import streamlit as st
import pandas as pd
import joblib
import pydeck as pdk
import numpy as np

# 1. LOAD THE BRAIN
# Ensure these two files are in your GitHub repo!
model = joblib.load('commuter_pulse_model.pkl')
model_features = joblib.load('model_features.pkl')

st.set_page_config(page_title="Toronto Bike Share Predictor", layout="centered")

st.title("🚲 Toronto Bike Share: September Surgical Predictor")
st.markdown("This model uses **68 surgical features** to predict station pressure 2 hours in advance.")

# 2. SIDEBAR INPUTS
with st.sidebar:
    st.header("📍 Station Location")
    # Dynamically pull district names from your 68 features
    districts = [f.replace('dist_', '') for f in model_features if f.startswith('dist_')]
    selected_district = st.selectbox("Select District", sorted(districts))
    
    st.header("🕒 Time & Weather")
    selected_hour = st.slider("Hour of Day (0-23)", 0, 23, 8)
    day_map = {"Monday":0, "Tuesday":1, "Wednesday":2, "Thursday":3, "Friday":4, "Saturday":5, "Sunday":6}
    selected_day = st.selectbox("Day of Week", list(day_map.keys()))
    
    temp = st.number_input("Temperature (°C)", value=18)
    precip = st.number_input("Precipitation (mm)", value=0.0)
    wind = st.number_input("Wind Speed (km/h)", value=15)

    st.header("📊 Current Flow")
    flow_1h = st.number_input("Net Flow (Last 1h)", value=0)
    flow_3h = st.number_input("Net Flow (Last 3h)", value=0)
    capacity = st.number_input("Station Capacity", value=20)

# 3. FEATURE ENGINE (Matches the Kaggle logic)
def prepare_input_vector():
    # Initialize a dictionary with all 68 features set to 0
    input_dict = {feat: 0 for feat in model_features}
    
    # Fill temporal features
    input_dict['hour'] = selected_hour
    input_dict['day_of_week'] = day_map[selected_day]
    input_dict['is_weekend'] = 1 if day_map[selected_day] >= 5 else 0
    input_dict['is_rush_hour'] = 1 if selected_hour in [7,8,9,16,17,18] else 0
    
    # Cyclical Encoding
    input_dict['hour_sin'] = np.sin(2 * np.pi * selected_hour / 24)
    input_dict['hour_cos'] = np.cos(2 * np.pi * selected_hour / 24)
    
    # Weather & Interaction
    input_dict['temp'] = temp
    input_dict['precip'] = precip
    input_dict['wind_spd'] = wind
    input_dict['temp_rush_interaction'] = temp * input_dict['is_rush_hour']
    
    # Trends & Capacity
    input_dict['flow_1h'] = flow_1h
    input_dict['flow_3h_rolling'] = flow_3h
    input_dict['flow_trend'] = flow_1h - (flow_3h / 3)
    input_dict['cap_feature'] = capacity
    
    # One-Hot District (Set the specific selected district to 1)
    dist_col = f"dist_{selected_district}"
    if dist_col in input_dict:
        input_dict[dist_col] = 1
        
    # Create DataFrame in the exact order the model was trained on
    return pd.DataFrame([input_dict])[model_features]

# 4. PREDICTION
if st.button("Generate Prediction"):
    input_df = prepare_input_vector()
    
    # Get class and probabilities
    prediction = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0]
    
    # Mapping
    labels = {0: "Low Supply (Emptying)", 1: "Normal", 2: "High Supply (Filling)"}
    result = labels[prediction]
    conf = probs[prediction]
    
    # Display Result
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
