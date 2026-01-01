from datetime import datetime
import streamlit as st
import pandas as pd
import joblib
import pydeck as pdk

# --- 1. LOAD ASSETS ---
@st.cache_resource
def load_model_and_features():
    # Ensure these match your latest v2 exports!
    model = joblib.load('commuter_pulse_model.pkl')
    features = joblib.load('model_features.pkl')
    return model, features

@st.cache_data
def load_station_inventory():
    # This loads your high-quality, Tiered CSV from Jupyter
    df = pd.read_csv('station_districts_mapped.csv')
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
    return df.dropna(subset=['lat', 'lon'])

model, model_features = load_model_and_features()
stations_inventory = load_station_inventory()

# --- 2. UI SETUP ---
st.set_page_config(page_title="Toronto Bike Pressure", page_icon="🚲", layout="wide")
st.title("🚲 Toronto Bike Share: Surgical Pressure Predictor")

# --- 3. SIDEBAR INPUTS ---
st.sidebar.header("📍 Station Context")
districts = [col.replace('dist_', '') for col in model_features if col.startswith('dist_')]
selected_district = st.sidebar.selectbox("Select District", sorted(districts))

capacity = st.sidebar.slider("Station Capacity", 5, 100, 25)
hour = st.sidebar.slider("Current Hour (24h)", 0, 23, 12)
flow_3h = st.sidebar.number_input("Net Flow (Last 3 Hours)", value=0)

st.sidebar.header("☁️ Weather Conditions")
temp = st.sidebar.slider("Temperature (°C)", -10, 40, 20)
precip = st.sidebar.number_input("Precipitation (mm)", value=0.0)
wind_spd = st.sidebar.slider("Wind Speed (km/h)", 0, 60, 10)

# --- 4. DATA PREPARATION ---
def prepare_input():
    input_df = pd.DataFrame(0, index=[0], columns=model_features)
    input_df['hour'] = hour
    input_df['cap_feature'] = capacity
    input_df['flow_3h_rolling'] = flow_3h
    input_df['temp'] = temp
    input_df['precip'] = precip
    input_df['wind_spd'] = wind_spd
    input_df['is_rush_hour'] = 1 if hour in [7, 8, 9, 16, 17, 18] else 0
    input_df['is_raining'] = 1 if precip > 0.2 else 0
    input_df['is_windy'] = 1 if wind_spd > 25 else 0
    
    dist_col = f"dist_{selected_district}"
    if dist_col in input_df.columns:
        input_df[dist_col] = 1
    return input_df

# --- 5. PREDICTION & DASHBOARD ---
if st.button("Predict 2-Hour Future State"):
    input_df = prepare_input()
    probs = model.predict_proba(input_df)[0]
    
    # Apply Detective's Bias for 'Normal' (30% threshold)
    if probs[1] > 0.30:
        final_state = 1
    else:
        final_state = model.predict(input_df)[0]
    
    states = {
        0: ("🔴 LOW SUPPLY", "Empty station alert. Riders won't find bikes.", "error"),
        1: ("🟢 NORMAL", "Balanced flow. Station is healthy.", "success"),
        2: ("🔵 HIGH SUPPLY", "Full station alert. Riders won't find docks.", "info")
    }
    label, advice, color_type = states[final_state]
    
    # --- DISPLAY RESULTS ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Forecast for { (hour + 2) % 24}:00")
        if color_type == "error": st.error(label)
        elif color_type == "success": st.success(label)
        else: st.info(label)
        st.write(f"**Advice:** {advice}")

    with col2:
        st.subheader("Model Confidence")
        conf_val = probs[final_state]
        st.metric("Confidence Score", f"{conf_val:.1%}")
        st.progress(conf_val)

    # --- 6. THE DECISION SUPPORT SYSTEM ---
    st.divider()
    st.subheader(f"🛠️ Deployment Plan: {selected_district}")
    
    local_stations = stations_inventory[stations_inventory['district'] == selected_district]
    
    if not local_stations.empty:
        is_core_zone = "Tier-1-Core" in local_stations['tier'].values
        
        if is_core_zone:
            st.warning("⚡ **High-Pressure Core Detected:** This district contains critical hub stations. Prioritize rebalancing.")
        
        tab1, tab2 = st.tabs(["📍 Priority Map", "📋 Station List"])
        
        with tab1:
            st.pydeck_chart(pdk.Deck(
                map_style='mapbox://styles/mapbox/dark-v9',
                initial_view_state=pdk.ViewState(
                    latitude=local_stations['lat'].mean(),
                    longitude=local_stations['lon'].mean(),
                    zoom=13,
                    pitch=45,
                ),
                layers=[
                    pdk.Layer(
                        'ScatterplotLayer',
                        data=local_stations,
                        get_position='[lon, lat]',
                        get_color='[200, 30, 0, 160]',
                        get_radius=100,
                    ),
                ],
            ))
        
        with tab2:
            st.dataframe(local_stations, use_container_width=True)
            action = "ADD BIKES" if final_state == 0 else "CLEAR DOCKS" if final_state == 2 else "MONITOR"
            st.success(f"**Action Plan:** {action} across {len(local_stations)} stations.")
    else:
        st.warning(f"No specific station data found for '{selected_district}' in the mapped CSV.")

    with st.expander("Debug: Model Feature Vector"):
        st.dataframe(input_df, use_container_width=True)

else:
    st.info("👈 Adjust the sidebar parameters and click 'Predict' to generate the deployment plan.")