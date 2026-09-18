## Toronto Bike Share: Spatial-Temporal Pressure & Rebalancing Predictor

An end-to-end machine learning pipeline and live operational web application designed to predict station-level inventory pressure across Toronto's Bike Share network.

The system processes raw trip data, aggregates station flows into spatial tiers, engineers temporal lag and rolling features, and trains a LightGBM model to forecast 2-hour ahead station pressure (`Low`, `Normal`, `High`) with confidence thresholding for operational rebalancing.

---

## Technical Highlights & Results

* **Baseline vs. Tuned Performance:** Improved overall classification accuracy from a **50% baseline (Random Forest)** to **56% (LightGBM)** on unseen test data (September ridership).
* **Operational Precision:** Implemented a **55% confidence decision threshold**, driving precision for actionable alerts to **72% for Low Pressure** and **68% for High Pressure**, drastically reducing false dispatch alarms for logistics teams.
* **Spatial Tiering:** Segmented 1,000+ stations into spatial tiers (Tier-1 Core vs. Tier-2 Feeder) to filter high-density commuter sinks from low-signal background noise.
* **Feature Engineering:** Built rolling flow rates (3h moving averages), temporal lags ($t-1$, $t-2$, $t-24$), and cyclical sine/cosine time transformations to capture rush-hour seasonality.

---

## Project Architecture & Pipeline

```text
Raw CSV Trips (2.7M+) ──► Data Cleaning & Duration Filtering (1–480 min)
                                    │
JSON Station Metadata ──► District Parsing & Spatial Tiering (Tier-1 Core)
                                    │
                                    ▼
                      Hourly Arrivals / Departures Aggregation
                                    │
                                    ▼
               Feature Engineering (Lags, Rolling Stats, Cyclical Time)
                                    │
                                    ▼
                Temporal Split (Train: Jul–Aug | Test: Sept)
                                    │
                                    ▼
            LightGBM Classifier + Threshold Tuning (Confidence ≥ 0.55)
                                    │
                                    ▼
              Interactive Streamlit Deployment & Visual Dashboard

## Dataset & Feature Summary
The model trains on 38 engineered features combining spatial metadata, temporal indicators, weather metrics, and historic flow dynamics:Spatial: District one-hot encodings, station capacity, spatial tier classification.Temporal: hour_sin, hour_cos, day_of_week, is_rush_hour, is_weekend.Flow & Dynamics: Net flow ($t-1$), rolling flow mean (3h), rolling flow standard deviation (3h), flow trend indicator.Weather Integration: Temperature, precipitation, wind speed, temperature/rush-hour interaction.

## Model Evaluation & Decision Thresholding
Standard argmax evaluation forces decisions on uncertain probabilities, leading to high false-alarm rates. By enforcing custom confidence thresholds for extreme events:
Pressure State,Baseline Precision,Tuned Precision (≥ 0.55 Conf),Primary Operational Use Case
Low Pressure,63%,72%,Dispatch restocking vans before station empties
Normal State,53%,49%,Standard operations
High Pressure,57%,68%,Dispatch clear-out vans before station overflows

## Repository Structure
├── data/                       # Station metadata & processed trip aggregates
├── notebooks/                  # Interactive exploratory data analysis & model development
│   └── toronto-bikeshare-surgical-pressure-prediction.ipynb
├── models/                     # Serialized LightGBM model and feature pkl files
├── app.py                      # Streamlit web application
├── requirements.txt            # Python dependencies
└── README.md

## How to Run Locally
1. Prerequisites & Environment Setup
git clone [https://github.com/your-username/toronto-bikeshare-pressure-prediction.git](https://github.com/your-username/toronto-bikeshare-pressure-prediction.git)
cd toronto-bikeshare-pressure-prediction
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

## streamlit run app.py
