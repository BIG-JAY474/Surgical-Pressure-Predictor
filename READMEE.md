# Toronto Bike Share: Spatiotemporal Pressure & Rebalancing Predictor

An end-to-end machine learning pipeline and live operational web application designed to predict station-level inventory pressure across Toronto's Bike Share network.

The system processes raw trip data, aggregates station flows into spatial tiers, engineers temporal lag and rolling features, and trains a LightGBM model to forecast station pressure two hours ahead.

## Technical Highlights & Results

- **Baseline vs. tuned performance:** Improved overall classification accuracy from a **50% baseline using Random Forest** to **56% using LightGBM** on unseen September ridership data.
- **Operational precision:** A **55% confidence decision threshold** increased precision for actionable alerts to **72% for Low Pressure** and **68% for High Pressure**, reducing unnecessary interventions.
- **Spatial tiering:** Segmented more than 1,000 stations into spatial tiers—Tier 1 Core and Tier 2 Feeder—to distinguish high-density commuter destinations from lower-signal stations.
- **Feature engineering:** Built rolling flow rates, including three-hour moving averages; temporal lags (`t-1`, `t-2`, and `t-24`); and cyclical sine/cosine time transformations to capture rush-hour seasonality.

## Project Architecture & Pipeline

```text
Raw CSV Trips (2.7M+)
        │
        ▼
Data Cleaning & Duration Filtering (1–480 minutes)
        │
        ├── JSON Station Metadata
        │       └── District Parsing & Spatial Tiering (Tier 1 Core)
        │
        ▼
Hourly Arrivals / Departures Aggregation
        │
        ▼
Feature Engineering
(Lags, Rolling Statistics, and Cyclical Time Features)
        │
        ▼
Temporal Split
(Training: July–August | Testing: September)
        │
        ▼
LightGBM Classifier + Threshold Tuning
(Confidence ≥ 0.55)
        │
        ▼
Interactive Streamlit Deployment & Visual Dashboard
```

## Dataset & Feature Summary

The model trains on **38 engineered features** combining spatial metadata, temporal indicators, weather metrics, and historical flow dynamics.

### Feature Groups

- **Spatial:** District one-hot encodings, station capacity, and spatial tier indicators.
- **Temporal:** Hour, day of week, month, rush-hour indicators, and cyclical sine/cosine transformations.
- **Historical flow:** Lagged arrivals and departures, net flow, and three-hour rolling averages.
- **Weather:** Weather variables used to capture conditions that influence ridership demand.

## Model Evaluation & Decision Thresholding

Standard `argmax` classification forces a decision even when predicted probabilities are uncertain. Custom confidence thresholds are applied to extreme pressure states so that operational alerts are issued only when the model has sufficient confidence.

| Pressure State | Baseline Precision | Tuned Precision (≥ 0.55 confidence) | Primary Operational Use Case |
|---|---:|---:|---|
| Low Pressure | 63% | **72%** | Dispatch restocking vans before a station empties |
| Normal State | 53% | 49% | Standard operations |
| High Pressure | 57% | **68%** | Dispatch clear-out vans before a station overflows |

## Repository Structure

```text
├── data/                       # Station metadata and processed trip aggregates
├── notebooks/                  # Exploratory analysis and model development
│   └── toronto-bikeshare-surgical-pressure-prediction.ipynb
├── models/                     # Serialized LightGBM model and feature pickle files
├── app.py                      # Streamlit web application
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/BIG-JAY474/Surgical-Pressure-Predictor.git
cd Surgical-Pressure-Predictor
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

On macOS or Linux:

```bash
source venv/bin/activate
```

On Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the Streamlit application

```bash
streamlit run app.py
```
