# Toronto Bike Share: Surgical Pressure Predictor

**A 41-feature machine learning model for real-time station rebalancing.**

### 📊 Project Overview

This project addresses the operational challenge of station "dock-outs" (empty stations) and "overflows" (full stations). By prioritizing **Recall over Accuracy**, the model acts as an early-warning system to identify high-pressure events 2 hours in advance, specifically tuned for the volatile transit patterns of September in Toronto.

### 🛠 Technical Architecture

The model utilizes a **Random Forest Classifier** optimized for zero-lag inference in a web environment.

* **Feature Vector (41 dimensions):**
* **Temporal:** Cyclical hour encoding (Sin/Cos), weekend flags, and rush hour identification.
* **Weather:** Interaction terms (`temp * is_rush_hour`) and precipitation/wind metrics.
* **Momentum:** 1h and 3h net flow trends and station capacity features.
* **Spatial:** One-hot encoded data for 27 unique Toronto districts.


* **Model Size:** 22MB (Pruned for Streamlit Cloud deployment).
* **Data Source:** Toronto Open Data (Bike Share Ridership July–August 2024).

### 📈 Performance Metrics

The model is biased toward operational action, favoring the detection of extreme supply shifts.

| Class | Precision | Recall | F1-Score |
| --- | --- | --- | --- |
| **Low Supply** | 0.49 | **0.66** | 0.56 |
| **High Supply** | 0.47 | **0.61** | 0.53 |
| **Normal** | 0.54 | 0.32 | 0.40 |

**Operational Logic:** The high recall (0.66/0.61) ensures that 60%+ of potential station crises are identified for intervention.

### 🚀 Deployment & Usage

The app is live and accepts real-time inputs for logistics planning.

1. **Select District:** Choose from the 27 monitored Toronto sectors.
2. **Input Conditions:** Provide current weather and last-hour flow data.
3. **Actionable Output:** The model outputs a categorical result (Low/Normal/High) with specific dispatch recommendations.

**Repository Structure:**

* `app.py`: Streamlit application script.
* `commuter_pulse_model.pkl`: Trained Random Forest model.
* `model_features.pkl`: Serialized list of the 41 features.
* `requirements.txt`: Python dependencies.

### 🔗 Links

* **Live Application:** `[(https://surgical-pressure-predictor-7mpkvqe2pexeybrz6gcure.streamlit.app/)]`
* **Training Notebook:** `[Your Kaggle URL]`

---
