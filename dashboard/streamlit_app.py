import streamlit as st
import requests
from PIL import Image

st.set_page_config(
    page_title="NYC Mobility Intelligence Platform",
    layout="wide"
)

# =========================
# HEADER
# =========================

st.title("🚖 NYC Mobility Intelligence Platform")
st.markdown("""
Predict taxi trip duration using a machine learning model trained on **3.6M+ NYC Taxi trips**.
""")

# =========================
# MODEL METRICS
# =========================

st.subheader("📊 Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("MAE", "3.59 min")

with col2:
    st.metric("RMSE", "6.06 min")

with col3:
    st.metric("R² Score", "0.805")

st.divider()

# =========================
# PREDICTION SECTION
# =========================

st.subheader("🔮 Trip Duration Prediction")

col1, col2 = st.columns(2)

with col1:
    passenger_count = st.number_input(
        "Passenger Count",
        min_value=1,
        max_value=10,
        value=1
    )

    trip_distance = st.number_input(
        "Trip Distance",
        min_value=0.1,
        value=5.0
    )

    RatecodeID = st.number_input(
        "RatecodeID",
        min_value=1,
        value=1
    )

    payment_type = st.number_input(
        "Payment Type",
        min_value=1,
        value=1
    )

with col2:
    PULocationID = st.number_input(
        "Pickup Location ID",
        min_value=1,
        value=100
    )

    DOLocationID = st.number_input(
        "Dropoff Location ID",
        min_value=1,
        value=150
    )

    pickup_hour = st.number_input(
        "Pickup Hour",
        min_value=0,
        max_value=23,
        value=10
    )

    pickup_day = st.number_input(
        "Pickup Day",
        min_value=1,
        max_value=31,
        value=15
    )

    pickup_weekday = st.number_input(
        "Pickup Weekday",
        min_value=0,
        max_value=6,
        value=2
    )

    pickup_month = st.number_input(
        "Pickup Month",
        min_value=1,
        max_value=12,
        value=1
    )

if st.button("Predict Duration"):

    payload = {
        "passenger_count": passenger_count,
        "trip_distance": trip_distance,
        "RatecodeID": RatecodeID,
        "PULocationID": PULocationID,
        "DOLocationID": DOLocationID,
        "payment_type": payment_type,
        "pickup_hour": pickup_hour,
        "pickup_day": pickup_day,
        "pickup_weekday": pickup_weekday,
        "pickup_month": pickup_month
    }

   requests.post(
    "http://api:8000/predict",
    json=payload

    )

    prediction = response.json()

    st.success(
        f"Estimated Trip Duration: {prediction['predicted_duration_minutes']} minutes"
    )

st.divider()

# =========================
# VISUALIZATIONS
# =========================

st.subheader("📈 Model Insights")

col1, col2 = st.columns(2)

with col1:
    st.image(
        "outputs/feature_importance.png",
        caption="Feature Importance"
    )

with col2:
    st.image(
        "outputs/actual_vs_predicted.png",
        caption="Actual vs Predicted"
    )

st.divider()

# =========================
# BUSINESS INSIGHTS
# =========================

st.subheader("💡 Key Insights")

st.markdown("""
- **RatecodeID** and **Trip Distance** are the strongest predictors of trip duration.
- Airport and special-fare trips generally have longer travel times.
- Pickup timing significantly impacts trip duration.
- The XGBoost model explains approximately **80%** of trip-duration variability.
- Average prediction error is approximately **3.6 minutes**.
""")

# =========================
# PROJECT OVERVIEW
# =========================

st.subheader("📋 Project Overview")

st.markdown("""
**Dataset:** NYC Yellow Taxi Trips

**Records Processed:** 3.6M+

**Algorithms Evaluated:**
- Linear Regression
- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- CatBoost

**Best Model:** XGBoost Regressor

**Tech Stack:**
- Python
- Pandas
- Scikit-Learn
- XGBoost
- FastAPI
- Streamlit
- Docker (upcoming)
""")