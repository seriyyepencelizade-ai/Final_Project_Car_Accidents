import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Road Accident Severity Predictor", page_icon="🚗", layout="centered")
st.title("🚗 Road Accident Severity Predictor")
st.markdown("*Please Enter Features:*")

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "../notebooks/accident_model.pkl")
model = joblib.load(model_path)

col1, col2 = st.columns(2)

with col1:
    longitude = st.number_input("Enter Longitude:", value=-0.12, format="%.5f")
    latitude = st.number_input("Enter Latitude:", value=51.50, format="%.5f")
    hour = st.selectbox("Select Hour:", list(range(24)))
    month = st.selectbox("Select Month:", list(range(1, 13)))
    number_of_vehicles = st.number_input("Enter Number of Vehicles:", min_value=1, max_value=10, value=2)
    number_of_casualties = st.number_input("Enter Number of Casualties:", min_value=0, max_value=20, value=1)

with col2:
    temperature_2m = st.number_input("Enter Temperature (°C):", value=15.0)
    wind_speed_10m = st.number_input("Enter Wind Speed (km/h):", value=15.0)
    cloud_cover = st.number_input("Enter Cloud Cover (%):", min_value=0, max_value=100, value=50)
    rain = st.number_input("Enter Rain (mm):", value=0.0)
    european_aqi = st.number_input("Enter European AQI:", min_value=0, max_value=300, value=50)
    nitrogen_dioxide = st.number_input("Enter Nitrogen Dioxide (NO₂):", min_value=0.0, value=0.0)
    pm10 = st.number_input("Enter PM10:", min_value=0.0, value=0.0)
    pm2_5 = st.number_input("Enter PM2.5:", min_value=0.0, value=0.0)

if st.button("Predict Accident Severity"):
    st.write("Model feature names:", model.feature_names_in_.tolist())
    input_df = pd.DataFrame({
    "longitude": [longitude],
    "latitude": [latitude],
    "nitrogen_dioxide": [nitrogen_dioxide],
    "temperature_2m": [temperature_2m],
    "wind_speed_10m": [wind_speed_10m],
    "pm10": [pm10],
    "pm2_5": [pm2_5],
    "european_aqi": [european_aqi],
    "number_of_vehicles": [number_of_vehicles],
    "hour": [hour],
    "cloud_cover": [cloud_cover],
    "month": [month],
    "year": [2015],
    "rain": [rain],
    "number_of_casualties": [number_of_casualties]
})

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.divider()
    st.subheader("Prediction Result:")

    if prediction == 1:
        st.error(f"🚨 Severe Accident — Probability: {probability*100:.1f}%")
    else:
        st.success(f"✅ Non-Severe Accident — Probability: {probability*100:.1f}%")