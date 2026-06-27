import streamlit as st
import joblib
import pandas as pd
import os
from datetime import date

st.set_page_config(page_title="Predict AQI", page_icon="🤖")

st.write("Current folder:", os.getcwd())

# Load AI Model
model = joblib.load("aqi_model.pkl")

# ---------------------------
# Earth Engine
# ---------------------------
import ee
import json

# Initialize Earth Engine using Streamlit Secrets
if not ee.data.is_initialized():
    service_account = dict(st.secrets["gcp_service_account"])
    credentials = ee.ServiceAccountCredentials(
        service_account["client_email"],
        key_data=json.dumps(service_account)
    )
    ee.Initialize(
        credentials=credentials,
        project=service_account["project_id"]
    )
# ---------------------------
# Kerala Boundary
# ---------------------------
state = st.selectbox(
    "📍 Select State",
    [
        "Kerala",
        "Tamil Nadu",
        "Karnataka",
        "Andhra Pradesh",
        "Telangana",
        "Maharashtra",
        "Gujarat",
        "Delhi"
    ]
)
start_date = st.date_input(
    "📅 Start Date",
    value=date(2024, 1, 1)
)

end_date = st.date_input(
    "📅 End Date",
    value=date(2024, 1, 31)
)

if start_date > end_date:
    st.error("Start Date cannot be after End Date.")
    st.stop()
def fetch_satellite_data():

    state_bounds = {
        "Kerala": [74.85, 8.18, 77.58, 12.79],
        "Tamil Nadu": [76.15, 8.00, 80.35, 13.60],
        "Karnataka": [74.00, 11.50, 78.60, 18.50],
        "Andhra Pradesh": [76.75, 12.60, 84.75, 19.20],
        "Telangana": [77.20, 15.80, 81.00, 19.90],
        "Maharashtra": [72.50, 15.60, 80.90, 22.10],
        "Gujarat": [68.10, 20.10, 74.50, 24.70],
        "Delhi": [76.84, 28.40, 77.35, 28.88]
    }

    region = ee.Geometry.Rectangle(state_bounds[state])

    start = start_date.strftime("%Y-%m-%d")
    end = end_date.strftime("%Y-%m-%d")

    def mean_value(collection, band):
        try:
            image = (
                ee.ImageCollection(collection)
                .select(band)
                .filterDate(start, end)
                .mean()
            )

            value = image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=10000,
                maxPixels=1e9
            ).getInfo()

            return value.get(band, 0)

        except Exception as e:
            st.error(f"{band} failed")
            st.exception(e)
            return 0

    return {
        "CO": mean_value(
            "COPERNICUS/S5P/OFFL/L3_CO",
            "CO_column_number_density"
        ),
        "NO2": mean_value(
            "COPERNICUS/S5P/OFFL/L3_NO2",
            "tropospheric_NO2_column_number_density"
        ),
        "SO2": mean_value(
            "COPERNICUS/S5P/OFFL/L3_SO2",
            "SO2_column_number_density"
        ),
        "O3": mean_value(
            "COPERNICUS/S5P/OFFL/L3_O3",
            "O3_column_number_density"
        ),
        "HCHO": mean_value(
            "COPERNICUS/S5P/OFFL/L3_HCHO",
            "tropospheric_HCHO_column_number_density"
        )
    }



    # ---------------------------
# Streamlit UI
# ---------------------------

st.title("🤖 AQI Prediction using Live Satellite Data")

st.write("Fetch live Sentinel-5P data and predict AQI.")

if st.button(f"🛰 Fetch Satellite Data for {state}"):

    with st.spinner("Fetching live satellite data..."):
        sat = fetch_satellite_data()

    st.success("Satellite data fetched successfully!")

    st.write("### 📍 Selected Region")
    st.write(state)

    st.write("### 📅 Date Range")
    st.write(f"{start_date} → {end_date}")

    st.subheader("Live Satellite Data")

    st.metric("CO", sat["CO"])
    st.metric("NO₂", sat["NO2"])
    st.metric("SO₂", sat["SO2"])
    st.metric("O₃", sat["O3"])
    st.metric("HCHO", sat["HCHO"])

    try:
        prediction = model.predict([[
            sat["CO"],
            sat["NO2"],
            sat["SO2"],
            sat["O3"],
            sat["HCHO"]
        ]])

        aqi = int(prediction[0])

        st.success("✅ AQI predicted successfully")

        history = pd.DataFrame([{
            "State": state,
            "Date": str(end_date),
            "AQI": int(aqi),
            "HCHO": float(sat["HCHO"]),
            "CO": float(sat["CO"]),
            "NO2": float(sat["NO2"]),
            "SO2": float(sat["SO2"]),
            "O3": float(sat["O3"])
        }])

        history.to_csv(
            "history.csv",
            mode="a",
            header=not os.path.exists("history.csv"),
            index=False
        )

        st.success("✅ History saved successfully")

        st.header(f"Predicted AQI : {aqi}")

        if aqi <= 50:
            st.success("🟢 Good")
        elif aqi <= 100:
            st.info("🟡 Satisfactory")
        elif aqi <= 200:
            st.warning("🟠 Moderate")
        elif aqi <= 300:
            st.warning("🔴 Poor")
        elif aqi <= 400:
            st.error("🟣 Very Poor")
        else:
            st.error("⚫ Severe")

    except Exception as e:
        st.error("Prediction Failed")
        st.exception(e)

st.markdown("---")
st.caption("AirGuardian AI • Live Sentinel-5P Satellite Monitoring")






       

