import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import json

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Theni Dengue Prediction System",
    page_icon="🦟",
    layout="wide"
)

# ==================================================
# SESSION STATE
# ==================================================
if "page" not in st.session_state:
    st.session_state.page = "control"

# ==================================================
# LOAD DATA (DATASET ALREADY THENI)
# ==================================================
data = pd.read_csv("data/final_dengue_data.csv")
data["date"] = pd.to_datetime(data["date"])

theni_data = data.copy()

if theni_data.empty:
    st.error("❌ No data found for Theni district")
    st.stop()

theni_data = theni_data.sort_values("date")
theni_data["time_index"] = np.arange(len(theni_data))

# ==================================================
# ASSIGN TALUK BASED ON LOCATION (SIMPLIFIED LOGIC)
# ==================================================
def assign_taluk(lat, lon):
    if lat > 10.7:
        return "Periyakulam"
    elif lon < 77.1:
        return "Bodinayakanur"
    elif lon > 77.5:
        return "Andipatti"
    elif lat < 10.2:
        return "Uthamapalayam"
    else:
        return "Theni"

theni_data["taluk"] = theni_data.apply(
    lambda row: assign_taluk(row.latitude, row.longitude),
    axis=1
)

taluk_summary = theni_data.groupby("taluk")["dengue_cases"].mean().reset_index()

def taluk_color(cases):
    if cases < 30:
        return "green"
    elif cases < 60:
        return "orange"
    else:
        return "red"

# ==================================================
# LOAD THENI DISTRICT BOUNDARY
# ==================================================
with open("data/theni_boundary.geojson", "r", encoding="utf-8") as f:
    theni_boundary = json.load(f)

# ==================================================
# MACHINE LEARNING MODELS
# ==================================================
X = theni_data[["temperature", "humidity", "rainfall"]]
y = theni_data["dengue_cases"]

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X, y)

ts_model = LinearRegression()
ts_model.fit(theni_data[["time_index"]], y)

# ==================================================
# PAGE 1 — CONTROL PANEL
# ==================================================
if st.session_state.page == "control":

    st.title("🛠️ Theni Dengue Risk Control Panel")
    st.markdown("### Enter Environmental Conditions for Theni District")

    col1, col2, col3 = st.columns(3)
    with col1:
        temp = st.slider("🌡 Temperature (°C)", 20, 40, 30)
    with col2:
        humidity = st.slider("💧 Humidity (%)", 40, 90, 70)
    with col3:
        rainfall = st.slider("🌧 Rainfall (mm)", 0, 200, 50)

    if st.button("🚀 Run AI Prediction", use_container_width=True):
        st.session_state.predicted_cases = int(
            rf_model.predict([[temp, humidity, rainfall]])[0]
        )
        st.session_state.page = "analytics"

# ==================================================
# PAGE 2 — ANALYTICS
# ==================================================
elif st.session_state.page == "analytics":

    st.title("📊 Theni Dengue Risk Analytics")

    cases = st.session_state.predicted_cases

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Cases", cases)
    col2.metric(
        "Risk Level",
        "High" if cases > 60 else "Moderate" if cases > 30 else "Low"
    )
    col3.metric("District", "Theni")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=cases,
        title={"text": "Outbreak Severity"},
        gauge={
            "axis": {"range": [0, 100]},
            "steps": [
                {"range": [0, 30], "color": "green"},
                {"range": [30, 60], "color": "yellow"},
                {"range": [60, 100], "color": "red"},
            ],
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

    if st.button("➡ View Theni GIS Map"):
        st.session_state.page = "gis"

# ==================================================
# PAGE 3 — GIS MAP (BOUNDARY + TALUK ZONES)
# ==================================================
elif st.session_state.page == "gis":

    st.title("🗺️ Theni District Dengue Risk Map")

    center = [theni_data.latitude.mean(), theni_data.longitude.mean()]
    m = folium.Map(location=center, zoom_start=10, tiles="cartodbpositron")

    # District boundary
    folium.GeoJson(
        theni_boundary,
        style_function=lambda x: {
            "fillColor": "none",
            "color": "blue",
            "weight": 3
        }
    ).add_to(m)

    # Taluk risk zones
    for taluk in taluk_summary.itertuples():
        subset = theni_data[theni_data["taluk"] == taluk.taluk]
        lat, lon = subset.latitude.mean(), subset.longitude.mean()

        folium.Circle(
            location=[lat, lon],
            radius=5000,
            color=taluk_color(taluk.dengue_cases),
            fill=True,
            fill_opacity=0.25,
            popup=f"""
            <b>Taluk:</b> {taluk.taluk}<br>
            <b>Avg Cases:</b> {int(taluk.dengue_cases)}<br>
            <b>Risk:</b> {taluk_color(taluk.dengue_cases).upper()}
            """
        ).add_to(m)

    # Point markers
    for _, row in theni_data.iterrows():
        folium.CircleMarker(
            location=[row.latitude, row.longitude],
            radius=6,
            color="red",
            fill=True,
            fill_opacity=0.8
        ).add_to(m)

    st_folium(m, width=900, height=500)

    if st.button("➡ View Heatmap"):
        st.session_state.page = "heatmap"

# ==================================================
# PAGE 4 — HEATMAP
# ==================================================
elif st.session_state.page == "heatmap":

    st.title("🔥 Theni Dengue Density Heatmap")

    hm = folium.Map(
        location=[theni_data.latitude.mean(), theni_data.longitude.mean()],
        zoom_start=10,
        tiles="cartodbpositron"
    )

    folium.GeoJson(
        theni_boundary,
        style_function=lambda x: {
            "fillColor": "none",
            "color": "blue",
            "weight": 3
        }
    ).add_to(hm)

    HeatMap(
        [[row.latitude, row.longitude, row.dengue_cases * 3]
         for _, row in theni_data.iterrows()],
        radius=35,
        blur=25
    ).add_to(hm)

    st_folium(hm, width=900, height=500)

    if st.button("➡ Forecast Dengue Trend"):
        st.session_state.page = "forecast"

# ==================================================
# PAGE 5 — FORECAST
# ==================================================
elif st.session_state.page == "forecast":

    st.title("📈 Theni Dengue Forecast (Next 6 Months)")

    future_steps = 6
    last_index = theni_data.time_index.max()

    future_index = np.arange(last_index + 1, last_index + future_steps + 1)
    future_dates = pd.date_range(
        start=theni_data.date.max(),
        periods=future_steps + 1,
        freq="M"
    )[1:]

    future_predictions = ts_model.predict(future_index.reshape(-1, 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=theni_data.date,
        y=theni_data.dengue_cases,
        name="Historical Cases"
    ))
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_predictions,
        name="Forecasted Cases",
        line=dict(dash="dash")
    ))

    st.plotly_chart(fig, use_container_width=True)

    if max(future_predictions) > 60:
        st.error("🚨 High outbreak risk expected in Theni")
    elif max(future_predictions) > 30:
        st.warning("⚠️ Moderate dengue rise expected")
    else:
        st.success("✅ Stable dengue trend predicted")

