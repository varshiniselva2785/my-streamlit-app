import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import json
import random
import smtplib
from email.mime.text import MIMEText

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Denguard AI – Outbreak Intelligence System",
    page_icon="🦟",
    layout="wide"
)

# ==================================================
# TALUK OFFICER CONTACT DATABASE (8 TALUKS)
# ==================================================
officer_contacts = {
    "Theni": {"officer": "Ragul R", "email": "rahul@gmail.com"},
    "Periyakulam": {"officer": "Divya M", "email": "divya@gmail.com"},
    "Andipatti": {"officer": "Karthik R", "email": "karthik@gmail.com"},
    "Bodinayakanur": {"officer": "Ananya P", "email": "ananya@gmail.com"},
    "Veerapandi": {"officer": "Officer 5", "email": "mail5@gmail.com"},
    "Cumbum": {"officer": "Officer 6", "email": "mail6@gmail.com"},
    "Chinnamanur": {"officer": "Officer 7", "email": "mail7@gmail.com"},
    "Uthamapalayam": {"officer": "Officer 8", "email": "mail8@gmail.com"},
}

# ==================================================
# LOAD DATA
# ==================================================
@st.cache_data
def load_data():
    data = pd.read_csv("data/final_dengue_data.csv")
    data["date"] = pd.to_datetime(data["date"])
    return data

@st.cache_resource
def train_models(data):
    data = data.sort_values("date")
    data["time_index"] = np.arange(len(data))

    X = data[["temperature", "humidity", "rainfall"]]
    y = data["dengue_cases"]

    rf = RandomForestRegressor(random_state=42)
    rf.fit(X, y)

    ts = LinearRegression()
    ts.fit(data[["time_index"]], y)

    return rf, ts, data

data = load_data()
rf_model, ts_model, theni_data = train_models(data)

# ==================================================
# AUTO CREATE TALUK + COORDINATES (IF NOT PRESENT)
# ==================================================
if "taluk" not in theni_data.columns:

    taluks = {
        "Theni": (10.80, 77.48),
        "Periyakulam": (10.90, 77.55),
        "Andipatti": (10.75, 77.60),
        "Bodinayakanur": (10.60, 77.35),
        "Veerapandi": (10.60, 77.50),
        "Cumbum": (10.45, 77.25),
        "Chinnamanur": (10.45, 77.45),
        "Uthamapalayam": (10.30, 77.30),
    }

    taluk_list = list(taluks.keys())

    theni_data["taluk"] = [
        random.choice(taluk_list) for _ in range(len(theni_data))
    ]

    theni_data["latitude"] = [
        taluks[t][0] + np.random.uniform(-0.02, 0.02)
        for t in theni_data["taluk"]
    ]

    theni_data["longitude"] = [
        taluks[t][1] + np.random.uniform(-0.02, 0.02)
        for t in theni_data["taluk"]
    ]

# ==================================================
# EMAIL ALERT FUNCTION (SEND FOR ALL RISK LEVELS)
# ==================================================
def send_location_alert(taluk, predicted_cases, risk):

    officer = officer_contacts[taluk]

    sender_email = st.secrets["kalamvisionaries@gmail.com"]
    sender_password = st.secrets["ddlgkfzlxufzwciz"]

    subject = f"{risk} Risk Dengue Alert - {taluk}"

    body = f"""
DENGUE RISK ALERT

Taluk: {taluk}
Officer: {officer['officer']}
Predicted Cases: {predicted_cases}
Risk Level: {risk}

Please take appropriate preventive measures.

- Denguard AI System
"""

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = officer["email"]

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, officer["email"], msg.as_string())
        server.quit()
        return "Email Sent Successfully ✅"
    except Exception as e:
        return f"Email Error: {e}"

# ==================================================
# LOAD GEOJSON
# ==================================================
with open("data/theni_boundary.geojson", "r", encoding="utf-8") as f:
    theni_boundary = json.load(f)

# ==================================================
# TABS
# ==================================================
tab1, tab2, tab3 = st.tabs([
    "🧠 Prediction",
    "🗺 GIS Intelligence",
    "📈 Forecast Engine"
])

# ==================================================
# TAB 1 – PREDICTION
# ==================================================
with tab1:

    st.subheader("Dengue Risk Prediction Engine")

    selected_taluk = st.selectbox(
        "Select Taluk",
        list(officer_contacts.keys())
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        temp = st.slider("Temperature", 20, 40, 30)
    with col2:
        humidity = st.slider("Humidity", 40, 90, 70)
    with col3:
        rainfall = st.slider("Rainfall", 0, 200, 50)

    if st.button("Run AI Model", use_container_width=True):

        predicted = int(
            rf_model.predict([[temp, humidity, rainfall]])[0]
        )

        risk = "Low" if predicted <= 30 else "Moderate" if predicted <= 60 else "High"

        colA, colB = st.columns(2)
        colA.metric("Predicted Cases", predicted)
        colB.metric("Risk Level", risk)

        # 🔥 SEND EMAIL FOR ALL RISK LEVELS
        status = send_location_alert(selected_taluk, predicted, risk)

        if risk == "High":
            st.error("HIGH RISK ZONE DETECTED 🚨")
        elif risk == "Moderate":
            st.warning("Moderate Risk – Increase monitoring.")
        else:
            st.success("Low Risk – Maintain preventive measures.")

        st.info(status)

# ==================================================
# TAB 2 – GIS MAP
# ==================================================
with tab2:

    st.subheader("Theni District Spatial Intelligence")

    selected_taluk = st.selectbox(
        "Select Taluk",
        options=["All"] + sorted(theni_data["taluk"].unique())
    )

    m = folium.Map(location=[10.75, 77.45], zoom_start=11)

    folium.GeoJson(
        theni_boundary,
        style_function=lambda x: {
            "fillColor": "#00000000",
            "color": "#00f7ff",
            "weight": 2,
        }
    ).add_to(m)

    subset = theni_data if selected_taluk == "All" else theni_data[theni_data["taluk"] == selected_taluk]

    for _, r in subset.iterrows():

        if r.dengue_cases > 60:
            color = "red"
        elif r.dengue_cases > 30:
            color = "orange"
        else:
            color = "green"

        folium.CircleMarker(
            location=[r.latitude, r.longitude],
            radius=6,
            color=color,
            fill=True,
            fill_opacity=0.8,
            popup=f"{r.taluk} | Cases: {r.dengue_cases}"
        ).add_to(m)

    st_folium(m, width=1100, height=600)

# ==================================================
# TAB 3 – FORECAST
# ==================================================
with tab3:

    st.subheader("AI Outbreak Forecast (Next 6 Months)")

    future_steps = 6
    future_index = np.arange(
        theni_data.time_index.max() + 1,
        theni_data.time_index.max() + future_steps + 1
    )

    future_dates = pd.date_range(
        start=theni_data.date.max(),
        periods=future_steps + 1,
        freq="ME"
    )[1:]

    future_predictions = ts_model.predict(
        future_index.reshape(-1, 1)
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=theni_data.date,
        y=theni_data.dengue_cases,
        name="Historical"
    ))
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_predictions,
        name="Forecast",
        line=dict(dash="dash")
    ))

    st.plotly_chart(fig, use_container_width=True)