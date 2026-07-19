import os
import sys
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "featurepipeline"))
from fetch_data import fetch_aqicn_data, fetch_openweather_data

load_dotenv()

MODEL_PATH = "trainingpipeline/saved_model/random_forest_model.pkl"

FEATURE_COLUMNS = [
    "hour", "day", "month", "day_of_week",
    "pm25", "pm10", "no2", "so2", "co", "o3",
    "temperature", "humidity", "pressure", "wind_speed",
    "aqi_change_rate",
]


def aqi_category(aqi):
    if aqi <= 50:
        return "Good", "🟢"
    elif aqi <= 100:
        return "Moderate", "🟡"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "🟠"
    elif aqi <= 200:
        return "Unhealthy", "🔴"
    elif aqi <= 300:
        return "Very Unhealthy", "🟣"
    else:
        return "Hazardous", "🟤"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def get_current_features():
    aqi_data = fetch_aqicn_data()
    weather_data = fetch_openweather_data()
    iaqi = aqi_data.get("iaqi", {})

    def get_val(key):
        return iaqi.get(key, {}).get("v", None)

    now = datetime.now()
    current_aqi = aqi_data.get("aqi")

    features = {
        "hour": now.hour,
        "day": now.day,
        "month": now.month,
        "day_of_week": now.weekday(),
        "pm25": get_val("pm25"),
        "pm10": get_val("pm10"),
        "no2": get_val("no2"),
        "so2": get_val("so2"),
        "co": get_val("co"),
        "o3": get_val("o3"),
        "temperature": weather_data["main"]["temp"],
        "humidity": weather_data["main"]["humidity"],
        "pressure": weather_data["main"]["pressure"],
        "wind_speed": weather_data["wind"]["speed"],
        "aqi_change_rate": 0,
    }
    return features, current_aqi


def predict_next_3_days(model, base_features, current_aqi):
    predictions = []
    last_aqi = current_aqi

    for day_offset in range(1, 4):
        future_date = datetime.now() + timedelta(days=day_offset)
        feats = base_features.copy()
        feats["day"] = future_date.day
        feats["month"] = future_date.month
        feats["day_of_week"] = future_date.weekday()
        feats["aqi_change_rate"] = 0

        X = pd.DataFrame([feats])[FEATURE_COLUMNS]
        X = X.fillna(X.median(numeric_only=True)).fillna(0)

        pred = model.predict(X)[0]
        predictions.append({"date": future_date.strftime("%Y-%m-%d (%A)"), "predicted_aqi": round(pred, 1)})
        last_aqi = pred

    return predictions


# ---------------- UI ----------------

st.set_page_config(page_title="Pearls AQI Predictor", page_icon="🌫️", layout="centered")
st.title("🌫️ Pearls AQI Predictor — Karachi")
st.caption("3-day Air Quality Index forecast, powered by a Random Forest model")

with st.spinner("Fetching current AQI and weather data..."):
    try:
        features, current_aqi = get_current_features()
        model = load_model()
    except Exception as e:
        st.error(f"Failed to fetch live data or load model: {e}")
        st.stop()

category, emoji = aqi_category(current_aqi)

st.subheader("Current AQI")
col1, col2 = st.columns(2)
col1.metric("AQI", f"{current_aqi:.0f}" if current_aqi else "N/A")
col2.metric("Category", f"{emoji} {category}")

if current_aqi and current_aqi > 150:
    st.error(f"⚠️ Hazardous air quality alert! Current AQI ({current_aqi:.0f}) is in the '{category}' range. Limit outdoor exposure.")

st.subheader("3-Day Forecast")
predictions = predict_next_3_days(model, features, current_aqi)

pred_df = pd.DataFrame(predictions)
st.dataframe(pred_df, use_container_width=True, hide_index=True)
st.line_chart(pred_df.set_index("date")["predicted_aqi"])

for p in predictions:
    cat, emo = aqi_category(p["predicted_aqi"])
    st.write(f"**{p['date']}**: {p['predicted_aqi']} AQI — {emo} {cat}")

st.subheader("🔍 Feature Importance (SHAP)")
st.caption("Which factors are driving today's AQI prediction the most?")

with st.spinner("Computing SHAP explanations..."):
    explainer = shap.TreeExplainer(model)

    X_current = pd.DataFrame([features])[FEATURE_COLUMNS]
    X_current = X_current.fillna(X_current.median(numeric_only=True)).fillna(0)

    shap_values = explainer.shap_values(X_current)

    shap_df = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "shap_value": shap_values[0]
    }).sort_values("shap_value", key=abs, ascending=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["red" if v > 0 else "green" for v in shap_df["shap_value"]]
    ax.barh(shap_df["feature"], shap_df["shap_value"], color=colors)
    ax.set_xlabel("Impact on AQI Prediction")
    ax.set_title("SHAP Feature Contributions")
    ax.axvline(x=0, color="black", linewidth=0.8)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)

    st.caption("🔴 Red bars increase predicted AQI | 🟢 Green bars decrease predicted AQI")

st.caption("Model: Random Forest | Trained on historical Karachi AQI + weather data")