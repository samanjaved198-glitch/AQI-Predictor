import os
from datetime import datetime
from dotenv import load_dotenv
import hopsworks
import pandas as pd
from feature_engineering import compute_features

load_dotenv()


def get_last_aqi(fs):
    """Try to get the most recent AQI value from the feature store for change-rate calculation."""
    try:
        fg = fs.get_feature_group(name="aqi_features", version=1)
        df = fg.read()
        if len(df) > 0:
            df_sorted = df.sort_values("timestamp", ascending=False)
            return df_sorted.iloc[0]["target_aqi"]
    except Exception:
        pass
    return None


def push_features():
    project = hopsworks.login(
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        project=os.getenv("HOPSWORKS_PROJECT_NAME"),
    )
    fs = project.get_feature_store()

    previous_aqi = get_last_aqi(fs)
    features = compute_features(previous_aqi=previous_aqi)

    df = pd.DataFrame([features])
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    numeric_cols = ["pm25", "pm10", "no2", "so2", "co", "o3",
                     "temperature", "humidity", "pressure", "wind_speed",
                     "aqi_change_rate", "target_aqi"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    df["hour"] = df["hour"].astype("float64")
    df["day"] = df["day"].astype("float64")
    df["month"] = df["month"].astype("float64")
    df["day_of_week"] = df["day_of_week"].astype("float64")

    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        description="AQI + weather features for Karachi",
        primary_key=["timestamp"],
        event_time="timestamp",
        time_travel_format="HUDI",
    )

    fg.insert(df)
    print(f"✅ Pushed 1 row to feature store. AQI: {features['target_aqi']}")


if __name__ == "__main__":
    push_features()