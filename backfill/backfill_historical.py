import os
from datetime import datetime
from dotenv import load_dotenv
import hopsworks
import pandas as pd

load_dotenv()

CSV_PATH = "backfill/karachi_daily_aqi_weather.csv"


def load_and_transform():
    df = pd.read_csv(CSV_PATH)
    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date").reset_index(drop=True)

    out = pd.DataFrame()
    out["timestamp"] = df["date"]
    out["city"] = "Karachi"

    out["hour"] = 12
    out["day"] = df["date"].dt.day
    out["month"] = df["date"].dt.month
    out["day_of_week"] = df["date"].dt.dayofweek

    out["pm25"] = df["PM2.5"]
    out["pm10"] = df["PM10"]
    out["no2"] = df["NO2"]
    out["so2"] = df["SO2"]
    out["co"] = df["CO"]
    out["o3"] = df["O3"]

    out["temperature"] = df["Temperature"]
    out["humidity"] = df["Humidity"]
    out["pressure"] = None
    out["wind_speed"] = None

    out["aqi_change_rate"] = df["AQI"].diff()
    out["target_aqi"] = df["AQI"]

    out = out.dropna(subset=["aqi_change_rate"]).reset_index(drop=True)

    numeric_cols = ["hour", "day", "month", "day_of_week",
                     "pm25", "pm10", "no2", "so2", "co", "o3",
                     "temperature", "humidity", "pressure", "wind_speed",
                     "aqi_change_rate", "target_aqi"]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")

    return out


def push_backfill():
    project = hopsworks.login(
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        project=os.getenv("HOPSWORKS_PROJECT_NAME"),
    )
    fs = project.get_feature_store()

    df = load_and_transform()
    print(f"Prepared {len(df)} historical rows for backfill.")

    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        description="AQI + weather features for Karachi",
        primary_key=["timestamp"],
        event_time="timestamp",
        time_travel_format="HUDI",
    )

    fg.insert(df)
    print(f"✅ Backfilled {len(df)} rows into feature store.")


if __name__ == "__main__":
    push_backfill()