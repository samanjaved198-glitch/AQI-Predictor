import os
from datetime import datetime
from dotenv import load_dotenv
from fetch_data import fetch_aqicn_data, fetch_openweather_data

load_dotenv()


def compute_features(previous_aqi=None):
    """Fetch raw data and compute structured features + target."""
    aqi_data = fetch_aqicn_data()
    weather_data = fetch_openweather_data()

    now = datetime.now()
    iaqi = aqi_data.get("iaqi", {})

    def get_val(key):
        return iaqi.get(key, {}).get("v", None)

    current_aqi = aqi_data.get("aqi")

    # AQI change rate (needs previous reading)
    aqi_change_rate = None
    if previous_aqi is not None and current_aqi is not None:
        aqi_change_rate = current_aqi - previous_aqi

    features = {
        # Identifiers
        "timestamp": now.isoformat(),
        "city": aqi_data.get("city", {}).get("name"),

        # Time-based features
        "hour": now.hour,
        "day": now.day,
        "month": now.month,
        "day_of_week": now.weekday(),

        # Pollutant features (from AQICN iaqi)
        "pm25": get_val("pm25"),
        "pm10": get_val("pm10"),
        "no2": get_val("no2"),
        "so2": get_val("so2"),
        "co": get_val("co"),
        "o3": get_val("o3"),

        # Weather features (from OpenWeather)
        "temperature": weather_data["main"]["temp"],
        "humidity": weather_data["main"]["humidity"],
        "pressure": weather_data["main"]["pressure"],
        "wind_speed": weather_data["wind"]["speed"],

        # Derived feature
        "aqi_change_rate": aqi_change_rate,

        # Target (what we want to predict)
        "target_aqi": current_aqi,
    }

    return features


if __name__ == "__main__":
    result = compute_features(previous_aqi=155)  # dummy previous value for testing
    for key, value in result.items():
        print(f"{key}: {value}")