import os
import requests
from dotenv import load_dotenv

load_dotenv()

AQICN_API_KEY = os.getenv("AQICN_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
CITY_NAME = os.getenv("CITY_NAME", "karachi")


def fetch_aqicn_data(city=CITY_NAME):
    """Fetch AQI + pollutant data from AQICN."""
    url = f"https://api.waqi.info/feed/{city}/?token={AQICN_API_KEY}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    if data.get("status") != "ok":
        raise Exception(f"AQICN API error: {data}")

    return data["data"]


def fetch_openweather_data(city=CITY_NAME):
    """Fetch weather data from OpenWeather."""
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    print("Fetching AQICN data...")
    aqi_data = fetch_aqicn_data()
    print("AQI:", aqi_data.get("aqi"))
    print("Station:", aqi_data.get("city", {}).get("name"))
    print("Pollutants (iaqi):", aqi_data.get("iaqi"))

    print("\nFetching OpenWeather data...")
    weather_data = fetch_openweather_data()
    print("Temperature:", weather_data["main"]["temp"])
    print("Humidity:", weather_data["main"]["humidity"])
    print("Wind speed:", weather_data["wind"]["speed"])