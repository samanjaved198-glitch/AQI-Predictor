<<<<<<< HEAD
# Pearls AQI Predictor

End-to-end serverless system to predict Air Quality Index (AQI) for the next 3 days.

## Architecture
- **Feature Pipeline**: Fetches weather + pollutant data, computes features, stores in Hopsworks Feature Store
- **Training Pipeline**: Trains ML models (Ridge Regression, Random Forest, TensorFlow) on historical features
- **Automation**: GitHub Actions runs feature pipeline hourly, training pipeline daily
- **Dashboard**: Streamlit app showing 3-day AQI forecast with SHAP explanations and hazard alerts

## Tech Stack
Python, Scikit-learn, TensorFlow, Hopsworks, GitHub Actions, Streamlit, AQICN/OpenWeather APIs, SHAP

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Add API keys to `.env`
3. Run feature pipeline, then training pipeline
4. Launch dashboard: `streamlit run app/dashboard.py`
=======
# AQI-Predictor
>>>>>>> 151bbd193bd6d9ad051acc1ef4770073b3b6f76d
