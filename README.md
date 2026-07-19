# 🌫️ Pearls AQI Predictor

An end-to-end, serverless system to predict the Air Quality Index (AQI) for Karachi over the next 3 days, built as part of the 10Pearls Data Science Internship.

## Overview

This project follows the **FTI (Feature, Training, Inference) architecture** pattern — using Hopsworks as the feature store and model registry, GitHub Actions for automated pipeline scheduling, and Streamlit for the live dashboard.

## Architecture
Weather & Pollution APIs (AQICN, OpenWeather)
↓
Feature Pipeline (Python) ──► Hopsworks Feature Store
↓ (hourly, GitHub Actions)          ↓
Training Pipeline (Python) (daily, GitHub Actions)
↓
Hopsworks Model Registry
↓
Streamlit Dashboard (live predictions)

## Tech Stack

- **Language:** Python
- **Data sources:** AQICN API, OpenWeather API
- **Feature Store / Model Registry:** Hopsworks (Serverless free tier)
- **ML Models:** Scikit-learn (Ridge Regression, Random Forest)
- **Automation:** GitHub Actions (cron-based scheduling)
- **Dashboard:** Streamlit
- **Explainability:** SHAP
- **Version control:** Git / GitHub

## Project Structure
AQI Predictor/
├── .github/workflows/     # GitHub Actions (hourly feature pipeline, daily training pipeline)
├── featurepipeline/       # Fetch data, engineer features, push to feature store
├── backfill/               # Historical data backfill script
├── trainingpipeline/      # Model training + evaluation + registry upload
├── app/                    # Streamlit dashboard
├── eda/                     # Exploratory Data Analysis notebook
├── config/                  # Settings/constants
├── .env                    # API keys (not committed)
├── requirements.txt
└── report.md / AQI_Predictor_Report.docx

## Model Performance

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Ridge Regression | 11.80 | 7.53 | 0.865 |
| **Random Forest (best)** | **9.62** | **6.74** | **0.910** |

Trained on 947 historical rows (Kaggle Karachi AQI + weather dataset, 2023) plus ongoing hourly live data collection.

## Features

- 🔄 **Automated hourly feature collection** (AQICN + OpenWeather → Hopsworks Feature Store)
- 🤖 **Daily model retraining** on accumulated data
- 📊 **Live dashboard** — current AQI, category, hazardous alerts, 3-day forecast
- 🔍 **SHAP explainability** — see which factors drive each prediction
- 📈 **EDA notebook** — trends, seasonality, correlations

## Setup

1. Install dependencies:
```bash
   pip install -r requirements.txt
```
2. Add API keys to `.env`:
AQICN_API_KEY=your_key
OPENWEATHER_API_KEY=your_key
HOPSWORKS_API_KEY=your_key
HOPSWORKS_PROJECT_NAME=your_project
CITY_NAME=karachi
3. Run the feature pipeline:
```bash
   python featurepipeline/push_to_store.py
```
4. Train the model:
```bash
   python trainingpipeline/train_model.py
```
5. Launch the dashboard:
```bash
   streamlit run app/dashboard.py
```

## Automation

Both pipelines run automatically via GitHub Actions:
- **Feature Pipeline:** every hour
- **Training Pipeline:** daily

See `.github/workflows/` for the workflow definitions.

## Full Report

See [`AQI_Predictor_Report.docx`](./AQI_Predictor_Report.docx) or `report.md` for full documentation, including challenges faced and how they were resolved.