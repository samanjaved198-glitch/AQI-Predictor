import os
from dotenv import load_dotenv
import hopsworks
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from hsml.schema import Schema
from hsml.model_schema import ModelSchema
import joblib

load_dotenv()

FEATURE_COLUMNS = [
    "hour", "day", "month", "day_of_week",
    "pm25", "pm10", "no2", "so2", "co", "o3",
    "temperature", "humidity", "pressure", "wind_speed",
    "aqi_change_rate",
]
TARGET_COLUMN = "target_aqi"


def load_data():
    project = hopsworks.login(
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        project=os.getenv("HOPSWORKS_PROJECT_NAME"),
    )
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name="aqi_features", version=1)
    df = fg.read()
    return df, project


def preprocess(df):
    df = df.dropna(subset=[TARGET_COLUMN])

    for col in FEATURE_COLUMNS:
        if col in df.columns:
            median_val = df[col].median()
            if pd.isna(median_val):
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna(median_val)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y


def evaluate(name, y_test, y_pred):
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\n{name} Results:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE:  {mae:.2f}")
    print(f"  R²:   {r2:.3f}")
    return rmse, mae, r2


def train():
    df, project = load_data()
    print(f"Loaded {len(df)} rows from feature store.")

    if len(df) < 10:
        print("⚠️ Not enough data yet for meaningful training (need more rows). Exiting.")
        return

    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model 1: Ridge Regression
    ridge = Ridge()
    ridge.fit(X_train, y_train)
    ridge_pred = ridge.predict(X_test)
    ridge_rmse, ridge_mae, ridge_r2 = evaluate("Ridge Regression", y_test, ridge_pred)

    # Model 2: Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_rmse, rf_mae, rf_r2 = evaluate("Random Forest", y_test, rf_pred)

    # Pick best model by RMSE
    if rf_rmse < ridge_rmse:
        best_model, best_name, best_rmse, best_mae, best_r2 = rf, "random_forest", rf_rmse, rf_mae, rf_r2
    else:
        best_model, best_name, best_rmse, best_mae, best_r2 = ridge, "ridge", ridge_rmse, ridge_mae, ridge_r2

    print(f"\n🏆 Best model: {best_name} (RMSE: {best_rmse:.2f})")

    # Save locally first — this is what the dashboard actually reads,
    # so it must succeed regardless of what happens with the registry below.
    os.makedirs("trainingpipeline/saved_model", exist_ok=True)
    model_path = f"trainingpipeline/saved_model/{best_name}_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"✅ Model saved locally to {model_path}")

    # Push to Hopsworks Model Registry — wrapped so a registry-side failure
    # (e.g. the recurring HTTP 500 on repeated version creation) doesn't
    # fail the whole pipeline. The model is already safely saved locally above.
    try:
        mr = project.get_model_registry()
        input_schema = Schema(X_train)
        output_schema = Schema(y_train)
        model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)
        model = mr.python.create_model(
            name="aqi_predictor_v2",
            metrics={"rmse": best_rmse, "mae": best_mae, "r2": best_r2},
            description=f"AQI predictor using {best_name}",
            model_schema=model_schema,
        )
        model.save(model_path)
        print("✅ Model registered to Hopsworks Model Registry.")
    except Exception as e:
        print(f"⚠️ Model registry upload failed (continuing anyway — local model is saved): {e}")


if __name__ == "__main__":
    train()