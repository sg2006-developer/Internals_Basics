import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import json, os
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/training_data.csv")
X = df[["slope_degrees", "rainfall_mm", "soil_depth_m", "vegetation_index"]]
y = df["land_stability_score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("geosurvey-land-stability-score")

results = []

# --- SVR ---
svr_params = {"kernel": "rbf", "C": 100, "epsilon": 0.1}
with mlflow.start_run(run_name="SVR") as run:
    mlflow.set_tag("priority", "high")
    model = SVR(**svr_params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mlflow.log_params(svr_params)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.sklearn.log_model(model, "model")
    svr_run_id = run.info.run_id
    results.append({"name": "SVR", "mae": round(mae, 4), "rmse": round(rmse, 4), "run_id": svr_run_id})
    joblib.dump(model, "models/svr_model.pkl")

# --- RandomForest ---
rf_params = {"n_estimators": 100, "random_state": 42}
with mlflow.start_run(run_name="RandomForest") as run:
    mlflow.set_tag("priority", "high")
    model = RandomForestRegressor(**rf_params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mlflow.log_params(rf_params)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.sklearn.log_model(model, "model")
    rf_run_id = run.info.run_id
    results.append({"name": "RandomForest", "mae": round(mae, 4), "rmse": round(rmse, 4), "run_id": rf_run_id})
    joblib.dump(model, "models/rf_model.pkl")

# Pick best by RMSE
best = min(results, key=lambda x: x["rmse"])

# Save best model name for other scripts
with open("models/best_model.txt", "w") as f:
    f.write(best["name"])
with open("models/best_run_id.txt", "w") as f:
    f.write(best["run_id"])

step1 = {
    "experiment_name": "geosurvey-land-stability-score",
    "models": [{"name": r["name"], "mae": r["mae"], "rmse": r["rmse"]} for r in results],
    "best_model": best["name"],
    "best_metric_name": "rmse",
    "best_metric_value": best["rmse"]
}
with open("results/step1_s1.json", "w") as f:
    json.dump(step1, f, indent=2)

print("Task 1 done:", json.dumps(step1, indent=2))