import mlflow
import mlflow.sklearn
import json
import os

mlflow.set_experiment("geosurvey-land-stability-score")

run_id = open("models/best_run_id.txt").read().strip()

# Read best RMSE from step1 results
with open("results/step1_s1.json") as f:
    step1 = json.load(f)
best_rmse = step1["best_metric_value"]

model_uri = f"runs:/{run_id}/model"
registered = mlflow.register_model(model_uri, "geosurvey-land-stability-score-predictor")

result = {
    "registered_model_name": "geosurvey-land-stability-score-predictor",
    "version": int(registered.version),
    "run_id": run_id,
    "source_metric": "rmse",
    "source_metric_value": best_rmse
}

os.makedirs("results", exist_ok=True)
with open("results/step4_s6.json", "w") as f:
    json.dump(result, f, indent=2)

print("Task 4 done:", json.dumps(result, indent=2))