import argparse
import joblib
import numpy as np
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--slope_degrees", type=float, required=True)
parser.add_argument("--rainfall_mm", type=float, required=True)
parser.add_argument("--soil_depth_m", type=float, required=True)
parser.add_argument("--vegetation_index", type=float, required=True)
args = parser.parse_args()

# Load best model
best_name = open("models/best_model.txt").read().strip()
model_path = "models/rf_model.pkl" if best_name == "RandomForest" else "models/svr_model.pkl"
model = joblib.load(model_path)

features = np.array([[args.slope_degrees, args.rainfall_mm, args.soil_depth_m, args.vegetation_index]])
prediction = float(model.predict(features)[0])

result = {
    "image_name": "geosurvey-predictor",
    "image_tag": "v1",
    "base_image": "python:3.12-slim",
    "test_input": {
        "slope_degrees": args.slope_degrees,
        "rainfall_mm": args.rainfall_mm,
        "soil_depth_m": args.soil_depth_m,
        "vegetation_index": args.vegetation_index
    },
    "prediction": round(prediction, 4)
}

os.makedirs("results", exist_ok=True)
with open("results/step2_s3.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))