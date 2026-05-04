from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np
import json
import os

app = FastAPI()

best_name = open("models/best_model.txt").read().strip()
model_path = "models/rf_model.pkl" if best_name == "RandomForest" else "models/svr_model.pkl"
model = joblib.load(model_path)

class Features(BaseModel):
    slope_degrees: float = Field(..., ge=5, le=45)
    rainfall_mm: float = Field(..., ge=50, le=500)
    soil_depth_m: float = Field(..., ge=0.5, le=5)
    vegetation_index: float = Field(..., ge=0.1, le=0.9)

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}

@app.post("/infer")
def infer(data: Features):
    features = np.array([[data.slope_degrees, data.rainfall_mm, data.soil_depth_m, data.vegetation_index]])
    prediction = float(model.predict(features)[0])

    # Save results on first call with test input
    result = {
        "health_endpoint": "/health",
        "predict_endpoint": "/infer",
        "port": 8888,
        "health_response": {"status": "healthy", "model_loaded": True},
        "test_input": {
            "slope_degrees": data.slope_degrees,
            "rainfall_mm": data.rainfall_mm,
            "soil_depth_m": data.soil_depth_m,
            "vegetation_index": data.vegetation_index
        },
        "prediction": round(prediction, 4)
    }
    os.makedirs("results", exist_ok=True)
    with open("results/step3_s4.json", "w") as f:
        json.dump(result, f, indent=2)

    return {"prediction": round(prediction, 4)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)