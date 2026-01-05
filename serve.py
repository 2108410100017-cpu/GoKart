from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json
from datetime import datetime
import uuid
import glob
import os

app = FastAPI()

class InputData(BaseModel):
    temperature: int
    time_of_day: int

@app.post("/predict")
def predict(data: InputData):
    # Load latest model
    model_files = glob.glob('model_v*.pkl')
    if not model_files:
        raise ValueError("No model found.")
    versions = [int(f.split('_v')[1].split('.')[0]) for f in model_files]
    latest_version = max(versions)
    model_path = f'model_v{latest_version}.pkl'
    model = joblib.load(model_path)

    # Predict
    input_array = [[data.temperature, data.time_of_day]]
    prediction = model.predict(input_array)[0]

    # Log
    input_features = data.dict()
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "model_version": f"v{latest_version}",
        "input_features": input_features,
        "prediction": int(prediction),
        "request_id": str(uuid.uuid4())
    }
    with open('live_data.jsonl', 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

    return {"prediction": int(prediction)}