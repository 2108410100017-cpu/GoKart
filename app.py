import json
import uuid
import glob
from datetime import datetime

import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import mlflow
import mlflow.sklearn

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# -------------------------------------------------
# FastAPI App
# -------------------------------------------------
app = FastAPI(
    title="Smart Lemonade Stand MLOps Demo",
    description="End-to-end ML feedback loop with MLflow",
    version="1.0"
)

# -------------------------------------------------
# MLflow Config
# -------------------------------------------------
mlflow.set_experiment("lemonade-stand-experiment")

# -------------------------------------------------
# Input Schema
# -------------------------------------------------
class PredictionInput(BaseModel):
    temperature: int
    time_of_day: int  # 0–23

# -------------------------------------------------
# Load Latest Model
# -------------------------------------------------
def load_latest_model():
    model_files = glob.glob("model_v*.pkl")
    if not model_files:
        raise HTTPException(
            status_code=500,
            detail="No trained model found. Run initial training first."
        )

    versions = [int(f.split("_v")[1].split(".")[0]) for f in model_files]
    latest_version = max(versions)

    model_path = f"model_v{latest_version}.pkl"
    model = joblib.load(model_path)

    return model, latest_version, model_path

# -------------------------------------------------
# Prediction Endpoint
# -------------------------------------------------
@app.post("/predict")
def predict(data: PredictionInput):
    model, version, _ = load_latest_model()

    input_array = [[data.temperature, data.time_of_day]]
    prediction = int(model.predict(input_array)[0])

    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "model_version": f"v{version}",
        "input_features": data.dict(),
        "prediction": prediction,
        "request_id": str(uuid.uuid4())
    }

    with open("live_data.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return {
        "prediction": prediction,
        "model_version": f"v{version}"
    }

# -------------------------------------------------
# Ground Truth Simulation
# -------------------------------------------------
@app.post("/cycle/simulate-ground-truth")
def simulate_ground_truth():
    try:
        with open("live_data.jsonl", "r") as f:
            lines = [json.loads(line) for line in f]
    except FileNotFoundError:
        return {"status": "no live data to process"}

    processed = []

    for entry in lines:
        temp = entry["input_features"]["temperature"]
        time = entry["input_features"]["time_of_day"]

        prob = (temp - 10) / 30.0 * 0.5
        if 10 <= time <= 18:
            prob += 0.5
        prob = np.clip(prob, 0, 1)

        ground_truth = int(np.random.binomial(1, prob))
        entry["ground_truth"] = ground_truth
        processed.append(entry)

    with open("processed_data.jsonl", "a") as f:
        for entry in processed:
            f.write(json.dumps(entry) + "\n")

    open("live_data.jsonl", "w").close()

    return {
        "status": "success",
        "processed_entries": len(processed)
    }

# -------------------------------------------------
# Create Latest Dataset
# -------------------------------------------------
@app.post("/cycle/create-dataset")
def create_dataset():
    data = []

    try:
        with open("processed_data.jsonl", "r") as f:
            for line in f:
                entry = json.loads(line)
                data.append({
                    "temperature": entry["input_features"]["temperature"],
                    "time_of_day": entry["input_features"]["time_of_day"],
                    "will_buy": entry["ground_truth"]
                })
    except FileNotFoundError:
        return {"status": "no processed data"}

    if not data:
        return {"status": "no data"}

    df = pd.DataFrame(data)
    df.to_csv("latest_dataset.csv", index=False)

    return {
        "status": "success",
        "rows": len(df)
    }

# -------------------------------------------------
# Retrain Model (MLflow Integrated)
# -------------------------------------------------
@app.post("/cycle/retrain")
def retrain():
    try:
        current_model, current_version, _ = load_latest_model()
    except:
        return {"status": "no model to compare against"}

    try:
        df = pd.read_csv("latest_dataset.csv")
    except FileNotFoundError:
        return {"status": "no dataset"}

    if len(df) < 10:
        return {"status": "not enough data to retrain"}

    X = df[["temperature", "time_of_day"]]
    y = df["will_buy"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    candidate = RandomForestClassifier(random_state=42)
    candidate.fit(X_train, y_train)

    current_acc = accuracy_score(y_test, current_model.predict(X_test))
    candidate_acc = accuracy_score(y_test, candidate.predict(X_test))

    with mlflow.start_run(run_name=f"candidate_v{current_version+1}"):

        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("features", "temperature,time_of_day")
        mlflow.log_metric("current_accuracy", current_acc)
        mlflow.log_metric("candidate_accuracy", candidate_acc)

        if candidate_acc > current_acc:
            new_version = current_version + 1
            model_path = f"model_v{new_version}.pkl"

            joblib.dump(candidate, model_path)

            mlflow.sklearn.log_model(
                candidate,
                artifact_path="model",
                registered_model_name="LemonadeDemandModel"
            )

            mlflow.set_tag("model_status", "promoted")

            return {
                "current_version": current_version,
                "current_accuracy": round(current_acc, 4),
                "candidate_accuracy": round(candidate_acc, 4),
                "new_model_saved": model_path
            }

        else:
            mlflow.set_tag("model_status", "rejected")

            return {
                "current_version": current_version,
                "current_accuracy": round(current_acc, 4),
                "candidate_accuracy": round(candidate_acc, 4),
                "message": "No improvement - keeping current model"
            }

# -------------------------------------------------
# Full Cycle Trigger
# -------------------------------------------------
@app.post("/cycle/run-full")
def run_full_cycle():
    return {
        "ground_truth": simulate_ground_truth(),
        "dataset": create_dataset(),
        "retrain": retrain()
    }

# -------------------------------------------------
# Root
# -------------------------------------------------
@app.get("/")
def root():
    try:
        _, version, path = load_latest_model()
        model_info = f"Latest: model_v{version}.pkl ({path})"
    except:
        model_info = "No model yet"

    return {
        "message": "Smart Lemonade Stand MLOps API",
        "current_model": model_info,
        "endpoints": {
            "POST /predict": "Online inference",
            "POST /cycle/simulate-ground-truth": "Step 1",
            "POST /cycle/create-dataset": "Step 2",
            "POST /cycle/retrain": "Step 3",
            "POST /cycle/run-full": "Run full pipeline"
        }
    }
