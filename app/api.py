from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_PATH = BASE_DIR / "artifacts" / "best_model.pkl"

app = FastAPI(title="HR Attrition Risk API", version="1.0.0")

_bundle = None


def load_bundle():
    global _bundle
    if _bundle is None:
        if not ARTIFACT_PATH.exists():
            raise RuntimeError("Model artifact not found. Run training first.")
        _bundle = joblib.load(ARTIFACT_PATH)
    return _bundle


class PredictRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(..., description="List of employee feature dictionaries")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    bundle = load_bundle()
    return {
        "target_col": bundle.get("target_col"),
        "threshold": bundle.get("threshold", 0.5),
        "feature_count": len(bundle.get("feature_columns", [])),
    }


@app.post("/predict")
def predict(req: PredictRequest):
    bundle = load_bundle()
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    threshold = bundle.get("threshold", 0.5)

    df = pd.DataFrame(req.records)
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required features: {missing}")

    df = df[feature_columns]
    proba = model.predict_proba(df)
    if proba.shape[1] == 2:
        attrition_prob = proba[:, 1]
    else:
        attrition_prob = proba.max(axis=1)

    preds = (attrition_prob >= threshold).astype(int)
    out = []
    for i in range(len(df)):
        out.append({
            "attrition_probability": float(attrition_prob[i]),
            "prediction": int(preds[i]),
            "risk_label": "high" if attrition_prob[i] >= 0.7 else "medium" if attrition_prob[i] >= 0.4 else "low",
        })
    return {"predictions": out}
