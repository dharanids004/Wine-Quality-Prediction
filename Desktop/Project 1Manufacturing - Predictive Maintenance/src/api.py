from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

from features import FeatureConfig, build_features, clean_and_interpolate

app = Flask(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

MODEL = joblib.load(ARTIFACTS_DIR / "model.joblib")
FEATURE_COLS: List[str] = joblib.load(ARTIFACTS_DIR / "feature_cols.joblib")
FEATURE_CFG: FeatureConfig = joblib.load(ARTIFACTS_DIR / "feature_cfg.joblib")


def prepare_single_payload(payload: Dict) -> pd.DataFrame:
    # Expected payload format:
    # {
    #   "history": [ {row1}, {row2}, ... ],
    #   "current": {row}
    # }
    history = payload.get("history", [])
    current = payload.get("current")
    if current is None:
        raise ValueError("'current' record is required")

    rows = history + [current]
    df = pd.DataFrame(rows)

    df = clean_and_interpolate(df, FEATURE_CFG.time_col, FEATURE_CFG.id_col)
    df, _ = build_features(df, FEATURE_CFG)

    # Use the last row as the prediction row
    df = df.sort_values([FEATURE_CFG.id_col, FEATURE_CFG.time_col])
    last = df.iloc[-1:]
    return last


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    row = prepare_single_payload(payload)

    X = row[FEATURE_COLS]
    proba = MODEL.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)

    return jsonify({"failure_probability": float(proba[0]), "prediction": int(pred[0])})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
