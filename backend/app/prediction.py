"""Loads the trusted local artifacts and runs real inference. No retraining,
no substitute model, no invented preprocessing — this module only replays the
saved LogisticRegression + StandardScaler exactly as they were fit."""
from pathlib import Path

import joblib
import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

try:
    MODEL = joblib.load(MODELS_DIR / "logistic_model.pkl")
    SCALER = joblib.load(MODELS_DIR / "scaler.pkl")
    MODEL_COLUMNS = joblib.load(MODELS_DIR / "model_columns.pkl")
    ARTIFACTS_OK = True
except FileNotFoundError:
    MODEL, SCALER, MODEL_COLUMNS = None, None, []
    ARTIFACTS_OK = False


def risk_level(probability: float) -> str:
    if probability < 0.34:
        return "Low"
    if probability < 0.67:
        return "Moderate"
    return "High"


def predict(feature_row: np.ndarray) -> dict:
    """feature_row: (1, 56) raw (unscaled) array, in MODEL_COLUMNS order."""
    if not ARTIFACTS_OK:
        raise RuntimeError("Model artifacts are not loaded.")

    scaled = SCALER.transform(feature_row)
    pred = int(MODEL.predict(scaled)[0])

    # classes_[1] is confirmed to be the positive ("delayed") class.
    delay_idx = list(MODEL.classes_).index(1)
    proba = MODEL.predict_proba(scaled)[0]
    delay_prob = float(proba[delay_idx])
    on_time_prob = 1.0 - delay_prob

    return {
        "prediction": pred,
        "prediction_label": "Delayed" if pred == 1 else "On Time",
        "delay_probability": round(delay_prob, 4),
        "on_time_probability": round(on_time_prob, 4),
        "risk_level": risk_level(delay_prob),
        "scaled_row": scaled[0],
    }
