import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .explainability import individual_drivers
from .prediction import ARTIFACTS_OK, MODEL, MODEL_COLUMNS, SCALER, predict
from .preprocessing import build_feature_vector
from .schemas import AIRPORTS, CARRIERS, FlightInput, HealthResponse, PredictionResponse

app = FastAPI(
    title="FlightOps AI API",
    description="Real-time inference API for the Airline Flight Delay Prediction model.",
    version="1.0.0",
)

allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if ARTIFACTS_OK else "degraded",
        model_loaded=MODEL is not None,
        scaler_loaded=SCALER is not None,
        n_features=len(MODEL_COLUMNS),
    )


@app.get("/options")
def options():
    """Valid categorical values, derived from the actual model artifacts."""
    return {"carriers": CARRIERS, "airports": AIRPORTS}


@app.post("/predict", response_model=PredictionResponse)
def predict_delay(flight: FlightInput):
    if not ARTIFACTS_OK:
        raise HTTPException(status_code=503, detail="Model artifacts unavailable.")

    feature_row, dep_delay_15 = build_feature_vector(flight)

    try:
        result = predict(feature_row)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    drivers = individual_drivers(result["scaled_row"])

    return PredictionResponse(
        prediction=result["prediction"],
        prediction_label=result["prediction_label"],
        delay_probability=result["delay_probability"],
        on_time_probability=result["on_time_probability"],
        risk_level=result["risk_level"],
        departure_delay_15_flag=int(dep_delay_15),
        drivers=drivers,
        driver_type="individual",
    )
