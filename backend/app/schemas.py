"""Pydantic request/response models for the FlightOps AI prediction API."""
from pydantic import BaseModel, Field, field_validator

CARRIERS = [
    "Alaska Airlines", "Allegiant Air", "American Airlines", "Delta Air Lines",
    "Envoy Air", "Frontier Airlines", "JetBlue Airways", "PSA Airlines",
    "Republic Airways", "SkyWest Airlines", "Southwest Airlines",
    "Spirit Airlines", "United Airlines",
]

# 19 individually-modeled airports + the baseline (ATL) + the "Other" bucket.
AIRPORTS = [
    "ATL", "BOS", "CLT", "DCA", "DEN", "DFW", "DTW", "EWR", "FLL", "IAH",
    "LAS", "LAX", "LGA", "MCO", "MIA", "ORD", "PHX", "SEA", "SFO", "SLC",
    "Other",
]


class FlightInput(BaseModel):
    operating_carrier: str = Field(..., description="Operating carrier")
    origin: str = Field(..., description="Origin airport (top-20 code or 'Other')")
    dest: str = Field(..., description="Destination airport (top-20 code or 'Other')")
    dep_delay: float = Field(..., ge=-60, le=600, description="Departure delay, minutes")
    distance: float = Field(..., gt=0, le=6000, description="Scheduled distance, miles")
    departure_hour: int = Field(..., ge=0, le=23, description="Scheduled departure hour, 0-23")

    @field_validator("operating_carrier")
    @classmethod
    def check_carrier(cls, v):
        if v not in CARRIERS:
            raise ValueError(f"operating_carrier must be one of {CARRIERS}")
        return v

    @field_validator("origin", "dest")
    @classmethod
    def check_airport(cls, v):
        if v not in AIRPORTS:
            raise ValueError(f"airport must be one of {AIRPORTS}")
        return v


class Driver(BaseModel):
    feature: str
    label: str
    direction: str  # "increases" | "decreases"
    magnitude: float
    strength: str  # "Strong" | "Moderate" | "Slight"


class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    delay_probability: float
    on_time_probability: float
    risk_level: str
    departure_delay_15_flag: int
    drivers: list[Driver]
    driver_type: str  # "individual" | "global"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    scaler_loaded: bool
    n_features: int
