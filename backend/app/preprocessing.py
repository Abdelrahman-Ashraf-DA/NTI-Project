"""
Reproduces the original training-time feature representation exactly.

Column order and identity are taken verbatim from model_columns.pkl (which is
byte-identical to scaler.feature_names_in_). Baseline categories (the ones
pandas.get_dummies(drop_first=True) dropped) are never encoded — they are
represented implicitly as an all-zero row for that dummy group:

  Operating Carrier baseline -> Alaska Airlines
  ORIGIN / DEST baseline     -> ATL

Both were reconstructed from the artifacts themselves (alphabetical position
within the observed dummy columns), not invented.
"""
import numpy as np
import pandas as pd

from .schemas import FlightInput
from .prediction import MODEL_COLUMNS

CARRIER_PREFIX = "Operating Carrier_"
ORIGIN_PREFIX = "ORIGIN_"
DEST_PREFIX = "DEST_"

NUMERIC_INDEX = {name: i for i, name in enumerate(MODEL_COLUMNS)}


def build_feature_vector(flight: FlightInput) -> np.ndarray:
    """Return a (1, 56) row ordered exactly as MODEL_COLUMNS expects."""
    row = np.zeros((1, len(MODEL_COLUMNS)), dtype=float)

    dep_delay_15 = 1.0 if flight.dep_delay >= 15 else 0.0

    row[0, NUMERIC_INDEX["DEP_DELAY"]] = flight.dep_delay
    row[0, NUMERIC_INDEX["Departure Delay ≥ 15 Minutes"]] = dep_delay_15
    row[0, NUMERIC_INDEX["DISTANCE"]] = flight.distance
    row[0, NUMERIC_INDEX["Departure Hour"]] = flight.departure_hour

    carrier_col = CARRIER_PREFIX + flight.operating_carrier
    if carrier_col in NUMERIC_INDEX:  # absent => baseline (Alaska Airlines), stays all-zero
        row[0, NUMERIC_INDEX[carrier_col]] = 1.0

    origin_col = ORIGIN_PREFIX + flight.origin
    if origin_col in NUMERIC_INDEX:  # absent => baseline (ATL), stays all-zero
        row[0, NUMERIC_INDEX[origin_col]] = 1.0

    dest_col = DEST_PREFIX + flight.dest
    if dest_col in NUMERIC_INDEX:  # absent => baseline (ATL), stays all-zero
        row[0, NUMERIC_INDEX[dest_col]] = 1.0

    return pd.DataFrame(row, columns=MODEL_COLUMNS), dep_delay_15
