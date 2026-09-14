"""
Individual prediction drivers.

For a linear model on scaled features, the logit contribution of feature i is
`scaled_value_i * coef_i`. We compute this per-feature for the specific
request, then aggregate one-hot groups (carrier / origin / destination) into
a single human-readable driver so a category isn't reported as 56 line items.

This produces a genuine per-prediction explanation (not a restatement of
global importance) because it uses this request's own scaled feature values.
"""
import numpy as np

from .prediction import MODEL, MODEL_COLUMNS

FRIENDLY_LABELS = {
    "DEP_DELAY": "Departure Delay",
    "Departure Delay ≥ 15 Minutes": "Departure Delay 15+ Min Flag",
    "DISTANCE": "Flight Distance",
    "Departure Hour": "Departure Hour",
}


def _group_of(col: str) -> str:
    if col.startswith("Operating Carrier_"):
        return "Operating Carrier"
    if col.startswith("ORIGIN_"):
        return "Origin Airport"
    if col.startswith("DEST_"):
        return "Destination Airport"
    return FRIENDLY_LABELS.get(col, col)


def _strength(abs_contribution: float) -> str:
    if abs_contribution >= 0.5:
        return "Strong"
    if abs_contribution >= 0.15:
        return "Moderate"
    return "Slight"


def individual_drivers(scaled_row: np.ndarray, top_n: int = 5) -> list[dict]:
    coefs = MODEL.coef_[0]
    contributions = scaled_row * coefs  # per-feature logit contribution

    grouped: dict[str, float] = {}
    for col, contrib in zip(MODEL_COLUMNS, contributions):
        group = _group_of(col)
        grouped[group] = grouped.get(group, 0.0) + float(contrib)

    ranked = sorted(grouped.items(), key=lambda kv: abs(kv[1]), reverse=True)

    drivers = []
    for label, value in ranked[:top_n]:
        if abs(value) < 1e-4:
            continue
        drivers.append(
            {
                "feature": label,
                "label": label,
                "direction": "increases" if value > 0 else "decreases",
                "magnitude": round(float(abs(value)), 4),
                "strength": _strength(abs(value)),
            }
        )
    return drivers


def global_drivers(top_n: int = 5) -> list[dict]:
    """Fallback: global |coefficient| ranking, explicitly labeled as global."""
    coefs = MODEL.coef_[0]
    grouped: dict[str, float] = {}
    for col, coef in zip(MODEL_COLUMNS, coefs):
        group = _group_of(col)
        grouped[group] = grouped.get(group, 0.0) + abs(float(coef))

    ranked = sorted(grouped.items(), key=lambda kv: kv[1], reverse=True)
    return [
        {
            "feature": label,
            "label": label,
            "direction": "increases",
            "magnitude": round(float(value), 4),
            "strength": _strength(value),
        }
        for label, value in ranked[:top_n]
    ]
