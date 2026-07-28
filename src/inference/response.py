from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PredictionResponse:
    """
    Production prediction response schema.
    """

    prediction: str

    confidence: float

    risk_level: str

    recommended_action: str

    model_version: str