from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """
    Request body for biosignal prediction.
    """

    features: list[float] = Field(
        ...,
        description="131-dimensional fusion feature vector",
    )



class PredictionResponseAPI(BaseModel):
    """
    API response schema.
    """

    prediction: str

    confidence: float

    risk_level: str

    recommended_action: str

    model_version: str