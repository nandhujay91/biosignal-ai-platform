from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    PredictionRequest,
    PredictionResponseAPI,
)
from src.inference.predictor import BiosignalPredictor

router = APIRouter()


# Load model once when API starts
predictor = BiosignalPredictor()


@router.post(
    "/predict",
    response_model=PredictionResponseAPI,
)
def predict(
    request: PredictionRequest,
):

    try:

        result = predictor.predict(request.features)

        return PredictionResponseAPI(
            prediction=result.prediction,
            confidence=result.confidence,
            risk_level=result.risk_level,
            recommended_action=result.recommended_action,
            model_version=result.model_version,
        )

    except Exception as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )
