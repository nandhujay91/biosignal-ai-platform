from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import Response

from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from src.api.routes import router


app = FastAPI(
    title="Biosignal Embedding Model API",
    description=(
        "TS2Vec embedding model + DNN classifier "
        "for biosignal classification"
    ),
    version="v1",
)


# Register API routes

app.include_router(
    router
)


@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "service": "biosignal-classifier-api",
    }



@app.get("/model-info")
def model_info():

    return {
        "model_type": "TS2Vec Embedding + DNN Classifier",
        "input_features": 131,
        "embedding_dimension": 128,
        "signal_features": 3,
        "classes": [
            "Normal",
            "Alert",
            "Critical",
            "Borderline",
        ],
        "version": "v1",
    }



# Prometheus metrics endpoint

@app.get("/metrics")
def metrics():

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )