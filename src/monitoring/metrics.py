from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# =====================================================
# Prediction metrics
# =====================================================

PREDICTION_COUNT = Counter(
    "biosignal_predictions_total",
    "Total number of biosignal predictions performed",
)


PREDICTION_LABELS = Counter(
    "biosignal_prediction_labels_total",
    "Total predictions grouped by predicted label",
    ["prediction"],
)


# =====================================================
# Model confidence metrics
# =====================================================

CONFIDENCE_SCORE = Histogram(
    "biosignal_confidence_score",
    "Distribution of model prediction confidence scores",
    buckets=[
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ],
)


# =====================================================
# Inference performance metrics
# =====================================================

INFERENCE_TIME = Histogram(
    "biosignal_inference_seconds",
    "Time taken for model inference in seconds",
    buckets=[
        0.01,
        0.05,
        0.1,
        0.2,
        0.5,
        1.0,
    ],
)


# =====================================================
# API health metrics
# =====================================================

ACTIVE_REQUESTS = Gauge(
    "biosignal_active_requests",
    "Number of active prediction requests",
)


MODEL_VERSION = Gauge(
    "biosignal_model_version_info",
    "Current deployed model version",
    ["version"],
)
