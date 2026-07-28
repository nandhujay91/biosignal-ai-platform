from pathlib import Path

# Root artifact directory
ARTIFACT_ROOT = Path("artifacts")

# Dataset
DATASET_VERSION = "v1"

DATASET_ROOT = (
    ARTIFACT_ROOT
    / "datasets"
    / DATASET_VERSION
)

# Models
MODEL_ROOT = (
    ARTIFACT_ROOT
    / "models"
)

# Embeddings
EMBEDDING_ROOT = (
    ARTIFACT_ROOT
    / "embeddings"
)

# Logs
LOG_ROOT = (
    ARTIFACT_ROOT
    / "logs"
)