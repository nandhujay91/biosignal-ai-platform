from pathlib import Path

PROJECT_ROOT = Path.cwd()

CONFIG_DIR = PROJECT_ROOT / "configs"

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PARSED_DATA_DIR = DATA_DIR / "parsed"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

WINDOWS_DIR = DATA_DIR / "windows"

EMBEDDINGS_DIR = DATA_DIR / "embeddings"

MODELS_DIR = PROJECT_ROOT / "models"

REPORTS_DIR = PROJECT_ROOT / "reports"

LOGS_DIR = PROJECT_ROOT / "logs"
