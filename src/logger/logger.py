import sys

from loguru import logger

from src.config.artifact_config import LOG_ROOT

# Create log directory
LOG_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

# Remove default logger
logger.remove()

# Console logger
logger.add(
    sys.stdout,
    level="INFO",
    colorize=True,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level:<8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:"
        "<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    ),
)

# File logger
logger.add(
    LOG_ROOT / "biosignal_pipeline.log",
    level="DEBUG",
    rotation="10 MB",
    retention="30 days",
    compression="zip",
    enqueue=True,
    backtrace=True,
    diagnose=True,
    encoding="utf-8",
    format=(
        "{time:YYYY-MM-DD HH:mm:ss} | "
        "{level:<8} | "
        "{name}:{function}:{line} | "
        "{message}"
    ),
)
