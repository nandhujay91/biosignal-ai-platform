"""
Project Constants
"""

# Project
PROJECT_NAME = "Embedding Model Biosignal"

# Random Seed
RANDOM_STATE = 42

# Window Configuration
WINDOW_DURATION = 5  # seconds
WINDOW_OVERLAP = 2.5  # seconds

# Classification Labels
NORMAL = "Normal"
ALERT = "Alert"
CRITICAL = "Critical"

LABELS = [NORMAL, ALERT, CRITICAL]

# Supported Signals
SUPPORTED_SIGNALS = ["Aux", "IMU", "Ephy", "Oxym"]

# File Extension
BIN_EXTENSION = ".bin"
