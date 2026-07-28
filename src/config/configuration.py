from pathlib import Path

import yaml


class ConfigurationManager:
    """
    Loads YAML configuration files.
    """

    def __init__(self, config_path="configs/config.yaml"):
        self.config_path = Path(config_path)

    def load(self):
        with open(self.config_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
