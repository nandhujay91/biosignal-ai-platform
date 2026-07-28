from __future__ import annotations

from pathlib import Path

import yaml


class ConfigLoader:
    """
    Load application configuration.
    """

    def __init__(
        self,
        path: str = "configs/inference.yaml",
    ):

        self.path = Path(path)

    def load(self):

        with open(
            self.path,
            "r",
        ) as file:

            return yaml.safe_load(file)
