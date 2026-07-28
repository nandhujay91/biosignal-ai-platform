from __future__ import annotations

from pathlib import Path

import yaml


class InferenceConfig:
    """
    Configuration loader for inference service.
    """


    def __init__(
        self,
        path: str = "configs/inference.yaml",
    ) -> None:

        self.path = Path(path)

        self.config = self._load()



    def _load(self):

        with open(
            self.path,
            "r",
        ) as file:

            return yaml.safe_load(file)



    @property
    def model_directory(self):

        return self.config["model"]["directory"]



    @property
    def input_dim(self):

        return self.config["model"]["input_dim"]



    @property
    def num_classes(self):

        return self.config["model"]["num_classes"]



    @property
    def confidence_thresholds(self):

        return self.config["confidence"]