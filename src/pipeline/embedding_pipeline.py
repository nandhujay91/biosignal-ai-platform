from __future__ import annotations

from pathlib import Path

from src.embeddings.generate_embeddings import EmbeddingGenerator
from src.logger import logger


class EmbeddingPipeline:
    """
    Pipeline for generating TS2Vec embeddings.
    """

    def __init__(
        self,
        dataset_path: str = "artifacts/datasets/v1/Ephy/windows.npy",
        output_path: str = "artifacts/embeddings/embeddings.npy",
    ) -> None:

        self.dataset_path = Path(dataset_path)

        self.output_path = Path(output_path)

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def run(self):

        logger.info(f"Generating embeddings from: {self.dataset_path}")

        generator = EmbeddingGenerator()

        embeddings = generator.generate(str(self.dataset_path))

        # Save generated embeddings

        import numpy as np

        np.save(
            self.output_path,
            embeddings,
        )

        logger.info(f"Embeddings saved: {embeddings.shape}")

        return embeddings


if __name__ == "__main__":

    pipeline = EmbeddingPipeline()

    pipeline.run()
