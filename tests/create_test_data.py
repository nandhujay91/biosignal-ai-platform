from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

# Synthetic raw signal data
DATA_DIR = Path("data/test")

# Synthetic processed dataset artifacts
ARTIFACT_DIR = Path("artifacts/datasets/v1")


def create_signal_file(
    name: str,
    channels: int,
    samples: int,
):
    """
    Create synthetic int16 biosignal binary file.
    """

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    data = np.random.randint(
        low=-1000,
        high=1000,
        size=(samples, channels),
        dtype=np.int16,
    )

    file_path = DATA_DIR / f"{name}.bin"

    data.tofile(file_path)

    print(f"Created BIN: {file_path}")


def create_dataset_artifact(
    name: str,
    shape: tuple,
    sampling_rate: int,
):
    """
    Create synthetic processed dataset artifact
    required by DatasetLoader.
    """

    signal_dir = ARTIFACT_DIR / name

    signal_dir.mkdir(parents=True, exist_ok=True)

    # Create synthetic windows

    dataset = np.random.randn(*shape).astype(np.float32)

    np.save(signal_dir / "windows.npy", dataset)

    metadata = {
        "signal_name": name,
        "sampling_rate": sampling_rate,
        "channels": shape[-1],
        "window_duration_seconds": 5,
        "window_size": shape[1],
        "step_size": shape[1] // 2,
        "overlap": 0.5,
        "normalization": "z_score",
        "filter_type": "none",
        "dtype": "float32",
        "num_windows": shape[0],
        "dataset_version": "v1",
    }

    with open(signal_dir / "metadata.yaml", "w", encoding="utf-8") as file:

        yaml.safe_dump(metadata, file)

    print(f"Created Artifact: {signal_dir}")


def main():

    print("Creating synthetic biosignal test data...")

    # ==========================
    # Raw BIN files
    # ==========================

    create_signal_file(
        name="Aux",
        channels=3,
        samples=16000,
    )

    create_signal_file(
        name="Ephy",
        channels=8,
        samples=25600,
    )

    create_signal_file(
        name="IMU",
        channels=9,
        samples=32000,
    )

    create_signal_file(
        name="Oxym",
        channels=2,
        samples=16000,
    )

    print("\nCreating synthetic dataset artifacts...")

    # ==========================
    # DatasetLoader artifacts
    # ==========================

    create_dataset_artifact(
        name="Aux",
        shape=(1132, 160, 3),
        sampling_rate=32,
    )

    create_dataset_artifact(
        name="Ephy",
        shape=(1132, 1280, 8),
        sampling_rate=256,
    )

    create_dataset_artifact(
        name="IMU",
        shape=(376, 320, 9),
        sampling_rate=64,
    )

    create_dataset_artifact(
        name="Oxym",
        shape=(1132, 640, 2),
        sampling_rate=128,
    )

    print("\nSynthetic test data creation completed.")


if __name__ == "__main__":

    main()
