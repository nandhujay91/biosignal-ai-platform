from __future__ import annotations

from pathlib import Path
import numpy as np


DATA_DIR = Path("data/test")


def create_signal_file(
    name: str,
    channels: int,
    samples: int,
):
    """
    Create synthetic int16 biosignal binary file.
    """

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    data = np.random.randint(
        low=-1000,
        high=1000,
        size=(samples, channels),
        dtype=np.int16,
    )


    file_path = DATA_DIR / f"{name}.bin"


    data.tofile(
        file_path
    )


    print(
        f"Created: {file_path}"
    )



def main():

    print(
        "Creating synthetic biosignal test data..."
    )


    # Match project signal configuration

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


    print(
        "Synthetic test data creation completed."
    )



if __name__ == "__main__":

    main()