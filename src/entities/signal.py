from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Signal:
    """
    Domain entity representing one biosignal.
    """

    name: str
    data: np.ndarray
    dtype: np.dtype
    channels: int
    sampling_rate: int
