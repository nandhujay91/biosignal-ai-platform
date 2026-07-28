from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class DatasetMetadata:
    """
    Metadata describing a processed biosignal dataset.
    """

    signal_name: str

    sampling_rate: int
    channels: int

    window_duration_seconds: int
    window_size: int
    step_size: int
    overlap: float

    normalization: str
    filter_type: str

    dtype: str

    num_windows: int | None = None
    created_at: str | None = None

    dataset_version: str = "v1"

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the metadata object to a dictionary.
        """
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "DatasetMetadata":
        """
        Create a DatasetMetadata object from a dictionary.
        """
        return cls(**data)
