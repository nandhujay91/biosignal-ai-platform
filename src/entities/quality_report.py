from dataclasses import dataclass, field


@dataclass(slots=True)
class QualityReport:
    """
    Standard quality assessment result for any biosignal.
    """

    signal_name: str

    passed: bool

    score: float

    warnings: list[str] = field(default_factory=list)

    errors: list[str] = field(default_factory=list)
