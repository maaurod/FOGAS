"""Current discrete and continuous SBEED solver exports."""

from .continuous_sbeed import ContinuousSBEED
from .discrete_sbeed import DiscreteSBEED

SBEED = DiscreteSBEED

__all__ = [
    "ContinuousSBEED",
    "DiscreteSBEED",
    "SBEED",
]
