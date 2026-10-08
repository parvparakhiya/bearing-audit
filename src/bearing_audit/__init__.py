"""Bearing fault diagnosis on the CWRU dataset, with a leakage audit.

Public API: BearingMonitor, Diagnosis, Geometry, SKF_6205.
"""

from .api import BearingMonitor, Diagnosis, InputError
from .bearing import SKF_6205, Geometry

__all__ = ["SKF_6205", "BearingMonitor", "Diagnosis", "Geometry", "InputError"]
__version__ = "1.0.0"
