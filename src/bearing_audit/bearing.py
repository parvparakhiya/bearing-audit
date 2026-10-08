"""Bearing geometry and the fault frequencies that follow from it.

n rolling elements, element diameter d, pitch diameter D, contact angle phi,
shaft speed f_r:

    BPFO = n/2 * f_r * (1 - d/D cos phi)          outer race
    BPFI = n/2 * f_r * (1 + d/D cos phi)          inner race
    BSF  = D/(2d) * f_r * (1 - (d/D cos phi)^2)   ball spin
    FTF  = 1/2 * f_r * (1 - d/D cos phi)          cage

Careful with ball faults: a damaged ball hits both races once per spin, so the
impact rate is 2 x BSF. CWRU's table lists 2 x BSF (4.7135 x f_r) as the
rolling-element frequency, while the textbook BSF is 2.3567 x f_r. Both are
exposed here under explicit names, so a ball peak is never read off by 2x.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Geometry:
    name: str
    n_elements: int
    element_diameter: float  # d (any unit)
    pitch_diameter: float  # D (same unit as d)
    contact_angle_deg: float = 0.0

    @property
    def _ratio(self) -> float:
        return self.element_diameter / self.pitch_diameter * math.cos(math.radians(self.contact_angle_deg))

    def multiples(self) -> dict[str, float]:
        """Fault frequencies as multiples of shaft speed (orders)."""
        r, n = self._ratio, self.n_elements
        bsf = self.pitch_diameter / (2 * self.element_diameter) * (1 - r**2)
        return {
            "BPFO": n / 2 * (1 - r),
            "BPFI": n / 2 * (1 + r),
            "BSF": bsf,
            "2xBSF": 2 * bsf,
            "FTF": 0.5 * (1 - r),
        }

    def frequencies(self, shaft_hz: float) -> dict[str, float]:
        """Fault frequencies in Hz at the given shaft speed."""
        return {k: v * shaft_hz for k, v in self.multiples().items()}


# CWRU drive-end bearing, dimensions in inches (from the CWRU website)
SKF_6205 = Geometry("SKF 6205-2RS JEM", n_elements=9, element_diameter=0.3126, pitch_diameter=1.537)

# CWRU's published values, only used to cross-check the formulas above
CWRU_PUBLISHED = {"BPFI": 5.4152, "BPFO": 3.5848, "FTF": 0.39828, "2xBSF": 4.7135}

# which line each fault type should produce in the envelope spectrum
SIGNATURE = {"IR": "BPFI", "OR": "BPFO", "B": "2xBSF"}


def shaft_hz(rpm: float) -> float:
    return rpm / 60.0
