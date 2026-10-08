"""Run the physics diagnoser on your own measurements.

    from bearing_audit import BearingMonitor, Geometry

    monitor = BearingMonitor(geometry=Geometry("my bearing", 9, 7.94, 39.04))
    monitor.fit_baseline([(x_healthy_a, fs, 1797), (x_healthy_b, fs, 1772)])
    result = monitor.diagnose(x, fs, rpm=1750)
    result.state     # "Healthy", "IR", "OR", "B" or "Undiagnosed"
    result.warnings  # e.g. rpm outside the baseline's range
    monitor.save("baseline.json")
    monitor = BearingMonitor.load("baseline.json")

There's no trained model: the baseline is the highest value each indicator
reached on the healthy measurements, stored as plain JSON.

Same diagnoser as the one evaluated in docs/industrial-preregistration.md. It did not
pass the plant-level gate (19 of 28 named diagnoses right, 3 false alarms on CWRU), so
treat it as a screening tool, not a certified diagnostic.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import bearing, config, dsp, features
from .industrial import PhysicsDiagnoser, recording_decision

ANALYSIS_FS = 12_000  # everything is resampled to this rate first
MIN_BASELINE_WINDOWS = 10
FORMAT_VERSION = 1


class InputError(ValueError):
    """Raised for a measurement or baseline the diagnoser can't judge."""


@dataclass(frozen=True)
class Diagnosis:
    state: str  # Healthy, IR, OR, B or Undiagnosed
    alarm_share: float  # fraction of 1 s windows in alarm
    window_states: dict[str, int]  # window count per state
    evidence: dict[str, float]  # median(indicator) - threshold; > 0 means above the healthy max
    warnings: list[str] = field(default_factory=list)

    @property
    def alarm(self) -> bool:
        return self.state != "Healthy"


def _analysis_settings() -> dict:
    """Stored with every baseline, so one made with other settings is rejected on load."""
    return {
        "fs": ANALYSIS_FS,
        "window_s": config.WINDOW_S,
        "bandwidth_hz": config.BANDWIDTH_HZ,
        "envelope_band_hz": list(config.ENVELOPE_BAND_HZ),
    }


def _check_signal(x, fs, rpm) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise InputError(f"expected a 1-D signal, got shape {x.shape}")
    if not np.isfinite(x).all():
        raise InputError("signal contains NaN or infinite values")
    if not isinstance(fs, (int, np.integer)) or fs < ANALYSIS_FS:
        raise InputError(
            f"sampling rate must be an integer >= {ANALYSIS_FS} Hz (analysis bandwidth is 5 kHz); got {fs}"
        )
    if x.size < fs * config.WINDOW_S:
        raise InputError(f"need at least {config.WINDOW_S:g} s of signal; got {x.size / fs:.2f} s")
    if not (isinstance(rpm, (int, float, np.number)) and math.isfinite(rpm) and 60 <= rpm <= 60_000):
        raise InputError(f"rpm must be a finite number between 60 and 60000; got {rpm}")
    if np.ptp(x) == 0:
        raise InputError("signal is constant: sensor disconnected or saturated?")
    return dsp.resample(x, int(fs), ANALYSIS_FS)


class BearingMonitor:
    """Healthy baseline plus three-state physics diagnosis for one bearing type."""

    def __init__(self, geometry: bearing.Geometry = bearing.SKF_6205):
        self.geometry = geometry
        self._diagnoser: PhysicsDiagnoser | None = None
        self._rpm_range: tuple[float, float] | None = None
        self._n_baseline_windows = 0

    def _features(self, x: np.ndarray, rpm: float) -> pd.DataFrame:
        rows = [
            features.window_features(w, ANALYSIS_FS, rpm / 60.0, geometry=self.geometry)
            for w in features.windows(x, ANALYSIS_FS)
        ]
        return pd.DataFrame(rows)[features.DIAGNOSER_FEATURES]

    def fit_baseline(self, healthy: list[tuple[np.ndarray, int, float]]) -> BearingMonitor:
        """Fit the thresholds on healthy measurements of this machine.

        ``healthy`` is a list of (signal, sampling_rate_hz, rpm). Cover every
        operating condition you want to monitor: the thresholds don't extrapolate
        (that's what caused the Normal_3 false alarms in the industrial evaluation).
        """
        if not healthy:
            raise InputError("the baseline needs at least one healthy measurement")
        frames, rpms = [], []
        for x, fs, rpm in healthy:
            frames.append(self._features(_check_signal(x, fs, rpm), rpm))
            rpms.append(float(rpm))
        X = pd.concat(frames, ignore_index=True)
        if len(X) < MIN_BASELINE_WINDOWS:
            raise InputError(f"the baseline needs at least {MIN_BASELINE_WINDOWS} one-second windows; got {len(X)}")
        self._diagnoser = PhysicsDiagnoser().fit(X, ["Normal"] * len(X))
        self._rpm_range = (min(rpms), max(rpms))
        self._n_baseline_windows = len(X)
        return self

    def diagnose(self, x: np.ndarray, fs: int, rpm: float) -> Diagnosis:
        if self._diagnoser is None or self._rpm_range is None:
            raise InputError("fit_baseline() or load() first")
        X = self._features(_check_signal(x, fs, rpm), rpm)
        states = pd.Series(self._diagnoser.predict(X))
        s, t = self._diagnoser._scores(X), self._diagnoser.thresholds_
        warnings = []
        lo, hi = self._rpm_range
        if not lo <= rpm <= hi:
            warnings.append(
                f"rpm {rpm:g} is outside the baseline range {lo:g}-{hi:g}: thresholds may not hold "
                "(on CWRU this is exactly where the diagnoser raised its false alarms)"
            )
        return Diagnosis(
            state=recording_decision(states),
            alarm_share=float((states != "Healthy").mean()),
            window_states=dict(Counter(states)),
            evidence={k: round(float(s[k].median() - t[k]), 4) for k in s.columns},
            warnings=warnings,
        )

    # persistence: plain JSON, no pickle
    def to_dict(self) -> dict:
        if self._diagnoser is None or self._rpm_range is None:
            raise InputError("nothing to save: fit_baseline() first")
        return {
            "format_version": FORMAT_VERSION,
            "geometry": asdict(self.geometry),
            "thresholds": {k: float(v) for k, v in self._diagnoser.thresholds_.items()},
            "baseline_rpm_range": list(self._rpm_range),
            "baseline_windows": self._n_baseline_windows,
            "analysis": _analysis_settings(),
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, d: dict) -> BearingMonitor:
        if d.get("format_version") != FORMAT_VERSION:
            raise InputError(f"unsupported baseline format {d.get('format_version')!r}")
        expected = _analysis_settings()
        if d.get("analysis") != expected:
            raise InputError(f"baseline was made with different analysis settings: {d.get('analysis')} != {expected}")
        m = cls(bearing.Geometry(**d["geometry"]))
        m._diagnoser = PhysicsDiagnoser()
        m._diagnoser.thresholds_ = pd.Series(d["thresholds"], dtype=float)
        m._rpm_range = (float(d["baseline_rpm_range"][0]), float(d["baseline_rpm_range"][1]))
        m._n_baseline_windows = int(d["baseline_windows"])
        return m

    @classmethod
    def load(cls, path: str | Path) -> BearingMonitor:
        return cls.from_dict(json.loads(Path(path).read_text()))
