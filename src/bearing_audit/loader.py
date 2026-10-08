"""Load a CWRU .mat file as a signal you can trust.

Three things go wrong silently with these files:
1. The sampling rate isn't stored. The normal baseline is 48 kHz, the fault
   files 12 kHz. Everything is resampled to 12 kHz with an anti-alias filter.
2. Some files hold more than one record. Normal_2.mat has record 99 and an
   exact copy of record 98, so channels are picked by record number.
3. Some files have no rpm. The catalogue's nominal speed is used, and
   Signal.rpm_source says so.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from . import catalog, dsp

TARGET_FS = 12_000


@dataclass(frozen=True)
class Signal:
    recording: catalog.Recording
    x: np.ndarray  # drive-end acceleration at fs
    fs: int
    rpm: float
    rpm_source: str  # "file" or "catalogue"

    @property
    def shaft_hz(self) -> float:
        return self.rpm / 60.0

    @property
    def duration_s(self) -> float:
        return self.x.size / self.fs


def _key(mat: dict, record: int, suffix: str) -> str | None:
    """Variable name for this record, e.g. 'X097_DE_time' or 'X100RPM'."""
    for k in mat:
        if k.startswith("X") and k.endswith(suffix):
            digits = k[1 : -len(suffix)].rstrip("_")
            if digits.isdigit() and int(digits) == record:
                return k
    return None


def read_raw(path: Path, rec: catalog.Recording) -> tuple[np.ndarray, float | None]:
    """Drive-end signal and stored rpm (None if missing), at the file's native rate."""
    mat = loadmat(path, simplify_cells=True)
    de = _key(mat, rec.record, "_DE_time")
    if de is None:
        found = sorted(k for k in mat if not k.startswith("__"))
        raise ValueError(f"{path.name}: no drive-end channel for record {rec.record}; found {found}")
    rpm_key = _key(mat, rec.record, "RPM")
    rpm = float(np.asarray(mat[rpm_key]).squeeze()) if rpm_key else None
    return np.asarray(mat[de], dtype=float).ravel(), rpm


def load(name: str, data_dir: Path | str = "data/raw", fs: int = TARGET_FS) -> Signal:
    rec = catalog.get(name)
    path = Path(data_dir) / rec.filename
    x, rpm = read_raw(path, rec)
    x = dsp.resample(x, rec.native_fs, fs)
    if rpm is None:
        return Signal(rec, x, fs, catalog.NOMINAL_RPM[rec.load_hp], "catalogue")
    return Signal(rec, x, fs, rpm, "file")


def sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
