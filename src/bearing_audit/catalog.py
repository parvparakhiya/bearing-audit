"""What each CWRU recording is.

File names encode fault type, size, outer-race position and load (IR007_0 is
inner race, 0.007", 0 HP; OR021@6_1 is outer race, 0.021", fault at the
6 o'clock position, 1 HP). The record number is CWRU's own ID for a recording,
and the variables inside its .mat file are named after it: IR007_0 is record
105 and holds X105_DE_time, X105_FE_time, X105_BA_time and X105RPM. Channels
are always picked by that number, because Normal_2.mat (record 99) also holds
a full copy of record 98.

Scope, fixed before training anything:
- core set: Normal, IR, B and OR@6 at 0.007/0.014/0.021" and 0-3 HP.
  40 recordings, 10 physical bearings. Same subset most CWRU papers use.
- physics only: OR@3 and OR@12 (only two sizes exist there). Used to check
  fault frequencies, never for training.
- excluded: every 0.028" file. Different bearing (NTN, not the SKF 6205), no
  rpm stored, and the record numbers inside (048-059) don't match the
  catalogue (3001-3008).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# record numbers from the CWRU catalogue (12k drive-end data + normal baseline)
RECORD_NUMBERS: dict[str, int] = {
    "Normal_0": 97,
    "Normal_1": 98,
    "Normal_2": 99,
    "Normal_3": 100,
    "IR007_0": 105,
    "IR007_1": 106,
    "IR007_2": 107,
    "IR007_3": 108,
    "IR014_0": 169,
    "IR014_1": 170,
    "IR014_2": 171,
    "IR014_3": 172,
    "IR021_0": 209,
    "IR021_1": 210,
    "IR021_2": 211,
    "IR021_3": 212,
    "B007_0": 118,
    "B007_1": 119,
    "B007_2": 120,
    "B007_3": 121,
    "B014_0": 185,
    "B014_1": 186,
    "B014_2": 187,
    "B014_3": 188,
    "B021_0": 222,
    "B021_1": 223,
    "B021_2": 224,
    "B021_3": 225,
    "OR007@6_0": 130,
    "OR007@6_1": 131,
    "OR007@6_2": 132,
    "OR007@6_3": 133,
    "OR014@6_0": 197,
    "OR014@6_1": 198,
    "OR014@6_2": 199,
    "OR014@6_3": 200,
    "OR021@6_0": 234,
    "OR021@6_1": 235,
    "OR021@6_2": 236,
    "OR021@6_3": 237,
    "OR007@3_0": 144,
    "OR007@3_1": 145,
    "OR007@3_2": 146,
    "OR007@3_3": 147,
    "OR021@3_0": 246,
    "OR021@3_1": 247,
    "OR021@3_2": 248,
    "OR021@3_3": 249,
    "OR007@12_0": 156,
    "OR007@12_1": 158,
    "OR007@12_2": 159,
    "OR007@12_3": 160,
    "OR021@12_0": 258,
    "OR021@12_1": 259,
    "OR021@12_2": 260,
    "OR021@12_3": 261,
}

# Nominal speed per load from the catalogue. Only used for Normal_1 and Normal_2,
# which don't store an rpm; verify.py checks these against the spectrum.
NOMINAL_RPM: dict[int, float] = {0: 1797.0, 1: 1772.0, 2: 1750.0, 3: 1730.0}

# The normal baseline is 48 kHz, the 12k drive-end files are 12 kHz. The files
# don't say so; verify.sampling_rate_check proves it.
NATIVE_FS: dict[str, int] = {"Normal": 48_000}
DEFAULT_FS = 12_000

FAULT_TYPES = ("Normal", "IR", "B", "OR")

_NAME = re.compile(r"^(?P<fault>Normal|IR|B|OR)(?P<size>\d{3})?(?:@(?P<pos>\d+))?_(?P<load>\d)$")


@dataclass(frozen=True)
class Recording:
    """One recording: one physical bearing at one motor load."""

    name: str
    record: int
    fault: str  # "Normal", "IR", "B" or "OR"
    size_mils: int  # defect diameter in mils, 0 for Normal
    position: int | None  # outer-race clock position, else None
    load_hp: int
    native_fs: int
    core: bool  # in the 40-recording classification set

    @property
    def bearing(self) -> str:
        """ID of the physical bearing.

        Each bearing was recorded at four loads, so this (not the file) is the unit
        of independence.
        """
        if self.fault == "Normal":
            return "Normal"
        pos = f"@{self.position}" if self.position is not None else ""
        return f"{self.fault}{self.size_mils:03d}{pos}"

    @property
    def label10(self) -> str:
        """Fault type + size, the 10-class label most CWRU papers use."""
        return "Normal" if self.fault == "Normal" else f"{self.fault}{self.size_mils:03d}"

    @property
    def filename(self) -> str:
        return f"{self.name}.mat"


def parse(name: str) -> Recording:
    m = _NAME.match(name)
    if m is None or name not in RECORD_NUMBERS:
        raise KeyError(f"not a recording in this study: {name!r}")
    fault = m["fault"]
    size = int(m["size"]) if m["size"] else 0
    pos = int(m["pos"]) if m["pos"] else None
    core = fault in ("Normal", "IR", "B") or pos == 6
    return Recording(
        name=name,
        record=RECORD_NUMBERS[name],
        fault=fault,
        size_mils=size,
        position=pos,
        load_hp=int(m["load"]),
        native_fs=NATIVE_FS.get(fault, DEFAULT_FS),
        core=core,
    )


ALL: tuple[Recording, ...] = tuple(parse(n) for n in RECORD_NUMBERS)
CORE: tuple[Recording, ...] = tuple(r for r in ALL if r.core)
FAULTED: tuple[Recording, ...] = tuple(r for r in ALL if r.fault != "Normal")


def get(name: str) -> Recording:
    return parse(name)
