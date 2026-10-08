"""Synthetic signals with known answers. Only tests marked ``data`` need the
CWRU files, and they skip when data/raw is empty.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # file-only backend, so the tests behave the same on every OS

DATA = Path(__file__).resolve().parents[1] / "data" / "raw"


def bearing_fault(
    fs=12_000, seconds=2.0, fault_hz=97.0, resonance_hz=3_000.0, damping=600.0, noise=0.05, seed=0
) -> np.ndarray:
    """Impacts at fault_hz, each ringing a damped resonance, plus white noise.

    This is the model envelope analysis rests on: the fault rate is in the
    envelope, the energy is at the resonance.
    """
    rng = np.random.default_rng(seed)
    n = int(fs * seconds)
    x = np.zeros(n)
    ring_t = np.arange(int(0.01 * fs)) / fs
    ring = np.exp(-damping * ring_t) * np.sin(2 * np.pi * resonance_hz * ring_t)
    for t0 in np.arange(0, seconds, 1 / fault_hz):
        i = int(t0 * fs)
        seg = ring[: n - i]
        x[i : i + seg.size] += seg
    return x + noise * rng.standard_normal(n)


def synthetic_table(seed: int = 0) -> pd.DataFrame:
    """Feature table shaped like the real one (40 recordings, 10 bearings,
    395 windows). Each fault type lights up its own envelope line, IR windows have
    sidebands and healthy windows show nothing, so the right answer is known for
    every system.
    """
    from bearing_audit import catalog, features

    rng = np.random.default_rng(seed)
    signal = {"IR": "BPFI", "OR": "BPFO", "B": "2xBSF"}
    cols = sorted(set(features.FEATURES + features.SK_RULE_FEATURES + features.DIAGNOSER_FEATURES))
    rows = []
    for r in catalog.CORE:
        for w in range(5 if r.name == "Normal_0" else 10):
            row = {c: rng.normal(0, 0.05) for c in cols}
            row |= {c: -0.5 + rng.normal(0, 0.05) for c in cols if c.startswith("env")}
            row |= {
                "kurtosis": 3 + rng.normal(0, 0.05),
                "crest": 4 + rng.normal(0, 0.05),
                "sk_max": rng.normal(0, 0.05),
            }
            if r.fault in signal:
                for prefix in ("env", "envsk"):
                    row[f"{prefix}_{signal[r.fault]}_h1"] = 1.5 + rng.normal(0, 0.05)
                row[f"env_{signal[r.fault]}_hm"] = 1.2
                if r.fault == "IR":
                    row["env_BPFI_sb"] = 1.0
                row["rms"] = 1.0 + r.size_mils / 10  # so fault size is learnable for the 10-class task
            row |= {
                "recording": r.name,
                "window": w,
                "bearing": r.bearing,
                "fault": r.fault,
                "label10": r.label10,
                "size_mils": r.size_mils,
                "load_hp": r.load_hp,
            }
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def fault_signal() -> np.ndarray:
    return bearing_fault()


def pytest_collection_modifyitems(config, items):
    have_data = DATA.exists() and any(DATA.glob("*.mat"))
    skip = pytest.mark.skip(reason="CWRU files not in data/raw (run `make fetch`)")
    for item in items:
        if "data" in item.keywords and not have_data:
            item.add_marker(skip)
