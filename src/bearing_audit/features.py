"""Cut recordings into windows and compute the features of each window.

Every feature uses a single window. The only recording-level step is the
loader's resampling, which is part of acquisition; filtering, envelopes and
spectra are all done per window, so nothing leaks across window boundaries.

Feature groups:
- time domain: RMS, kurtosis, skewness, crest factor, peak-to-peak
- spectral: relative energy in fixed bands up to 5 kHz, spectral kurtosis
- envelope: prominence of the envelope spectrum at BPFI, BPFO and 2xBSF,
  at each recording's own shaft speed
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from . import bearing, catalog, config, dsp, loader

SPECTRAL_EDGES_HZ = (0, 500, 1000, 2000, 3000, 4000, 5000)
SIGNATURES = ("BPFI", "BPFO", "2xBSF")

TIME_FEATURES = ["rms", "kurtosis", "skewness", "crest", "p2p"]
_BANDS = list(zip(SPECTRAL_EDGES_HZ[:-1], SPECTRAL_EDGES_HZ[1:], strict=True))
SPECTRAL_FEATURES = [f"band_{lo}_{hi}" for lo, hi in _BANDS] + ["sk_max", "sk_freq"]
ENVELOPE_FEATURES = [f"env_{s}_h{h}" for s in SIGNATURES for h in ("1", "m")]
FEATURES = TIME_FEATURES + SPECTRAL_FEATURES + ENVELOPE_FEATURES
SK_RULE_FEATURES = [f"envsk_{s}_h1" for s in SIGNATURES]
# inputs of the physics diagnoser (the audit models don't use them)
DIAGNOSER_FEATURES = [
    "env_BPFI_h1",
    "env_BPFI_sb",
    "env_BPFO_h1",
    "env_2xBSF_h1",
    "env_BSF_h1",
    "kurtosis",
    "crest",
    "sk_max",
]


def windows(x: np.ndarray, fs: int, seconds: float = config.WINDOW_S) -> Iterator[np.ndarray]:
    n = int(round(seconds * fs))
    for i in range(x.size // n):
        yield x[i * n : (i + 1) * n]


def prominence(
    f: np.ndarray, a: np.ndarray, target: float, ref_halfwidth: float, tol: float = config.PEAK_TOL
) -> tuple[float, float]:
    """Tallest peak within target*(1 +/- tol), divided by the median level
    within target +/- ref_halfwidth.

    A ratio rather than an amplitude: it asks whether a line stands out at this
    frequency, whatever the overall level of the machine.
    """
    search = (f >= target * (1 - tol)) & (f <= target * (1 + tol))
    ref = (f >= target - ref_halfwidth) & (f <= target + ref_halfwidth)
    if not search.any() or not ref.any():
        return float("nan"), float("nan")
    i = int(np.argmax(a[search]))
    floor = float(np.median(a[ref]))
    return float(f[search][i]), float(a[search][i] / max(floor, 1e-30))


def envelope_features(
    x: np.ndarray,
    fs: int,
    shaft_hz: float,
    band: tuple[float, float],
    prefix: str = "env",
    extended: bool = False,
    geometry: bearing.Geometry = bearing.SKF_6205,
) -> dict[str, float]:
    f, a = dsp.envelope_spectrum(x, fs, band)
    freqs = geometry.frequencies(shaft_hz)
    out: dict[str, float] = {}
    for sig in SIGNATURES:
        f0 = freqs[sig]
        proms = [prominence(f, a, k * f0, ref_halfwidth=0.5 * f0)[1] for k in range(1, config.N_HARMONICS + 1)]
        out[f"{prefix}_{sig}_h1"] = float(np.log10(proms[0]))
        out[f"{prefix}_{sig}_hm"] = float(np.log10(np.mean(proms)))
    if extended:
        # an inner-race defect turns with the shaft through the load zone, so its
        # impacts are amplitude-modulated at shaft speed, giving sidebands at
        # BPFI +/- f_r. Seeing them confirms the defect is on the inner race.
        bpfi = freqs["BPFI"]
        sb = [prominence(f, a, bpfi + d, ref_halfwidth=0.5 * shaft_hz)[1] for d in (-shaft_hz, shaft_hz)]
        out[f"{prefix}_BPFI_sb"] = float(np.log10(np.mean(sb)))
        # a ball defect can also show at 1 x BSF (one race), not just 2 x BSF
        bsf = freqs["BSF"]
        out[f"{prefix}_BSF_h1"] = float(np.log10(prominence(f, a, bsf, ref_halfwidth=0.5 * bsf)[1]))
    return out


def time_features(x: np.ndarray) -> dict[str, float]:
    x = x - x.mean()
    rms = float(np.sqrt(np.mean(x**2)))
    z = x / rms
    return {
        "rms": rms,
        "kurtosis": float(np.mean(z**4)),  # Pearson, so 3 for Gaussian noise
        "skewness": float(np.mean(z**3)),
        "crest": float(np.max(np.abs(x)) / rms),
        "p2p": float(np.ptp(x)),
    }


def spectral_features(x: np.ndarray, fs: int, bandwidth: float = config.BANDWIDTH_HZ) -> dict[str, float]:
    f, a = dsp.amplitude_spectrum(x, fs)
    p = a**2
    in_bw = f <= bandwidth
    total = p[in_bw].sum()
    out = {}
    for lo, hi in _BANDS:
        band = (f >= lo) & (f < hi)
        out[f"band_{lo}_{hi}"] = float(np.log10(p[band].sum() / total))
    fk, sk = dsp.spectral_kurtosis(x, fs, nperseg=64)
    ok = (fk >= config.SK_FMIN_HZ) & (fk <= bandwidth)
    j = int(np.argmax(sk[ok]))
    out["sk_max"] = float(sk[ok][j])
    out["sk_freq"] = float(fk[ok][j])
    return out


def window_features(
    x: np.ndarray,
    fs: int,
    shaft_hz: float,
    bandwidth: float = config.BANDWIDTH_HZ,
    geometry: bearing.Geometry = bearing.SKF_6205,
) -> dict[str, float]:
    """All features of one window.

    ``bandwidth`` is only changed by the post-hoc sensitivity check (the analysis
    itself uses 5 kHz). ``geometry`` lets the public API handle other bearings.
    """
    x = dsp.lowpass(x, fs, bandwidth)
    env_band = (config.ENVELOPE_BAND_HZ[0], min(config.ENVELOPE_BAND_HZ[1], bandwidth - 100.0))
    feats = time_features(x) | spectral_features(x, fs, bandwidth)
    feats |= envelope_features(x, fs, shaft_hz, env_band, prefix="env", extended=True, geometry=geometry)
    sk_band, _ = dsp.select_band(x, fs, fmin=config.SK_FMIN_HZ, fmax=bandwidth)
    sk = envelope_features(x, fs, shaft_hz, sk_band, prefix="envsk", geometry=geometry)
    feats |= {k: v for k, v in sk.items() if k.endswith("_h1")}
    return feats


def recording_rows(sig: loader.Signal, bandwidth: float = config.BANDWIDTH_HZ) -> list[dict]:
    r = sig.recording
    rows = []
    for i, w in enumerate(windows(sig.x, sig.fs)):
        rows.append(
            {
                "recording": r.name,
                "window": i,
                "bearing": r.bearing,
                "fault": r.fault,
                "label10": r.label10,
                "size_mils": r.size_mils,
                "load_hp": r.load_hp,
                "rpm": sig.rpm,
                **window_features(w, sig.fs, sig.shaft_hz, bandwidth),
            }
        )
    return rows


def build_table(
    recordings: Iterable[catalog.Recording], data_dir: Path | str, bandwidth: float = config.BANDWIDTH_HZ
) -> pd.DataFrame:
    rows: list[dict] = []
    for r in recordings:
        rows += recording_rows(loader.load(r.name, data_dir), bandwidth)
    return pd.DataFrame(rows)
