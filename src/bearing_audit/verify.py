"""Checks that have to pass before any model result means anything.

- manifest: what every file is, with its SHA-256
- sampling_rate_check: each file's sampling rate, proven from physics
- acquisition_confound: what the recorder mismatch would give a classifier for free
- physics_check: does each faulty recording show its fault frequency where the
  shaft speed says it should be?
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score

from . import bearing, catalog, config, dsp, features, loader


def manifest(data_dir: Path | str) -> pd.DataFrame:
    rows = []
    for r in catalog.ALL:
        path = Path(data_dir) / r.filename
        x, rpm = loader.read_raw(path, r)
        rows.append(
            {
                "file": r.filename,
                "record": r.record,
                "bearing": r.bearing,
                "fault": r.fault,
                "size_mils": r.size_mils,
                "position": r.position,
                "load_hp": r.load_hp,
                "core": r.core,
                "native_fs": r.native_fs,
                "n_samples": x.size,
                "duration_s": round(x.size / r.native_fs, 3),
                "rpm": rpm if rpm is not None else catalog.NOMINAL_RPM[r.load_hp],
                "rpm_source": "file" if rpm is not None else "catalogue",
                "sha256": loader.sha256(path),
            }
        )
    return pd.DataFrame(rows)


def sampling_rate_check(data_dir: Path | str, tol_hz: float = 0.3) -> pd.DataFrame:
    """The files don't store their sampling rate, so test both hypotheses.

    At the true rate, the tallest line within +/-10 % of rpm/60 sits on the shaft
    speed and follows it from load to load. At the wrong rate everything is scaled
    by 4 and something else lands there: for a 48 kHz file read as 12 kHz it's the
    120 Hz electrical line, stuck at 30.0 Hz while the shaft slows down.

    Two caveats, which is why all four loads and the duration are checked:
    - at 0 HP the misread line (29.96 Hz in Normal_0) is only 0.03 Hz from the
      29.93 Hz shaft speed, so that file on its own passes the wrong rate too;
    - read at 1/4 scale, a real 4th shaft harmonic would land exactly on rpm/60.
      It works here because the 120 Hz line is taller than that harmonic at
      1-3 HP. Duration is the independent check: at the catalogued rate files
      last ~5 or ~10 s, at the other rate the healthy files would be 20-40 s and
      the fault files 2.5 s.
    """
    rows = []
    for r in catalog.ALL:
        x, rpm = loader.read_raw(Path(data_dir) / r.filename, r)
        fr = (rpm if rpm is not None else catalog.NOMINAL_RPM[r.load_hp]) / 60.0
        row = {
            "recording": r.name,
            "group": "Normal" if r.fault == "Normal" else "fault",
            "catalogued_fs": r.native_fs,
            "shaft_hz": fr,
            "duration_12k_s": x.size / 12_000,
            "duration_48k_s": x.size / 48_000,
        }
        for fs in (12_000, 48_000):
            f, a = dsp.amplitude_spectrum(x, fs)
            w = (f >= 0.9 * fr) & (f <= 1.1 * fr)
            err = float(f[w][np.argmax(a[w])] - fr)
            row[f"err_{fs // 1000}k_hz"] = err
            row[f"match_{fs // 1000}k"] = abs(err) <= tol_hz
        rows.append(row)
    return pd.DataFrame(rows)


def lowpass_response(freqs=(4_500.0, 5_000.0, 5_200.0, 5_400.0, 5_600.0), fs: int = 12_000) -> dict[str, float]:
    """Gain of the analysis low-pass (zero-phase, so squared) at a few
    frequencies, to show how much of 5.0-5.4 kHz still gets through.
    """
    from scipy.signal import butter, sosfreqz

    sos = butter(8, config.BANDWIDTH_HZ / (fs / 2), btype="low", output="sos")
    _, h = sosfreqz(sos, worN=np.array(freqs), fs=fs)
    return {f"{f:.0f}": float(np.abs(g) ** 2) for f, g in zip(freqs, h, strict=True)}


def acquisition_confound(data_dir: Path | str) -> dict:
    """What a model gets for free from the recorder if nothing is band-limited.

    One feature (energy above 5.5 kHz relative to 1-4 kHz) on the raw signals,
    used to separate Normal from faulty windows with leave-one-load-out CV. Each
    fold holds out one healthy and nine faulty recordings, so both classes are
    always in the test set.
    """
    ratio: dict[str, list[float]] = {}
    X, y, g = [], [], []
    for r in catalog.CORE:
        s = loader.load(r.name, data_dir)
        f, p = welch(s.x, s.fs, nperseg=2048)
        ratio.setdefault(r.fault, []).append(
            float(np.median(p[(f > 5600) & (f < 5950)]) / np.median(p[(f > 1000) & (f < 4000)]))
        )
        for w in features.windows(s.x, s.fs):
            fw, pw = welch(w, s.fs, nperseg=1024)
            X.append([np.log10(pw[fw > 5500].sum() / pw[(fw > 1000) & (fw < 4000)].sum())])
            y.append(r.fault == "Normal")
            g.append(r.load_hp)
    acc = cross_val_score(
        LogisticRegression(), np.array(X), np.array(y), groups=g, cv=LeaveOneGroupOut(), scoring="balanced_accuracy"
    )
    return {
        "psd_ratio_5p6_6k_over_1_4k_median": {k: float(np.median(v)) for k, v in ratio.items()},
        "one_feature_balanced_accuracy_leave_one_load_out": float(acc.mean()),
        "one_feature_fold_accuracies": [float(v) for v in acc],
        "analysis_bandwidth_hz": config.BANDWIDTH_HZ,
    }


def physics_check(data_dir: Path | str) -> pd.DataFrame:
    """Envelope spectrum of every faulty recording (full length).

    For each one: the predicted signature frequency at that recording's speed,
    the measured peak near it and how much it stands out, in the fixed band and
    in the SK-selected band.
    """
    rows = []
    for r in catalog.FAULTED:
        s = loader.load(r.name, data_dir)
        x = dsp.lowpass(s.x, s.fs, config.BANDWIDTH_HZ)
        sig = bearing.SIGNATURE[r.fault]
        pred = bearing.SKF_6205.frequencies(s.shaft_hz)[sig]
        f, a = dsp.envelope_spectrum(x, s.fs, config.ENVELOPE_BAND_HZ)
        meas, prom = features.prominence(f, a, pred, ref_halfwidth=0.5 * pred)
        sk_band, sk = dsp.select_band(x, s.fs, fmin=config.SK_FMIN_HZ, fmax=config.BANDWIDTH_HZ)
        fs_, as_ = dsp.envelope_spectrum(x, s.fs, sk_band)
        _, prom_sk = features.prominence(fs_, as_, pred, ref_halfwidth=0.5 * pred)
        w = (f >= 50) & (f <= 400)
        top = float(f[w][np.argmax(a[w])])
        rows.append(
            {
                "recording": r.name,
                "bearing": r.bearing,
                "fault": r.fault,
                "core": r.core,
                "load_hp": r.load_hp,
                "rpm": s.rpm,
                "shaft_hz": s.shaft_hz,
                "signature": sig,
                "predicted_hz": pred,
                "measured_hz": meas,
                "measured_over_predicted": meas / pred,
                "measured_multiple": meas / s.shaft_hz,
                "prominence": prom,
                "present": prom >= config.PRESENT_PROMINENCE,
                "sk_band_lo": sk_band[0],
                "sk_band_hi": sk_band[1],
                "sk": sk,
                "prominence_sk_band": prom_sk,
                "top_peak_hz": top,
                "top_peak_multiple": top / s.shaft_hz,
            }
        )
    return pd.DataFrame(rows)
