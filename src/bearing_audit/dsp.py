"""Signal processing helpers on top of numpy/scipy.

Envelope analysis in short:

    raw -> band-pass around a resonance -> |hilbert| (envelope) -> FFT

A defect doesn't put its energy at the fault frequency. Each impact rings the
housing at a resonance in the kHz range, and the loudness of that ringing goes
up and down once per impact. So the fault frequency is easiest to see in the
envelope, not in the raw spectrum.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, hilbert, resample_poly, sosfiltfilt, stft


def amplitude_spectrum(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Single-sided amplitude spectrum with a Hann window.

    Scaled so a sine of amplitude A gives a peak of about A. The mean is removed
    first so the DC bin doesn't dominate.
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    w = np.hanning(x.size)
    amp = 2.0 * np.abs(np.fft.rfft(x * w)) / w.sum()
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
    return freqs, amp


def resample(x: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    """Resample with an anti-alias filter (polyphase).

    Don't decimate by slicing (x[::4]): anything above the new Nyquist folds back
    as a fake low-frequency line that looks exactly like a real one.
    """
    if fs_in == fs_out:
        return np.asarray(x, dtype=float)
    g = np.gcd(fs_in, fs_out)
    return resample_poly(np.asarray(x, dtype=float), fs_out // g, fs_in // g)


def bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass.

    Uses second-order sections because the (b, a) form gets numerically unstable
    for narrow, high-order band-passes. sosfiltfilt runs forward and backward, so
    impacts aren't shifted in time.
    """
    nyq = fs / 2
    if not 0 < lo < hi < nyq:
        raise ValueError(f"band must satisfy 0 < lo < hi < fs/2; got {lo}, {hi}, fs/2={nyq}")
    sos = butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, np.asarray(x, dtype=float))


def lowpass(x: np.ndarray, fs: float, cutoff: float, order: int = 8) -> np.ndarray:
    """Zero-phase Butterworth low-pass."""
    sos = butter(order, cutoff / (fs / 2), btype="low", output="sos")
    return sosfiltfilt(sos, np.asarray(x, dtype=float))


def envelope(x: np.ndarray) -> np.ndarray:
    """Envelope: magnitude of the analytic signal."""
    return np.abs(hilbert(np.asarray(x, dtype=float)))


def envelope_spectrum(x: np.ndarray, fs: float, band: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Band-pass, take the envelope, then its amplitude spectrum."""
    env = envelope(bandpass(x, fs, *band))
    return amplitude_spectrum(env, fs)


def spectral_kurtosis(x: np.ndarray, fs: float, nperseg: int) -> tuple[np.ndarray, np.ndarray]:
    """Spectral kurtosis per frequency bin, from an STFT.

        SK(f) = E|X(t,f)|^4 / (E|X(t,f)|^2)^2 - 2

    Roughly 0 for stationary Gaussian noise. Repeated impacts make the band they
    excite bursty over time, so SK peaks where the defect rings the structure.
    Used to pick the demodulation band instead of choosing it by eye.
    """
    f, _, z = stft(
        np.asarray(x, dtype=float),
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg * 3 // 4,
        boundary=None,
        padded=False,
    )
    p = np.abs(z) ** 2
    sk = np.mean(p**2, axis=1) / np.maximum(np.mean(p, axis=1) ** 2, 1e-30) - 2.0
    return f, sk


def select_band(
    x: np.ndarray,
    fs: float,
    fmin: float,
    fmax: float,
    npersegs: tuple[int, ...] = (16, 24, 32, 48, 64),
) -> tuple[tuple[float, float], float]:
    """Pick the demodulation band with the highest spectral kurtosis.

    Simplified kurtogram: SK at several STFT lengths. N samples resolve bands of
    about 2*fs/N, so short windows test wide bands and long ones narrow bands.
    Bands must lie inside [fmin, fmax], and ties go to the wider band.
    Returns ((lo, hi), sk).
    """
    best: tuple[float, tuple[float, float]] = (-np.inf, (fmin, fmax))
    for n in npersegs:
        half = fs / n  # half the band width this window length resolves
        f, sk = spectral_kurtosis(x, fs, n)
        for fc, k in zip(f, sk, strict=True):
            lo, hi = fc - half, fc + half
            if lo < fmin or hi > fmax:
                continue
            if k > best[0] + 1e-12:
                best = (float(k), (float(lo), float(hi)))
    return best[1], best[0]
