import numpy as np
import pytest

from bearing_audit import dsp


def test_amplitude_spectrum_recovers_sine_amplitude():
    fs, f0, amp = 12_000, 250.0, 0.7
    t = np.arange(fs * 2) / fs
    f, a = dsp.amplitude_spectrum(amp * np.sin(2 * np.pi * f0 * t), fs)
    assert f[np.argmax(a)] == pytest.approx(f0, abs=0.5)
    assert a.max() == pytest.approx(amp, rel=0.02)


def test_envelope_spectrum_finds_the_impact_rate(fault_signal):
    f, a = dsp.envelope_spectrum(fault_signal, 12_000, (2_000, 4_500))
    band = (f > 20) & (f < 500)
    assert f[band][np.argmax(a[band])] == pytest.approx(97.0, abs=1.0)


def test_raw_spectrum_does_not_show_the_impact_rate(fault_signal):
    """The reason envelope analysis exists: the energy isn't at the fault rate."""
    f, a = dsp.amplitude_spectrum(fault_signal, 12_000)
    assert 2_500 < f[np.argmax(a)] < 3_500


def test_resample_keeps_in_band_tone_and_removes_out_of_band_tone():
    fs_in, fs_out = 48_000, 12_000
    t = np.arange(fs_in * 2) / fs_in
    x = np.sin(2 * np.pi * 120 * t) + np.sin(2 * np.pi * 9_000 * t)  # 9 kHz is above the new Nyquist
    good = dsp.resample(x, fs_in, fs_out)
    f, a = dsp.amplitude_spectrum(good, fs_out)
    alias = (f > 2_900) & (f < 3_100)  # 9 kHz folds to 12 - 9 = 3 kHz
    assert a[np.argmin(np.abs(f - 120))] == pytest.approx(1.0, rel=0.02)
    assert a[alias].max() < 1e-3


def test_slicing_instead_of_resampling_creates_a_fake_line():
    """What resample() protects against: x[::4] turns 9 kHz into a 3 kHz line."""
    fs_in = 48_000
    t = np.arange(fs_in * 2) / fs_in
    x = np.sin(2 * np.pi * 9_000 * t)
    f, a = dsp.amplitude_spectrum(x[::4], 12_000)
    assert f[np.argmax(a)] == pytest.approx(3_000, abs=1)
    assert a.max() > 0.9


def test_bandpass_is_zero_phase():
    fs = 12_000
    x = np.zeros(fs)
    x[6_000] = 1.0
    y = dsp.bandpass(x, fs, 2_000, 4_000)
    assert abs(int(np.argmax(np.abs(y))) - 6_000) <= 1


def test_bandpass_rejects_impossible_band():
    with pytest.raises(ValueError):
        dsp.bandpass(np.zeros(100), 12_000, 5_000, 7_000)


def test_spectral_kurtosis_is_near_zero_for_gaussian_noise():
    x = np.random.default_rng(1).standard_normal(120_000)
    _, sk = dsp.spectral_kurtosis(x, 12_000, nperseg=64)
    assert abs(np.median(sk)) < 0.2


def test_spectral_kurtosis_peaks_at_the_resonance(fault_signal):
    f, sk = dsp.spectral_kurtosis(fault_signal, 12_000, nperseg=64)
    assert 2_500 <= f[np.argmax(sk)] <= 3_500


def test_select_band_stays_inside_limits_and_demodulates_the_fault(fault_signal):
    """The band only has to reveal the impact rate in the envelope.

    It doesn't have to sit on the resonance: very short STFT windows also see
    impact onsets as impulsive away from it. A known weakness of this simplified
    kurtogram, and one reason the fixed band is the main method.
    """
    band, sk = dsp.select_band(fault_signal, 12_000, fmin=500, fmax=5_000)
    assert 500 <= band[0] < band[1] <= 5_000
    assert sk > 1
    f, a = dsp.envelope_spectrum(fault_signal, 12_000, band)
    w = (f > 20) & (f < 500)
    assert f[w][np.argmax(a[w])] == pytest.approx(97.0, abs=1.0)
