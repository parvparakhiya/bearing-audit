"""Public API: known-answer signals, input validation, JSON round trip."""

import json

import numpy as np
import pytest

from bearing_audit import SKF_6205, BearingMonitor, Geometry, InputError

from .conftest import bearing_fault

FS, RPM = 12_000, 1797.0
SHAFT = RPM / 60


def noise(seconds=10.0, seed=0, fs=FS):
    return 0.1 * np.random.default_rng(seed).standard_normal(int(fs * seconds))


def inner_race(seconds=10.0, seed=1, fs=FS):
    """Impacts at BPFI whose strength goes up and down once per shaft turn,
    like an inner-race defect passing through the load zone.
    """
    rng = np.random.default_rng(seed)
    n = int(fs * seconds)
    x = np.zeros(n)
    bpfi = SKF_6205.frequencies(SHAFT)["BPFI"]
    t = np.arange(int(0.01 * fs)) / fs
    ring = np.exp(-600 * t) * np.sin(2 * np.pi * 3_000 * t)
    for t0 in np.arange(0, seconds, 1 / bpfi):
        i = int(t0 * fs)
        seg = ring[: n - i] * (1 + 0.8 * np.cos(2 * np.pi * SHAFT * t0))
        x[i : i + seg.size] += seg
    return x + 0.05 * rng.standard_normal(n)


def outer_race(seconds=10.0, fs=FS):
    return bearing_fault(fs=fs, seconds=seconds, fault_hz=SKF_6205.frequencies(SHAFT)["BPFO"])


@pytest.fixture(scope="module")
def monitor():
    return BearingMonitor().fit_baseline([(noise(seed=10), FS, 1797), (noise(seed=11), FS, 1772)])


def test_healthy_measurement_stays_healthy(monitor):
    d = monitor.diagnose(noise(seed=12), FS, RPM)
    assert d.state == "Healthy" and not d.alarm and not d.warnings


def test_inner_race_fault_is_named_with_its_sidebands(monitor):
    d = monitor.diagnose(inner_race(), FS, RPM)
    assert d.state == "IR"
    assert d.evidence["IR"] > 0 and d.evidence["IR_sb"] > 0


def test_outer_race_fault_is_named(monitor):
    assert monitor.diagnose(outer_race(), FS, RPM).state == "OR"


def test_higher_sampling_rates_are_resampled(monitor):
    assert monitor.diagnose(outer_race(fs=48_000), 48_000, RPM).state == "OR"


def test_rpm_outside_the_baseline_raises_a_warning(monitor):
    d = monitor.diagnose(noise(seed=13), FS, 1500)
    assert d.warnings and "outside the baseline range" in d.warnings[0]


def test_save_and_load_give_the_same_diagnosis(monitor, tmp_path):
    path = tmp_path / "baseline.json"
    monitor.save(path)
    saved = json.loads(path.read_text())
    assert set(saved) >= {"format_version", "geometry", "thresholds", "baseline_rpm_range", "analysis"}
    again = BearingMonitor.load(path)
    x = inner_race(seed=5)
    assert again.diagnose(x, FS, RPM) == monitor.diagnose(x, FS, RPM)


def test_load_refuses_a_baseline_made_with_other_settings(monitor, tmp_path):
    d = monitor.to_dict()
    for bad in ({**d, "format_version": 99}, {**d, "analysis": {**d["analysis"], "bandwidth_hz": 6000.0}}):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(bad))
        with pytest.raises(InputError):
            BearingMonitor.load(path)


def test_other_bearing_geometries_are_supported():
    geo = Geometry("SKF 6205 in mm", n_elements=9, element_diameter=7.94, pitch_diameter=39.04)
    assert geo.multiples()["BPFI"] == pytest.approx(SKF_6205.multiples()["BPFI"], rel=1e-3)
    m = BearingMonitor(geo).fit_baseline([(noise(seed=20), FS, RPM)])
    assert m.diagnose(outer_race(), FS, RPM).state == "OR"


@pytest.mark.parametrize(
    ("x", "fs", "rpm", "message"),
    [
        (np.zeros((2, 12_000)), FS, RPM, "1-D"),
        (np.full(12_000, np.nan), FS, RPM, "NaN"),
        (noise(0.5), FS, RPM, "at least 1 s"),
        (noise(), 10_000, RPM, "sampling rate"),
        (noise(), 12_000.0, RPM, "sampling rate"),
        (noise(), FS, 0, "rpm"),
        (noise(), FS, float("nan"), "rpm"),
        (np.ones(24_000), FS, RPM, "constant"),
    ],
)
def test_bad_input_is_refused_with_a_clear_message(monitor, x, fs, rpm, message):
    with pytest.raises(InputError, match=message):
        monitor.diagnose(x, fs, rpm)


def test_diagnose_before_a_baseline_is_refused():
    with pytest.raises(InputError, match="fit_baseline"):
        BearingMonitor().diagnose(noise(), FS, RPM)


def test_a_baseline_that_is_too_short_is_refused():
    with pytest.raises(InputError, match="at least 10"):
        BearingMonitor().fit_baseline([(noise(5.0), FS, RPM)])
