import numpy as np
import pytest

from bearing_audit import bearing, config, features, verify

from .conftest import bearing_fault


def test_time_features_of_a_sine():
    t = np.arange(12_000) / 12_000
    x = 2.0 * np.sin(2 * np.pi * 50 * t)
    ft = features.time_features(x)
    assert ft["rms"] == pytest.approx(2 / np.sqrt(2), rel=1e-3)
    assert ft["crest"] == pytest.approx(np.sqrt(2), rel=1e-3)
    assert ft["kurtosis"] == pytest.approx(1.5, rel=1e-2)
    assert ft["p2p"] == pytest.approx(4.0, rel=1e-3)


def test_gaussian_kurtosis_is_three():
    x = np.random.default_rng(0).standard_normal(200_000)
    assert features.time_features(x)["kurtosis"] == pytest.approx(3.0, abs=0.05)


def test_windows_do_not_overlap_and_drop_the_tail():
    x = np.arange(12_000 * 3 + 500, dtype=float)
    w = list(features.windows(x, 12_000, 1.0))
    assert len(w) == 3
    assert [v[0] for v in w] == [0, 12_000, 24_000]


def test_envelope_features_single_out_the_right_fault_frequency():
    shaft = 29.95
    bpfi = bearing.SKF_6205.frequencies(shaft)["BPFI"]
    x = bearing_fault(fault_hz=bpfi, seconds=1.0)
    env = features.envelope_features(x, 12_000, shaft, config.ENVELOPE_BAND_HZ)
    assert env["env_BPFI_h1"] > np.log10(config.PRESENT_PROMINENCE)
    assert env["env_BPFI_h1"] > env["env_BPFO_h1"] + 1
    assert env["env_BPFI_h1"] > env["env_2xBSF_h1"] + 1


def test_every_feature_is_finite():
    f = features.window_features(bearing_fault(seconds=1.0), 12_000, 29.95)
    assert set(features.FEATURES + features.SK_RULE_FEATURES) <= set(f)
    assert all(np.isfinite(v) for v in f.values())


@pytest.mark.parametrize("tone_hz", [5_300.0, 5_600.0])
def test_no_feature_reacts_to_content_where_the_recorders_differ(tone_hz):
    """Guard against the acquisition confound.

    The two recorders differ from ~5.2-5.3 kHz upwards. A tone there, louder than
    the bearing signal itself, must not change any feature (forest or rule).
    """
    x = bearing_fault(seconds=1.0)
    t = np.arange(x.size) / 12_000
    base = features.window_features(x, 12_000, 29.95)
    moved = features.window_features(x + 0.5 * np.sin(2 * np.pi * tone_hz * t), 12_000, 29.95)
    for k in sorted(set(features.FEATURES + features.SK_RULE_FEATURES + features.DIAGNOSER_FEATURES)):
        assert moved[k] == pytest.approx(base[k], rel=0.01, abs=0.005), k


def test_lowpass_transition_band_is_documented():
    g = verify.lowpass_response()
    assert g["4500"] > 0.99 and g["5000"] == pytest.approx(0.5, abs=0.01)
    assert g["5200"] < 0.03 and g["5400"] < 1e-3
