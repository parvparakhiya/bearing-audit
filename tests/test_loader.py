import numpy as np
import pytest
from scipy.io import savemat

from bearing_audit import catalog, loader


def _mat(path, **vars_):
    savemat(path, {k: np.asarray(v).reshape(-1, 1) for k, v in vars_.items()})


def test_loader_selects_channel_by_record_number_not_by_position(tmp_path):
    """Normal_2.mat really does contain record 98 before record 99."""
    rng = np.random.default_rng(0)
    rec98, rec99 = rng.standard_normal(48_000), rng.standard_normal(48_000) + 10
    _mat(tmp_path / "Normal_2.mat", X098_DE_time=rec98, X098_FE_time=rec98, X099_DE_time=rec99)
    x, rpm = loader.read_raw(tmp_path / "Normal_2.mat", catalog.get("Normal_2"))
    assert np.array_equal(x, rec99)
    assert rpm is None


def test_normal_files_are_resampled_to_12k_and_rpm_falls_back_to_catalogue(tmp_path):
    _mat(tmp_path / "Normal_2.mat", X099_DE_time=np.zeros(48_000 * 2))
    s = loader.load("Normal_2", tmp_path)
    assert s.fs == 12_000
    assert s.x.size == 12_000 * 2
    assert s.rpm == catalog.NOMINAL_RPM[2]
    assert s.rpm_source == "catalogue"


def test_fault_files_keep_their_rate_and_rpm(tmp_path):
    _mat(tmp_path / "IR007_0.mat", X105_DE_time=np.zeros(12_000), X105RPM=np.array([1797.0]))
    s = loader.load("IR007_0", tmp_path)
    assert (s.fs, s.x.size, s.rpm, s.rpm_source) == (12_000, 12_000, 1797.0, "file")


def test_missing_channel_is_an_error_not_a_silent_fallback(tmp_path):
    _mat(tmp_path / "IR007_0.mat", X999_DE_time=np.zeros(10))
    with pytest.raises(ValueError, match="record 105"):
        loader.load("IR007_0", tmp_path)


def test_catalogue_shape():
    assert len(catalog.ALL) == 56
    assert len(catalog.CORE) == 40
    assert len({r.bearing for r in catalog.CORE}) == 10
    assert all(r.size_mils != 28 for r in catalog.ALL)
    assert {r.native_fs for r in catalog.ALL if r.fault == "Normal"} == {48_000}
    assert catalog.get("OR007@6_2").bearing == "OR007@6"
    assert catalog.get("OR007@3_2").core is False
