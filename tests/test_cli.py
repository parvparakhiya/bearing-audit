"""CLI: argument parsing and the checksum rule in fetch."""

import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from bearing_audit import catalog, cli, loader


def test_help_lists_every_command(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    for cmd in ("fetch", "verify", "physics", "audit", "industrial", "figures", "all"):
        assert cmd in out


@pytest.fixture
def one_file_world(tmp_path, monkeypatch):
    """A catalogue with one recording, plus a manifest and a source folder."""
    src = tmp_path / "src"
    src.mkdir()
    good = src / "IR007_0.mat"
    savemat(good, {"X105_DE_time": np.zeros((100, 1)), "X105RPM": np.array([[1797.0]])})
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame({"file": ["IR007_0.mat"], "sha256": [loader.sha256(good)]}).to_csv(manifest, index=False)
    monkeypatch.setattr(catalog, "ALL", (catalog.get("IR007_0"),))
    monkeypatch.setattr(cli, "DATA", tmp_path / "raw")
    monkeypatch.setattr(cli, "MANIFEST", manifest)
    return src


def test_fetch_accepts_a_file_whose_checksum_matches(one_file_world):
    assert cli.main(["fetch", "--from-dir", str(one_file_world)]) == 0
    assert (cli.DATA / "IR007_0.mat").exists()


def test_fetch_rejects_and_removes_a_file_whose_checksum_does_not_match(one_file_world):
    savemat(one_file_world / "IR007_0.mat", {"X105_DE_time": np.ones((100, 1))})  # tampered
    assert cli.main(["fetch", "--from-dir", str(one_file_world)]) == 1
    assert not (cli.DATA / "IR007_0.mat").exists()
