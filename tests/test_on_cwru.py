"""Checks on the real recordings; skipped when the data isn't downloaded."""

from pathlib import Path

import pandas as pd
import pytest

from bearing_audit import loader, verify

from .conftest import DATA

pytestmark = pytest.mark.data
MANIFEST = Path(__file__).resolve().parents[1] / "data" / "manifest.csv"


def test_files_match_the_committed_checksums():
    m = pd.read_csv(MANIFEST)
    for file, sha in zip(m.file, m.sha256, strict=True):
        assert loader.sha256(DATA / file) == sha, file


def test_sampling_rates_are_proven_by_the_shaft_line():
    r = verify.sampling_rate_check(DATA)
    normal, fault = r[r.group == "Normal"], r[r.group == "fault"]
    assert normal.match_48k.all() and normal.match_12k.sum() <= 1
    assert fault.match_12k.sum() >= 50 and fault.match_48k.sum() <= 5


def test_inner_race_signature_present_in_every_recording():
    p = verify.physics_check(DATA)
    ir = p[p.fault == "IR"]
    assert ir.present.all()
    assert (ir.measured_over_predicted - 1).abs().max() < 0.01


def test_industrial_evaluation_reproduces_the_audit_forest_window_predictions():
    """The industrial evaluation re-runs the audit's forest in its own pipeline. The
    two must agree window for window, otherwise it isn't comparing like with like.
    Both sides are computed live, so this holds for any scikit-learn version.
    """
    from bearing_audit import evaluate, industrial

    feats = pd.read_csv(Path(__file__).resolve().parents[1] / "reports" / "results" / "features.csv")
    windows, _ = industrial.window_states(feats)
    audit = evaluate.predictions(feats, "C_unseen_bearing", "forest")
    merged = windows.merge(audit[["fold", "recording", "window", "pred"]], on=["fold", "recording", "window"])
    assert len(merged) == len(windows) == len(audit)
    assert (merged.forest == merged.pred.replace("Normal", "Healthy")).all()


def test_every_figure_renders(tmp_path):
    from bearing_audit import plots

    results = Path(__file__).resolve().parents[1] / "reports" / "results"
    plots.make_all(DATA, results, tmp_path)
    assert len(list(tmp_path.glob("fig*.png"))) == 9


def test_public_api_on_real_recordings():
    """Baseline from the healthy bearing at 0-2 HP, diagnose recordings at 3 HP.

    Signature bearings get the right name, a bearing without a signature stays
    undiagnosed, and the healthy bearing at 3 HP (where the industrial evaluation had its false alarms)
    comes with an out-of-range warning.
    """
    from bearing_audit import BearingMonitor, catalog, loader

    def raw(name):
        r = catalog.get(name)
        x, rpm = loader.read_raw(DATA / r.filename, r)
        return x, r.native_fs, rpm if rpm is not None else catalog.NOMINAL_RPM[r.load_hp]

    m = BearingMonitor().fit_baseline([raw("Normal_0"), raw("Normal_1"), raw("Normal_2")])
    for name, expected in [("IR014_3", "IR"), ("IR021_3", "IR"), ("OR021@6_3", "OR"), ("B014_3", "Undiagnosed")]:
        assert m.diagnose(*raw(name)).state == expected, name
    assert any("outside the baseline range" in w for w in m.diagnose(*raw("Normal_3")).warnings)
