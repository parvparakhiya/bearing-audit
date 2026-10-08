"""Industrial-evaluation decision logic on hand-built feature rows with known answers."""

import numpy as np
import pandas as pd
import pytest

from bearing_audit import config, industrial
from bearing_audit.industrial import AbstainingForest, PhysicsDiagnoser

from .conftest import synthetic_table

COLS = ["env_BPFI_h1", "env_BPFI_sb", "env_BPFO_h1", "env_2xBSF_h1", "env_BSF_h1", "kurtosis", "crest", "sk_max"]


def _rows(*rows):
    return pd.DataFrame(rows, columns=COLS)


@pytest.fixture
def fitted():
    # healthy training windows: every value stays at or below these
    healthy = _rows([0.3, 0.3, 0.3, 0.3, 0.3, 3.2, 4.0, 0.5], [0.2, 0.1, 0.2, 0.25, 0.2, 3.0, 3.8, 0.4])
    return PhysicsDiagnoser().fit(healthy, ["Normal", "Normal"])


def test_thresholds_are_the_healthy_maxima(fitted):
    t = fitted.thresholds_
    assert (t["IR"], t["IR_sb"], t["OR"], t["B"], t["kurtosis"]) == (0.3, 0.3, 0.3, 0.3, 3.2)


def test_quiet_window_is_healthy(fitted):
    assert fitted.predict(_rows([0.1, 0.1, 0.1, 0.1, 0.1, 3.0, 3.5, 0.3]))[0] == "Healthy"


def test_inner_race_needs_its_sidebands(fitted):
    with_sb = _rows([1.5, 0.9, 0.1, 0.1, 0.1, 3.0, 3.5, 0.3])
    without_sb = _rows([1.5, 0.2, 0.1, 0.1, 0.1, 3.0, 3.5, 0.3])
    assert fitted.predict(with_sb)[0] == "IR"
    assert fitted.predict(without_sb)[0] == "Undiagnosed"  # alarm, but not a confirmed inner-race defect


def test_outer_race_and_ball_via_either_bsf_line(fitted):
    assert fitted.predict(_rows([0.1, 0.1, 1.2, 0.1, 0.1, 3.0, 3.5, 0.3]))[0] == "OR"
    assert fitted.predict(_rows([0.1, 0.1, 0.1, 0.1, 0.9, 3.0, 3.5, 0.3]))[0] == "B"  # 1 x BSF only
    assert fitted.predict(_rows([0.1, 0.1, 0.1, 0.9, 0.1, 3.0, 3.5, 0.3]))[0] == "B"  # 2 x BSF only


def test_sidebands_alone_never_raise_an_alarm(fitted):
    """Regression test for the first run's bug: sidebands confirm an inner-race diagnosis,
    they don't trigger an alarm (see docs/industrial-preregistration.md).
    """
    assert fitted.predict(_rows([0.1, 2.0, 0.1, 0.1, 0.1, 3.0, 3.5, 0.3]))[0] == "Healthy"


def test_broadband_alarm_without_signature_is_undiagnosed(fitted):
    assert fitted.predict(_rows([0.1, 0.1, 0.1, 0.1, 0.1, 6.0, 3.5, 0.3]))[0] == "Undiagnosed"


def test_largest_margin_wins_between_candidates(fitted):
    assert fitted.predict(_rows([0.5, 0.9, 1.4, 0.1, 0.1, 3.0, 3.5, 0.3]))[0] == "OR"


def test_diagnoser_refuses_to_fit_without_healthy_data():
    with pytest.raises(ValueError):
        PhysicsDiagnoser().fit(_rows([1, 1, 1, 1, 1, 3, 3, 1]), ["IR"])


def test_abstaining_forest_says_undiagnosed_when_classes_are_indistinguishable():
    X = pd.DataFrame(np.zeros((30, 2)), columns=["a", "b"])  # three classes, identical features
    y = ["Normal"] * 10 + ["IR"] * 10 + ["OR"] * 10
    assert set(AbstainingForest().fit(X, y).predict(X)) == {"Undiagnosed"}


def test_abstaining_forest_keeps_confident_answers():
    X = pd.DataFrame({"a": [0.0] * 10 + [5.0] * 10})
    y = ["Normal"] * 10 + ["IR"] * 10
    pred = AbstainingForest().fit(X, y).predict(X)
    assert list(pred) == ["Healthy"] * 10 + ["IR"] * 10


@pytest.mark.parametrize(
    ("states", "expected"),
    [
        (["Healthy"] * 6 + ["IR"] * 4, "Healthy"),  # 40 % in alarm, below the 50 % needed
        (["Healthy"] * 5 + ["IR"] * 5, "IR"),  # exactly 50 % alarm counts
        (["IR"] * 6 + ["Undiagnosed"] * 4, "IR"),
        (["IR"] * 4 + ["OR"] * 4 + ["Healthy"] * 2, "Undiagnosed"),  # tie at the top
        (["IR"] * 3 + ["Undiagnosed"] * 7, "Undiagnosed"),  # no type holds half the alarm windows
    ],
)
def test_recording_decision(states, expected):
    assert industrial.recording_decision(pd.Series(states)) == expected


def test_outcomes():
    o = industrial.outcome
    assert (o("Normal", "Healthy"), o("Normal", "IR"), o("Normal", "Undiagnosed")) == (
        "correct",
        "false_alarm",
        "false_alarm",
    )
    assert (o("IR", "Healthy"), o("IR", "Undiagnosed"), o("IR", "OR"), o("IR", "IR")) == (
        "missed",
        "undiagnosed",
        "wrong_type",
        "correct",
    )


def test_metrics_and_gate():
    d = pd.DataFrame(
        {
            "fault": ["Normal", "Normal", "IR", "IR", "OR", "B"],
            "decision": ["Healthy", "Healthy", "IR", "Undiagnosed", "OR", "Healthy"],
        }
    )
    m = industrial.metrics(d)
    assert m.false_alarm_rate == 0.0
    assert m.detection_rate == pytest.approx(3 / 4)
    assert m.coverage == pytest.approx(2 / 3)
    assert m.diagnosis_precision == 1.0
    assert m.cost_per_recording == pytest.approx((1 + 10) / 6)  # one undiagnosed, one missed
    g = industrial.gate(m)
    assert g["checks"]["false_alarm_rate"] and not g["checks"]["detection_rate"]
    assert not g["passed"]


def test_gate_fails_when_nothing_is_named():
    d = pd.DataFrame({"fault": ["Normal", "IR"], "decision": ["Healthy", "Undiagnosed"]})
    m = industrial.metrics(d)
    assert np.isnan(m.diagnosis_precision)
    assert not industrial.gate(m)["passed"]


def test_gate_passes_a_perfect_system():
    d = pd.DataFrame({"fault": ["Normal", "IR", "OR", "B"], "decision": ["Healthy", "IR", "OR", "B"]})
    assert industrial.gate(industrial.metrics(d))["passed"]
    assert config.PLANT_GATE["max_false_alarm_rate"] == 0.0


def test_the_whole_industrial_pipeline_on_a_problem_with_a_known_answer(monkeypatch):
    from sklearn.ensemble import RandomForestClassifier

    from bearing_audit import evaluate, features

    monkeypatch.setitem(
        evaluate.MODELS, "forest", (RandomForestClassifier(n_estimators=25, random_state=0), features.FEATURES)
    )
    monkeypatch.setattr(industrial, "forest", lambda: RandomForestClassifier(n_estimators=25, random_state=0))
    df = synthetic_table()
    windows, extra = industrial.window_states(df)
    decisions = industrial.recording_decisions(windows)
    assert len(extra["physics_thresholds"]) == 12
    assert len(decisions) == 48  # 12 folds x (3 faulted + 1 healthy recording)
    summary = industrial.summarise(windows, decisions)
    for name in industrial.SYSTEMS:
        rec = summary[name]["recording_level"]
        assert rec["false_alarm_rate"] == 0.0 and rec["detection_rate"] == 1.0, name
        assert rec["diagnosis_precision"] == 1.0 and summary[name]["gate"]["passed"], name
