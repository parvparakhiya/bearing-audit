"""Evaluator and ship gate. All headline numbers and the verdict come from this
module, so it's tested on a synthetic table with a known answer.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from bearing_audit import config, evaluate, features

from .conftest import synthetic_table


@pytest.fixture
def table() -> pd.DataFrame:
    return synthetic_table()


@pytest.fixture(autouse=True)
def small_forest(monkeypatch):
    small = RandomForestClassifier(n_estimators=25, random_state=0)
    monkeypatch.setitem(evaluate.MODELS, "forest", (small, features.FEATURES))


def test_evaluate_scores_a_separable_problem_perfectly(table):
    for model in ("forest", "rule_fixed_band", "rule_sk_band"):
        r = evaluate.evaluate(table, "type4", "C_unseen_bearing", model)
        assert r.mean_macro_f1 == pytest.approx(1.0), model
        assert r.n_folds == 12


def test_confusion_counts_every_test_window(table):
    r = evaluate.evaluate(table, "type4", "C_unseen_bearing", "forest")
    n_test = sum(len(te) for _, te in evaluate.PROTOCOLS["C_unseen_bearing"](table, "fault"))
    assert np.array(r.confusion).sum() == n_test
    assert set(r.recall) == {"Normal", "IR", "B", "OR"}


def test_run_all_runs_exactly_the_meaningful_combinations(table):
    combos = {(r.task, r.protocol, r.model) for r in evaluate.run_all(table)}
    assert len(combos) == 11
    assert not any(t == "type_size10" and p == "C_unseen_bearing" for t, p, _ in combos)
    assert not any(t == "type_size10" and m.startswith("rule") for t, _, m in combos)


def _result(f1: float, worst: float) -> evaluate.Result:
    return evaluate.Result(
        task="type4",
        protocol="C_unseen_bearing",
        model="m",
        fold_macro_f1=[f1],
        mean_macro_f1=f1,
        sd_macro_f1=0.0,
        pooled_accuracy=f1,
        recall={"Normal": 1.0, "IR": 1.0, "B": worst, "OR": 1.0},
        labels=["Normal", "IR", "B", "OR"],
        confusion=[],
        n_folds=1,
    )


@pytest.mark.parametrize(
    ("f1", "worst", "passed"),
    [
        (0.95, 0.85, True),
        (0.89, 0.85, False),
        (0.95, 0.79, False),
        (config.GATE_MIN_MACRO_F1, config.GATE_MIN_CLASS_RECALL, True),
    ],
)
def test_gate_boundaries(f1, worst, passed):
    g = evaluate.gate(_result(f1, worst))
    assert g["passed"] is passed
    assert g["worst_class"] == "B"


def test_gates_only_judge_protocol_c(table):
    results = evaluate.run_all(table)
    judged = evaluate.gates(results)
    assert {g["model"] for g in judged} == {"forest", "rule_fixed_band", "rule_sk_band"}


def test_per_bearing_and_fold_f1_from_predictions(table):
    p = evaluate.protocol_c_predictions(table)
    assert set(evaluate.MODELS) <= set(p.columns)
    pb = evaluate.per_bearing(p)
    assert pb.shape == (10, 3) and (pb.to_numpy() == 1.0).all()
    single = p.rename(columns={"forest": "pred"})
    assert evaluate.fold_macro_f1(single) == pytest.approx([1.0] * 12)
