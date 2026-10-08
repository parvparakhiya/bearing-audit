"""Evaluate every model under every split protocol, then apply the ship gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, recall_score

from . import config, features
from .protocols import PROTOCOLS
from .rules import EnvelopeRule

TASKS = {"type4": "fault", "type_size10": "label10"}
TYPE_ORDER = ["Normal", "IR", "B", "OR"]


def forest() -> RandomForestClassifier:
    # deliberately untuned: without a hyper-parameter search nothing can leak through one
    return RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )


MODELS = {
    "forest": (forest(), features.FEATURES),
    "rule_fixed_band": (
        EnvelopeRule(("env_BPFI_h1", "env_BPFO_h1", "env_2xBSF_h1")),
        ["env_BPFI_h1", "env_BPFO_h1", "env_2xBSF_h1"],
    ),
    "rule_sk_band": (EnvelopeRule(tuple(features.SK_RULE_FEATURES)), features.SK_RULE_FEATURES),
}
RULES = {"rule_fixed_band", "rule_sk_band"}


@dataclass
class Result:
    task: str
    protocol: str
    model: str
    fold_macro_f1: list[float]
    mean_macro_f1: float
    sd_macro_f1: float
    pooled_accuracy: float
    recall: dict[str, float]
    labels: list[str]
    confusion: list[list[int]]
    n_folds: int
    extra: dict = field(default_factory=dict)


def evaluate(df: pd.DataFrame, task: str, protocol: str, model: str) -> Result:
    label = TASKS[task]
    est, cols = MODELS[model]
    y = df[label].to_numpy()
    X = df[cols]
    fold_f1: list[float] = []
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    extra: dict[str, list[float]] = {"thresholds": []}
    for train, test in PROTOCOLS[protocol](df, label):
        m = clone(est).fit(X.iloc[train], y[train])
        p = m.predict(X.iloc[test])
        fold_f1.append(float(f1_score(y[test], p, labels=np.unique(y[test]), average="macro", zero_division=0)))
        y_true.append(y[test])
        y_pred.append(p)
        if model in RULES:
            extra["thresholds"].append(m.threshold_)
    yt, yp = np.concatenate(y_true), np.concatenate(y_pred)
    labels = TYPE_ORDER if task == "type4" else sorted(np.unique(y), key=lambda s: (s != "Normal", s))
    rec = recall_score(yt, yp, labels=labels, average=None, zero_division=0)
    return Result(
        task=task,
        protocol=protocol,
        model=model,
        fold_macro_f1=fold_f1,
        mean_macro_f1=float(np.mean(fold_f1)),
        sd_macro_f1=float(np.std(fold_f1)),
        pooled_accuracy=float(np.mean(yt == yp)),
        recall={lab: float(r) for lab, r in zip(labels, rec, strict=True)},
        labels=labels,
        confusion=confusion_matrix(yt, yp, labels=labels).tolist(),
        n_folds=len(fold_f1),
        extra=extra if model in RULES else {},
    )


def run_all(df: pd.DataFrame) -> list[Result]:
    out = []
    for task in TASKS:
        for protocol in PROTOCOLS:
            # a held-out size never appears in training, so it can't be predicted
            if task == "type_size10" and protocol == "C_unseen_bearing":
                continue
            for model in MODELS:
                if model in RULES and task != "type4":
                    continue  # kinematics give the fault type, not its size
                out.append(evaluate(df, task, protocol, model))
    return out


def predictions(
    df: pd.DataFrame, protocol: str, model: str, cols: list[str] | None = None, seed: int | None = None
) -> pd.DataFrame:
    """Out-of-fold predictions for the 4-class task, one row per test window."""
    est, default_cols = MODELS[model]
    cols = cols or default_cols
    if seed is not None:
        est = clone(est).set_params(random_state=seed)
    y = df.fault.to_numpy()
    parts = []
    for fold, (train, test) in enumerate(PROTOCOLS[protocol](df, "fault")):
        m = clone(est).fit(df.iloc[train][cols], y[train])
        parts.append(
            df.iloc[test][["recording", "window", "bearing", "fault", "size_mils", "load_hp"]].assign(
                pred=m.predict(df.iloc[test][cols]), fold=fold
            )
        )
    return pd.concat(parts, ignore_index=True)


def fold_macro_f1(p: pd.DataFrame) -> list[float]:
    return [
        float(f1_score(g.fault, g.pred, labels=np.unique(g.fault), average="macro", zero_division=0))
        for _, g in p.groupby("fold")
    ]


def protocol_c_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Out-of-fold predictions of every model under protocol C."""
    out = None
    for model in MODELS:
        p = predictions(df, "C_unseen_bearing", model).rename(columns={"pred": model})
        out = p if out is None else out.merge(p, on=[c for c in p.columns if c != model])
    return out


def per_bearing(preds: pd.DataFrame) -> pd.DataFrame:
    """Fraction of each bearing's test windows classified correctly, per model."""
    return pd.DataFrame({m: preds[m].eq(preds.fault).groupby(preds.bearing).mean() for m in MODELS})


# --- post-hoc analyses (not pre-registered) ---
# Added after the first audit run to explain the result and check how robust
# it is. None of them feeds the gate.

ABLATION_GROUPS = {
    "all features": features.FEATURES,
    "time + spectral only": features.TIME_FEATURES + features.SPECTRAL_FEATURES,
    "envelope only": features.ENVELOPE_FEATURES,
}


def ablation(df: pd.DataFrame) -> list[dict]:
    """Which feature groups generalise to unseen bearings, and which only
    identify the bearing.

    In the core set each 10-class label is one physical bearing, so 10-class
    accuracy across loads (protocol B) shows how well a group recognises a
    bearing it has already seen.
    """
    rows = []
    y10 = df.label10.to_numpy()
    for name, cols in ABLATION_GROUPS.items():
        p = predictions(df, "C_unseen_bearing", "forest", cols)
        hits = []
        for train, test in PROTOCOLS["B_leave_one_load_out"](df):
            m = clone(forest()).fit(df.iloc[train][cols], y10[train])
            hits.append(m.predict(df.iloc[test][cols]) == y10[test])
        rows.append(
            {
                "features": name,
                "protocol_C_mean_macro_f1": float(np.mean(fold_macro_f1(p))),
                "protocol_C_per_bearing": p.pred.eq(p.fault).groupby(p.bearing).mean().to_dict(),
                "bearing_identity_accuracy_protocol_B": float(np.concatenate(hits).mean()),
            }
        )
    return rows


SEEDS = tuple(range(10))


def seed_sweep(df: pd.DataFrame, seeds: tuple[int, ...] = SEEDS) -> dict:
    """Forest score under protocol C for ten seeds.

    The gate uses seed 0; this shows how much of a single number is seed noise.
    """
    scores = [float(np.mean(fold_macro_f1(predictions(df, "C_unseen_bearing", "forest", seed=s)))) for s in seeds]
    return {
        "seeds": list(seeds),
        "mean_macro_f1": scores,
        "min": min(scores),
        "max": max(scores),
        "mean": float(np.mean(scores)),
    }


def bandwidth_sensitivity(data_dir, bandwidth: float = 4_700.0) -> dict:
    """Protocol C again with a stricter 4.7 kHz low-pass.

    The 5 kHz filter still lets part of 5.0-5.25 kHz through; at 4.7 kHz everything
    above 5 kHz is more than 40 dB down. If the scores barely move, the leftover
    recorder difference in that band isn't what drives them.
    """
    from . import catalog

    df = features.build_table(catalog.CORE, data_dir, bandwidth=bandwidth)
    return {
        "bandwidth_hz": bandwidth,
        **{m: float(np.mean(fold_macro_f1(predictions(df, "C_unseen_bearing", m)))) for m in MODELS},
    }


def gate(r: Result) -> dict:
    worst = min(r.recall, key=lambda k: r.recall[k])
    passed = r.mean_macro_f1 >= config.GATE_MIN_MACRO_F1 and r.recall[worst] >= config.GATE_MIN_CLASS_RECALL
    return {
        "model": r.model,
        "passed": bool(passed),
        "mean_macro_f1": r.mean_macro_f1,
        "min_macro_f1": config.GATE_MIN_MACRO_F1,
        "worst_class": worst,
        "worst_recall": r.recall[worst],
        "min_recall": config.GATE_MIN_CLASS_RECALL,
    }


def gates(results: list[Result]) -> list[dict]:
    return [gate(r) for r in results if r.task == "type4" and r.protocol == "C_unseen_bearing"]


def to_records(results: list[Result]) -> list[dict]:
    return [asdict(r) for r in results]
