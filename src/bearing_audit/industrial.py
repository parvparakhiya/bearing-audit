"""Industrial evaluation: judge the systems the way a plant would.

The leakage audit classified one-second windows into four classes. Here the
questions are:
- is the machine damaged? Detection rate, and above all false alarms.
- if so, which component? A system may answer Undiagnosed instead of
  guessing. Coverage is how often it names a type, precision how often that
  name is right.
- one decision per recording (its windows vote), not per window.
- what does it cost? Illustrative only, see config.OUTCOME_COST.

All of this was pre-registered (config.py, docs/industrial-preregistration.md) and runs
under protocol C only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone

from . import bearing, catalog, config, dsp, features, loader
from .evaluate import MODELS, forest
from .protocols import unseen_bearing

FAULTS = ("IR", "B", "OR")
STATES = ("Healthy", *FAULTS, "Undiagnosed")
OUTCOMES = ("correct", "false_alarm", "missed", "wrong_type", "undiagnosed")


class PhysicsDiagnoser(BaseEstimator):
    """Three-state physics diagnosis.

    Each threshold is the max of that quantity over the healthy training windows.
    Nothing else is learned.

    - alarm: any fault signature, or any scale-invariant broadband indicator
      (kurtosis, crest factor, SK max), above its healthy max. Scale-invariant,
      so a gain difference between recording sessions can't trigger an alarm.
    - named fault: only if its own signature is above threshold. Inner race also
      needs the shaft-rate sidebands. Ball uses the larger of 1x and 2x BSF. With
      several candidates, the largest margin over threshold wins.
    - otherwise Undiagnosed.
    """

    BROADBAND = ("kurtosis", "crest", "sk_max")
    ALARM_TRIGGERS = ("IR", "OR", "B", *BROADBAND)

    def _scores(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "IR": X["env_BPFI_h1"].to_numpy(),
                "IR_sb": X["env_BPFI_sb"].to_numpy(),
                "OR": X["env_BPFO_h1"].to_numpy(),
                "B": np.maximum(X["env_2xBSF_h1"].to_numpy(), X["env_BSF_h1"].to_numpy()),
                **{k: X[k].to_numpy() for k in self.BROADBAND},
            }
        )

    def fit(self, X: pd.DataFrame, y) -> PhysicsDiagnoser:
        healthy = np.asarray(y) == "Normal"
        if not healthy.any():
            raise ValueError("PhysicsDiagnoser needs healthy training windows to set its thresholds")
        self.thresholds_ = self._scores(X)[healthy].max()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        s, t = self._scores(X), self.thresholds_
        over = s.gt(t)
        # Alarm triggers as pre-registered: the three signatures plus the broadband
        # indicators. The sideband score only confirms an inner-race diagnosis and
        # never raises an alarm by itself (the first run got this wrong).
        alarm = over[list(self.ALARM_TRIGGERS)].any(axis=1).to_numpy()
        candidate = {"IR": over["IR"] & over["IR_sb"], "OR": over["OR"], "B": over["B"]}
        margin = pd.DataFrame({k: (s[k] - t[k]).where(candidate[k], -np.inf) for k in FAULTS})
        named = np.isfinite(margin.max(axis=1)).to_numpy()
        best = margin.idxmax(axis=1).to_numpy()
        return np.where(~alarm, "Healthy", np.where(named, best, "Undiagnosed")).astype(object)


class AbstainingForest(BaseEstimator):
    """The audit's forest, but allowed to say Undiagnosed when unsure."""

    def __init__(self, abstain_below: float = config.ABSTAIN_BELOW):
        self.abstain_below = abstain_below

    def fit(self, X: pd.DataFrame, y) -> AbstainingForest:
        self.model_ = clone(forest()).fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        p = self.model_.predict_proba(X)
        top = self.model_.classes_[np.argmax(p, axis=1)]
        out = np.where(top == "Normal", "Healthy", top).astype(object)
        out[p.max(axis=1) < self.abstain_below] = "Undiagnosed"
        return out


class ForcedChoice(BaseEstimator):
    """Wraps an audit model so its outputs use the state names used here."""

    def __init__(self, model: str = "forest"):
        self.model = model

    def fit(self, X: pd.DataFrame, y) -> ForcedChoice:
        est, self.cols_ = MODELS[self.model]
        self.est_ = clone(est).fit(X[self.cols_], y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        p = self.est_.predict(X[self.cols_])
        return np.where(p == "Normal", "Healthy", p).astype(object)


SYSTEMS = {
    "forest": (lambda: ForcedChoice("forest"), None),
    "rule": (lambda: ForcedChoice("rule_fixed_band"), None),
    "forest_abstain": (AbstainingForest, features.FEATURES),
    "physics_diagnoser": (PhysicsDiagnoser, features.DIAGNOSER_FEATURES),
}


def recording_decision(states: pd.Series) -> str:
    """One decision for a recording, from the states of its windows."""
    states = pd.Series(states)
    alarm = states != "Healthy"
    if alarm.mean() < config.ALARM_SHARE:
        return "Healthy"
    counts = states[alarm & states.isin(FAULTS)].value_counts()
    enough = len(counts) > 0 and counts.iloc[0] / alarm.sum() >= config.DIAGNOSIS_SHARE
    untied = len(counts) == 1 or (len(counts) > 1 and counts.iloc[0] > counts.iloc[1])
    return str(counts.index[0]) if enough and untied else "Undiagnosed"


def outcome(true_fault: str, decision: str) -> str:
    if true_fault == "Normal":
        return "correct" if decision == "Healthy" else "false_alarm"
    if decision == "Healthy":
        return "missed"
    if decision == "Undiagnosed":
        return "undiagnosed"
    return "correct" if decision == true_fault else "wrong_type"


@dataclass
class Metrics:
    false_alarm_rate: float
    detection_rate: float
    coverage: float
    diagnosis_precision: float
    cost_per_recording: float
    outcomes: dict[str, int]
    n_healthy: int
    n_faulty: int


def metrics(decisions: pd.DataFrame, cost: dict[str, float] | None = None) -> Metrics:
    """``decisions`` needs a ``fault`` (truth) and a ``decision`` column."""
    cost = cost or config.OUTCOME_COST
    healthy = decisions.fault.eq("Normal").to_numpy()
    dec = decisions.decision.to_numpy()
    alarm = dec != "Healthy"
    named = np.isin(dec, FAULTS)
    detected = ~healthy & alarm
    outs = [outcome(f, d) for f, d in zip(decisions.fault, dec, strict=True)]
    return Metrics(
        false_alarm_rate=float(alarm[healthy].mean()) if healthy.any() else float("nan"),
        detection_rate=float(alarm[~healthy].mean()) if (~healthy).any() else float("nan"),
        coverage=float(named[detected].mean()) if detected.any() else float("nan"),
        diagnosis_precision=float((dec[named] == decisions.fault.to_numpy()[named]).mean())
        if named.any()
        else float("nan"),
        cost_per_recording=float(np.mean([cost[o] for o in outs])),
        outcomes={o: outs.count(o) for o in OUTCOMES},
        n_healthy=int(healthy.sum()),
        n_faulty=int((~healthy).sum()),
    )


def gate(m: Metrics) -> dict:
    g = config.PLANT_GATE
    checks = {
        "false_alarm_rate": m.false_alarm_rate <= g["max_false_alarm_rate"],
        "detection_rate": m.detection_rate >= g["min_detection_rate"],
        "diagnosis_precision": m.diagnosis_precision >= g["min_diagnosis_precision"],  # NaN fails
        "coverage": m.coverage >= g["min_coverage"],
    }
    return {"checks": checks, "passed": all(checks.values())}


def window_states(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Out-of-fold window states of every system under protocol C.

    Also returns the physics diagnoser's thresholds per fold (envelope scores are
    log10 prominence; kurtosis, crest and sk_max are raw values).
    """
    y = df.fault.to_numpy()
    parts, thresholds = [], []
    for fold, (train, test) in enumerate(unseen_bearing(df)):
        part = df.iloc[test][["recording", "window", "bearing", "fault", "size_mils", "load_hp"]].assign(fold=fold)
        for name, (make, cols) in SYSTEMS.items():
            est = make()
            Xtr, Xte = (df.iloc[train], df.iloc[test]) if cols is None else (df.iloc[train][cols], df.iloc[test][cols])
            est.fit(Xtr, y[train])
            part[name] = est.predict(Xte)
            if name == "physics_diagnoser":
                thresholds.append({"fold": fold, **est.thresholds_.round(4).to_dict()})
        parts.append(part)
    return pd.concat(parts, ignore_index=True), {"physics_thresholds": thresholds}


def recording_decisions(windows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fold, rec), g in windows.groupby(["fold", "recording"], sort=True):
        row = {"fold": fold, "recording": rec, "bearing": g.bearing.iloc[0], "fault": g.fault.iloc[0]}
        for name in SYSTEMS:
            row[name] = recording_decision(g[name])
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(windows: pd.DataFrame, decisions: pd.DataFrame) -> dict:
    out = {}
    for name in SYSTEMS:
        rec = metrics(decisions.rename(columns={name: "decision"})[["fault", "decision"]])
        win = metrics(windows.rename(columns={name: "decision"})[["fault", "decision"]])
        sens = {}
        for c in config.COST_MISSED_SENSITIVITY:
            cost = {**config.OUTCOME_COST, "missed": c}
            sens[f"missed={c:g}"] = metrics(decisions.rename(columns={name: "decision"}), cost).cost_per_recording
        out[name] = {
            "recording_level": rec.__dict__,
            "window_level": {
                k: win.__dict__[k] for k in ("false_alarm_rate", "detection_rate", "coverage", "diagnosis_precision")
            },
            "cost_sensitivity": sens,
            "gate": gate(rec),
        }
    return out


def signature_extras(data_dir) -> pd.DataFrame:
    """Full-recording evidence behind the diagnoser's extra features, descriptive only:
    inner-race sidebands at BPFI +/- f_r, and ball lines at 1x BSF, 2x BSF and
    2x BSF - FTF (cage modulation).
    """
    geo = bearing.SKF_6205
    rows = []
    for r in catalog.CORE:
        if r.fault not in ("IR", "B"):
            continue
        s = loader.load(r.name, data_dir)
        x = dsp.lowpass(s.x, s.fs, config.BANDWIDTH_HZ)
        f, a = dsp.envelope_spectrum(x, s.fs, config.ENVELOPE_BAND_HZ)
        q, fr = geo.frequencies(s.shaft_hz), s.shaft_hz
        row: dict[str, str | float] = {"recording": r.name, "bearing": r.bearing, "fault": r.fault}
        if r.fault == "IR":
            row["BPFI_minus_fr"] = features.prominence(f, a, q["BPFI"] - fr, 0.5 * fr, tol=0.01)[1]
            row["BPFI_plus_fr"] = features.prominence(f, a, q["BPFI"] + fr, 0.5 * fr, tol=0.01)[1]
        else:
            row["BSF"] = features.prominence(f, a, q["BSF"], 0.5 * q["BSF"])[1]
            row["2xBSF"] = features.prominence(f, a, q["2xBSF"], 0.5 * q["BSF"])[1]
            row["2xBSF_minus_FTF"] = features.prominence(f, a, q["2xBSF"] - q["FTF"], q["FTF"], tol=0.01)[1]
        rows.append(row)
    return pd.DataFrame(rows)
