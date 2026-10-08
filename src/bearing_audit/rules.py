"""Physics rule, used as the baseline the ML model has to beat.

For each window, check how much the envelope spectrum stands out at BPFI, BPFO
and 2xBSF (at the recording's own shaft speed). If nothing stands out more
than a threshold the window is Normal, otherwise it's the fault whose line
stands out most.

Only the threshold is learned, on the training data. The rest is kinematics,
so the rule can't memorise a recording.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score

FAULT_OF = {"BPFI": "IR", "BPFO": "OR", "2xBSF": "B"}


class EnvelopeRule(ClassifierMixin, BaseEstimator):
    """``columns``: the log10-prominence features for BPFI, BPFO and 2xBSF."""

    def __init__(self, columns: tuple[str, ...] = ("env_BPFI_h1", "env_BPFO_h1", "env_2xBSF_h1")):
        self.columns = columns

    def _scores(self, X: pd.DataFrame) -> np.ndarray:
        return X[list(self.columns)].to_numpy(dtype=float)

    def _decide(self, s: np.ndarray, threshold: float) -> np.ndarray:
        faults = np.array([FAULT_OF[c.split("_")[1]] for c in self.columns])
        out = faults[np.argmax(s, axis=1)].astype(object)
        out[s.max(axis=1) < threshold] = "Normal"
        return out

    def fit(self, X: pd.DataFrame, y) -> EnvelopeRule:
        s, y = self._scores(X), np.asarray(y)
        self.classes_ = np.unique(y)
        grid = np.linspace(0.0, 2.0, 201)  # log10 prominence, from 1x to 100x
        f1 = [f1_score(y, self._decide(s, t), average="macro", zero_division=0) for t in grid]
        self.threshold_ = float(grid[int(np.argmax(f1))])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._decide(self._scores(X), self.threshold_)
