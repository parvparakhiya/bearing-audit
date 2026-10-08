"""Three ways to split the CWRU windows into train and test.

The core set has 10 physical bearings (1 healthy + 3 fault types x 3 sizes),
each recorded at 4 loads, each recording cut into ~10 windows. A split is only
as honest as the largest unit it keeps apart:

    window < recording < bearing

A  Random windows, 5 folds. Windows from the same recording end up on both
   sides, so this tests whether the model recognises a recording it has seen.
   Most of the "99 %" CWRU results use this kind of split.

B  Leave one load out. No recording is shared, but every bearing is (at
   another load). Tests recognising a known bearing at a new speed.

C  Unseen bearing, unseen load. For each fault size s and load l: train on the
   other two sizes at the other three loads, test on size s at load l. The
   test bearings never appear in training, at any load. Only fault type
   (4 classes) makes sense here, since a held-out size is never trained on.

Limitation: CWRU has a single healthy bearing, so Normal is only held out by
load and never tested on a new bearing.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from . import config

Split = tuple[np.ndarray, np.ndarray]


def window_random(df: pd.DataFrame, label: str, n_splits: int = 5) -> Iterator[Split]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
    yield from skf.split(np.zeros(len(df)), df[label])


def leave_one_load_out(df: pd.DataFrame, label: str | None = None) -> Iterator[Split]:
    for load in sorted(df.load_hp.unique()):
        test = df.load_hp.to_numpy() == load
        yield np.flatnonzero(~test), np.flatnonzero(test)


def unseen_bearing(df: pd.DataFrame, label: str | None = None) -> Iterator[Split]:
    normal = df.fault.to_numpy() == "Normal"
    size = df.size_mils.to_numpy()
    load = df.load_hp.to_numpy()
    for s in sorted(np.unique(size[~normal])):
        for ld in sorted(np.unique(load)):
            test = (load == ld) & (normal | (size == s))
            train = (load != ld) & (normal | (size != s))
            yield np.flatnonzero(train), np.flatnonzero(test)


PROTOCOLS: dict[str, Callable[..., Iterator[Split]]] = {
    "A_window_random": window_random,
    "B_leave_one_load_out": leave_one_load_out,
    "C_unseen_bearing": unseen_bearing,
}
