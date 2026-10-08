"""Tests for the split protocols, the core of the audit, on a table shaped
exactly like the real one.
"""

import numpy as np
import pandas as pd
import pytest

from bearing_audit import catalog, protocols


@pytest.fixture
def table() -> pd.DataFrame:
    rows = []
    for r in catalog.CORE:
        for w in range(5 if r.name == "Normal_0" else 10):
            rows.append(
                {
                    "recording": r.name,
                    "window": w,
                    "bearing": r.bearing,
                    "fault": r.fault,
                    "label10": r.label10,
                    "size_mils": r.size_mils,
                    "load_hp": r.load_hp,
                }
            )
    return pd.DataFrame(rows)


def _names(df, idx, col):
    return set(df.iloc[idx][col])


def test_window_random_tests_every_window_once_and_mixes_recordings(table):
    seen = np.concatenate([te for _, te in protocols.window_random(table, "fault")])
    assert sorted(seen) == list(range(len(table)))
    tr, te = next(protocols.window_random(table, "fault"))
    assert _names(table, tr, "recording") & _names(table, te, "recording")  # the leak, by design


def test_leave_one_load_out_never_shares_a_recording(table):
    for tr, te in protocols.leave_one_load_out(table):
        assert not _names(table, tr, "recording") & _names(table, te, "recording")
        assert len(_names(table, te, "load_hp")) == 1


def test_unseen_bearing_never_shares_a_faulted_bearing_or_a_load(table):
    folds = list(protocols.unseen_bearing(table))
    assert len(folds) == 12
    tested = []
    for tr, te in folds:
        faulted_tr = set(table.iloc[tr].query("fault != 'Normal'").bearing)
        faulted_te = set(table.iloc[te].query("fault != 'Normal'").bearing)
        assert not faulted_tr & faulted_te
        assert not _names(table, tr, "load_hp") & _names(table, te, "load_hp")
        assert set(table.iloc[te].fault) == {"Normal", "IR", "B", "OR"}
        tested += list(table.iloc[te].query("fault != 'Normal'").recording.unique())
    faulted = sorted(r.name for r in catalog.CORE if r.fault != "Normal")
    assert sorted(tested) == faulted  # each faulted recording is tested exactly once
