import numpy as np
import pandas as pd

from bearing_audit.rules import EnvelopeRule


def test_rule_learns_a_threshold_and_names_the_dominant_frequency():
    rng = np.random.default_rng(0)
    quiet = rng.uniform(0.0, 0.3, size=(30, 3))
    rows, labels = [quiet], ["Normal"] * 30
    for j, fault in enumerate(("IR", "OR", "B")):
        s = rng.uniform(0.0, 0.3, size=(30, 3))
        s[:, j] = rng.uniform(1.0, 2.0, size=30)
        rows.append(s)
        labels += [fault] * 30
    X = pd.DataFrame(np.vstack(rows), columns=["env_BPFI_h1", "env_BPFO_h1", "env_2xBSF_h1"])
    rule = EnvelopeRule().fit(X, labels)
    assert 0.3 <= rule.threshold_ <= 1.0
    assert (rule.predict(X) == np.array(labels)).mean() == 1.0
