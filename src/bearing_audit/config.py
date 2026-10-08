"""Analysis constants, all fixed before any model was trained.

None of them were tuned on classification scores; each one has its physical
or data-quality reason next to it.
"""

from __future__ import annotations

# Analysis bandwidth. The 12 kHz fault files roll off hard above ~5.2 kHz
# (the recorder's anti-alias filter) while the healthy files, resampled from
# 48 kHz, stay flat up to 6 kHz. Above 5 kHz we'd be measuring the recorder,
# not the bearing, so every signal is low-passed here and no feature looks higher.
BANDWIDTH_HZ = 5_000.0

# Fixed demodulation band: the 2-4 kHz resonance hump in the faulty spectra,
# capped just below the bandwidth. Picked from averaged spectra, not from
# classification scores. The SK-selected band is evaluated as well.
ENVELOPE_BAND_HZ = (2_000.0, 4_900.0)

# SK band search limits. Below 500 Hz it's mostly shaft harmonics.
SK_FMIN_HZ = 500.0

# 1 s windows give 1 Hz resolution in the envelope spectrum, enough to separate
# BPFO (~107 Hz), 2xBSF (~141 Hz) and BPFI (~162 Hz). No overlap: overlapping
# windows share samples and would inflate the random split even more.
WINDOW_S = 1.0

# Search tolerance around a predicted line. Rolling elements slip, so real
# peaks sit 1-2 % away from the kinematic value.
PEAK_TOL = 0.03

# harmonics used in the envelope features
N_HARMONICS = 3

# A line counts as present if its envelope peak is at least this many times
# the local median.
PRESENT_PROMINENCE = 10.0

# Ship gate, written down before the first audit run. A model ships only if,
# under protocol C (unseen bearing):
GATE_MIN_MACRO_F1 = 0.90  # mean macro-F1 over the folds >= this, and
GATE_MIN_CLASS_RECALL = 0.80  # every class's pooled recall >= this

RANDOM_STATE = 0

# --- industrial evaluation ---
# Pre-registered on 2026-09-25, before any of these systems ran on the recordings,
# but designed after seeing the leakage-audit results (see docs/industrial-preregistration.md).

# Forest with an "I don't know": below this top-class probability the window
# is Undiagnosed instead of a forced guess.
ABSTAIN_BELOW = 0.5

# A recording alarms if at least this share of its windows alarm. It gets a
# fault type if one type holds at least this share of the alarm windows
# (a tie at the top means Undiagnosed).
ALARM_SHARE = 0.5
DIAGNOSIS_SHARE = 0.5

# Illustrative cost per recording, in units of one manual inspection. A missed
# fault counts as 10 inspections; a wrong diagnosis sends someone to the
# wrong component.
OUTCOME_COST = {"correct": 0.0, "false_alarm": 1.0, "undiagnosed": 1.0, "wrong_type": 3.0, "missed": 10.0}
COST_MISSED_SENSITIVITY = (5.0, 10.0, 20.0)

# plant-level gate (protocol C, per recording). All four must hold:
PLANT_GATE = {
    "max_false_alarm_rate": 0.0,  # no healthy recording alarms
    "min_detection_rate": 0.90,  # >= 90 % of faulty recordings alarm
    "min_diagnosis_precision": 0.95,  # a named fault type is right >= 95 % of the time
    "min_coverage": 0.50,  # >= half of the detected faults get a name
}
