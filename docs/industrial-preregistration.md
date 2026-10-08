# Pre-registration: industrial evaluation

**Written on 2026-09-25, before any of these systems was run on the CWRU
recordings,** together with their code and constants (`config.py`, section
"industrial evaluation").

## Why this evaluation exists

The leakage audit scored one-second windows with 4-class macro-F1. A plant judges condition
monitoring differently: first *is the machine damaged* (with very few false
alarms), then *which component*, per measurement, and at what cost. A review
of the audit raised four gaps, and this evaluation addresses three of them
(the fourth, test coverage, is engineering and has no results):

1. Wrong decision unit and metric → recording-level decisions; detection
   rate, false-alarm rate, coverage, diagnosis precision, cost.
2. No "I don't know" → a fifth state, **Undiagnosed**: damage detected,
   component unclear.
3. Unused physics → inner-race sidebands at BPFI ± shaft speed, ball evidence
   at 1 × BSF as well as 2 × BSF, and envelope order spectra.

## What was already known when this was designed

This is not a blind design. Before this was written, the following were seen:

- all leakage-audit results (README, `reports/results/`);
- in the full-length recordings, BPFI ± shaft-speed sidebands ≥ 5× the local
  level in 12 of 12 inner-race recordings;
- a line ≥ 10× at 1 × BSF in 2 of 12 ball recordings; no cage-frequency
  sidebands around 2 × BSF;
- majority votes of the audit's window predictions per recording (forest 0.81,
  rule 0.69 accuracy), and the rule's window-level false alarms on healthy
  windows (17 %).

None of the new systems (`physics_diagnoser`, `forest_abstain`) and no
recording-level rule was run on real data before this was written.

## Design

**Protocol C only** (unseen bearing and unseen load, 12 folds), the same
folds as the audit.

### Systems compared

| name | window output | what is learned on the training fold |
|---|---|---|
| `forest` | the audit forest's forced 4-class choice | the forest |
| `rule` | the audit's fixed-band physics rule | one threshold |
| `forest_abstain` | the audit forest, but "Undiagnosed" if its top-class probability < 0.5 | the forest |
| `physics_diagnoser` | three-state physics diagnoser (below) | per-feature maxima over healthy training windows |

### `physics_diagnoser` (per window)

Every threshold is the highest value that quantity reached on any healthy
training window:

- **alarm** if any of: BPFI, BPFO or ball-evidence prominence; kurtosis;
  crest factor; spectral-kurtosis maximum exceeds its threshold. The
  broadband indicators are scale-invariant, so a sensor-gain difference
  between recording sessions cannot cause an alarm.
- **named type** if its signature exceeds its threshold. Inner race also
  requires the mean BPFI ± shaft-speed sideband prominence to exceed its
  threshold. Ball evidence = max(1 × BSF, 2 × BSF). If several types
  qualify, the largest margin over threshold wins.
- otherwise **Undiagnosed**.

### Recording decision (all systems, same rule)

Alarm if ≥ 50 % of its windows alarm. A named type if one type holds ≥ 50 %
of the alarm windows and is not tied for first place; otherwise Undiagnosed.

### Metrics (recording level)

Each healthy recording is tested in the 3 folds of its load, as in the audit.

- false-alarm rate = healthy recordings in alarm / healthy recordings
- detection rate = faulty recordings in alarm / faulty recordings
- coverage = detected faulty recordings given a named type / detected faulty recordings
- diagnosis precision = recordings given a named type that is correct / all recordings given a named type
- cost per recording, with costs per outcome: correct 0, false alarm 1,
  undiagnosed 1, wrong type 3, missed 10 (in units of one manual inspection;
  illustrative). Also reported with missed = 5 and 20.

### Plant-level gate

A system ships only if all four hold:

| criterion | threshold |
|---|---|
| false-alarm rate | 0 (no healthy recording in alarm) |
| detection rate | ≥ 0.90 |
| diagnosis precision | ≥ 0.95 |
| coverage | ≥ 0.50 |

### Known limits of this design (stated in advance)

- There is one healthy bearing. The false-alarm rate is measured on it at an
  unseen load, not on an unseen healthy machine.
- 12 healthy test instances (4 recordings × 3 folds) and 36 faulty ones. A
  false-alarm rate of 0 means "0 of 12", not "never".
- The cost ratios are an assumption, not data.

---

## Addendum: deviation found after run 1 (added 2026-09-25, after the run)

The design above is the original pre-registration: no rule, threshold or
criterion in it has changed.

**What happened.** In run 1, `PhysicsDiagnoser` raised an alarm whenever
*any* of its scores exceeded its healthy maximum, and that included the
inner-race sideband score. The design above lists the sidebands only as
confirmation for naming an inner-race fault, not as an alarm trigger. On
Normal_0 the sideband score exceeded its threshold in 60 % of windows, so
run 1 gave the physics diagnoser a false-alarm rate of 0.50.

### What was done

1. Run 1's results were kept, with every number unchanged.
2. The code was fixed to match this document. No threshold, constant or
   criterion changed.
3. A regression test was added: the sideband score alone must never alarm.
4. The evaluation was rerun.

**Result after the fix.** The physics diagnoser's false-alarm rate is 0.25.
All 3 remaining false alarms are Normal_3 (3 HP), the operating condition
outside the training baseline. Run 1 is kept in `reports/results/industrial_run1/`.
