# Walkthrough: every step, in plain words

This file explains what the code does and why, in the order the data flows
through it.

## 1. The physical picture

A ball bearing is an inner ring on the shaft, an outer ring in the housing,
and balls rolling between them. If one ring has a small pit, every ball that
rolls over it makes a tiny impact. The impacts are regular. How often they
happen depends only on the geometry and on how fast the shaft turns.

The geometry gives four rates, as multiples of shaft speed `f_r`
(`bearing.py`):

| defect on | name | multiple of f_r (SKF 6205) |
|---|---|---|
| inner ring | BPFI | 5.415 |
| outer ring | BPFO | 3.585 |
| ball | 2 × BSF | 4.713 |
| cage | FTF | 0.398 |

The inner-race rate is higher than the outer-race rate because the inner ring
turns with the shaft and meets the balls more often. The two always add up to
the number of balls (a test checks this).

One trap here: the textbook ball formula (BSF) gives 2.357. A ball defect
touches *both* rings once per spin, so the impact rate is twice that, and
CWRU's table lists the doubled value. Mixing the two conventions puts you off
by 2×, and nothing in the plot will tell you.

## 2. Sampling rate, and why it's checked with physics

The sensor measures the vibration thousands of times per second. That's the
sampling rate. The file stores the numbers but not the rate. The healthy
files are 48 kHz and the fault files 12 kHz. Read 48 kHz data as 12 kHz and
every frequency is 4 times too low.

The check (`verify.sampling_rate_check`): the shaft peak must sit at rpm/60,
and it must *move* when the rpm changes. At 0 HP the wrong reading shows a
line at 29.96 Hz, right next to the 29.93 Hz shaft. It's really the 120 Hz
electrical line (plus the 4th shaft harmonic) divided by 4. So one file can
fool you. Four loads can't: the shaft slows to 28.75 Hz and the fake line
stays at 30.00. File duration is a second, independent check: at the right
rate every file is about 5 or 10 seconds long.

In general, a peak in the right place isn't proof on its own. A peak that
moves the way the physics says it should is.

## 3. Spectrum basics

- **FFT** turns a signal into its frequencies. The amplitude spectrum shows
  how strong each frequency is.
- **Resolution = 1 / duration.** A 1-second window resolves 1 Hz. That's why
  the windows are 1 s: BPFO (~107 Hz), 2×BSF (~141 Hz) and BPFI (~162 Hz)
  have to be separable.
- **Hann window.** The signal is multiplied by a smooth bump before the FFT,
  so the abrupt start and end of the window don't smear energy across all
  frequencies.
- **Nyquist.** At 12 kHz you can see up to 6 kHz. Anything faster folds back
  and appears as a fake lower frequency (aliasing). `resample_poly` filters
  that out before reducing the rate. `x[::4]` does not, and a test shows a
  9 kHz tone turning into a clean, fake 3 kHz line.

## 4. Envelope analysis: the core method

Each impact doesn't hum at the fault rate. It rings the housing like a bell,
at its own high pitch (a structural resonance, here 2–5 kHz). So the raw
spectrum shows the bell rather than the tapping.

What does carry the fault rate is how loud the ringing is over time: loud
right after each impact, then fading. That loudness curve is the envelope.

1. Band-pass 2–4.9 kHz to keep only the ringing band. `sosfiltfilt` runs the
   filter forwards and backwards, so the impacts don't shift in time.
2. Take the magnitude of the Hilbert transform. This gives the envelope, the
   instantaneous loudness.
3. Take the FFT of the envelope, which is the envelope spectrum. The tapping
   rate shows up as a peak, with harmonics at 2× and 3×.

`tests/test_dsp.py` builds a synthetic fault (impacts at 97 Hz ringing a
3 kHz resonance) and checks that the raw spectrum peaks near 3 kHz while the
envelope spectrum peaks at 97 Hz. Those two assertions test the whole method.

## 5. Choosing the band: spectral kurtosis

Kurtosis measures how "spiky" a signal is. Gaussian noise has kurtosis 3
(spectral kurtosis 0). Impacts make a band spiky. Spectral kurtosis computes
that spikiness frequency by frequency, and the spikiest band should be where
the impacts ring (`dsp.select_band`, a simplified kurtogram).

On this rig, the fixed 2–4.9 kHz band beat it on 35 of 52 recordings. Very
short analysis windows also find the sharp start of each impact spiky away
from the resonance. Both methods are kept, and both are reported.

## 6. The acquisition confound

The healthy files went through one anti-alias filter (48 kHz recorder, then
this code's resampling). The fault files went through another (the 12 kHz
recorder). Above ~5.2 kHz they look completely different, for reasons that
have nothing to do with bearings. One feature from that region separates
healthy from faulty perfectly.

The rule: low-pass everything at 5 kHz, and stop every spectral band at
5 kHz. The filter isn't a brick wall: it passes half the amplitude at 5.0 kHz,
2.4 % at 5.2 kHz and 0.02 % at 5.4 kHz. The recorders start to differ from
about 5.2 kHz. A test adds a tone at 5.3 kHz and at 5.6 kHz, louder than the
whole bearing signal, and checks that no feature moves by more than 1 %. A
post-hoc run with a stricter 4.7 kHz cutoff changes the results by at most
0.015.

## 7. Features (`features.py`)

Every feature comes from one 1-second window only.

- Time domain: RMS (overall level), kurtosis (spikiness), skewness,
  crest factor (peak ÷ RMS, which jumps with impacts), peak-to-peak.
- Spectral: share of energy in six bands up to 5 kHz, plus the maximum
  spectral kurtosis and where it occurs.
- Envelope: at BPFI, BPFO and 2×BSF, how many times taller the envelope
  peak is than the local median ("prominence"). Computed at harmonic 1 and as
  the mean of harmonics 1–3. Prominence is a ratio, so it doesn't depend on
  how loud the machine is overall.

## 8. The three split protocols: the point of the project

There are three nested units: window < recording < physical bearing. A split
is only as honest as the largest unit it keeps apart.

- A. Random windows. Windows from the same 10-second recording end up in
  both train and test. The model recognises the recording. Result: 1.00.
- B. Leave one load out. No recording is shared, but every bearing was
  seen at another load. The model recognises the bearing. Result: 0.99.
- C. Unseen bearing and unseen load. Train on two fault sizes at three
  loads, test on the third size at the fourth load. The test bearings were
  never seen. Result: 0.73 (0.64–0.74 across ten seeds).

The task under C is fault type (4 classes), because a held-out size can't be
predicted if it never appears in training.

## 9. The physics rule (`rules.py`)

Take the three prominences. If none is above a threshold, call the window
healthy. Otherwise pick the fault whose frequency stands out most. The only
thing learned is the threshold, per fold, on training data only. It can't
memorise a recording, which is why it barely changes between A, B and C.

## 10. Reading the result

- Where the fault signature is physically present, the rule is right on every
  window of every unseen bearing. The forest isn't.
- Where it's absent (ball faults, OR014), nothing trained here can be
  trusted. The forest's good ball recall comes from using "ball" as a
  catch-all. It puts all of OR014 there, and even some windows from
  bearings with a clear signature (IR014, OR021).
- Nothing passes the pre-registered gate, so nothing ships. That outcome is
  the finding, and it's reported as one.

## 11. Questions a reviewer will ask

**"Your forest gets 1.00, so why do you say it doesn't work?"**
Because 1.00 is under a split where test windows come from recordings the
model trained on. On a bearing it never saw, it's 0.73 (0.64–0.74 depending
on the random seed), with 0.47 recall on outer-race faults.

**"Why not just group by recording?"**
That's protocol B, and it gives 0.99. Each CWRU fault is one physical bearing
recorded at four loads, so a held-out recording is still a known bearing.

**"How do you know your envelope spectrum is right and not just plausible?"**
Three ways: a synthetic signal with a known answer; the measured peak lands
within about 0.3 % of the prediction (median); and it moves with rpm across
four loads.

**"How did you choose the band?"**
By looking at averaged spectra (the resonance hump), below the 5 kHz limit.
It was compared against a spectral-kurtosis band, and the fixed band won on
35 of 52 recordings. It wasn't tuned on classification scores.

**"Is the healthy class really tested?"**
Only across loads. CWRU has one healthy bearing, so healthy recall of 1.00 is
recognition. That's stated as a limitation.

**"Why a random forest and not a CNN?"**
The question here is the evaluation, not the model class. Published CNN
results on CWRU are typically near-perfect under random window splits, for
the same reason the forest is. What would matter is a CNN's protocol-C
score, which wasn't measured here.

**"What would you do next?"**
Run the same audit on the Paderborn dataset, which has several bearings per
fault type and real (not only seeded) damage, so per-bearing validation has
more than nine bearings to work with. After that, replay the recordings over
MQTT into `pdm-service`.

## 12. Industrial evaluation: judged the way a plant judges it

The audit asked "which of four classes is this 1-second window?". A plant
asks something else: is the machine damaged, and how often do you raise a
false alarm? Which component? Per measurement, and at what cost? This
evaluation answers those under protocol C. Its design and gate were written down first
(`docs/industrial-preregistration.md`).

There are three possible answers: Healthy, a named fault, or "Undiagnosed",
meaning damage detected but type unclear. Saying "I don't know" is better
than sending a technician to the wrong component.

Decisions are made per recording. A recording is in alarm if at least half of
its windows alarm. It gets a named type if one type holds at least half of
the alarm windows.

In the physics diagnoser, every threshold is the highest value that
quantity reached on any healthy training window. An inner-race diagnosis also
needs the sidebands at BPFI ± shaft speed. They exist because the inner race
carries the defect around through the load zone once per revolution, so its
impacts are modulated at shaft speed.

What it found:

- Detection is solved on this data. Both forests catch 36 of 36 faulty
  recordings with 0 of 12 false alarms, but on the only healthy bearing.
- Diagnosis isn't. The best precision is 0.74, so one named diagnosis in four
  is wrong.
- The physics diagnoser never names a wrong type where a signature exists
  (19 of 20 correct, 1 undiagnosed).
- It fails in two places:
  - B021, a ball fault, looks exactly like an inner-race fault, sidebands
    included.
  - Healthy Normal_3 at 3 HP lies outside the baseline learned at 0–2 HP.
    That's why plants keep a separate baseline for each operating condition.

Dividing frequency by shaft speed turns the x-axis into orders, and the fault
lines then sit at the same place at every speed (fig. 8). Industrial tools
display envelope spectra this way.

The first run had a bug: the sideband score could raise alarms on its own,
which the design doesn't allow. That run was kept as produced, the code was
fixed and tested, the evaluation was rerun, and both runs are reported.
