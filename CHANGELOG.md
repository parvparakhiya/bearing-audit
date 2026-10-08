# Changelog

All notable changes. Versions follow [semantic versioning](https://semver.org/).

## [1.0.0] — 2026-10-08

First release.

### Leakage audit
- Loader that picks channels by record number and resamples the 48 kHz
  baseline to 12 kHz with an anti-alias filter.
- Bearing kinematics, envelope analysis and spectral kurtosis on top of
  numpy/scipy.
- Window features, a physics rule and a random forest (19 features).
- Split protocols A / B / C and a ship gate fixed before the first run. No
  model passes.
- Data checks (checksums, sampling-rate proof, acquisition confound) and a
  physics check on every faulty recording.
- Post-hoc ablation, seed sweep and bandwidth sensitivity.

### Industrial evaluation
- Recording-level decisions with false-alarm rate, detection rate, coverage,
  diagnosis precision and an illustrative cost model, pre-registered in
  `docs/industrial-preregistration.md`.
- Three-state physics diagnoser (Healthy / named fault / Undiagnosed) using
  inner-race sidebands and 1 × BSF ball evidence, and a forest that may
  abstain. No system passes the plant-level gate.
- Envelope order spectra (figure 8).

### Tooling
- `BearingMonitor` public API: healthy baseline stored as plain JSON, strict
  input checks, any bearing geometry, warning outside the baseline's rpm range.
- Two notebooks: a ten-minute presentation, and the full pipeline with 31
  consistency checks.
- 90 tests, mypy, ruff, pre-commit hooks, Dockerfile, pinned versions and a
  checksum-verified data fetch.
- CI: tests on Python 3.10-3.12, a regression job that reruns everything on
  the real data and fails if any result changes, both notebooks, and a Docker
  build.
