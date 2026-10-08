# Data provenance

## Source

Case Western Reserve University (CWRU) Bearing Data Center: the 12k drive-end
bearing fault data and the normal baseline data. The recordings aren't
redistributed in this repository.

## Integrity

`manifest.csv` lists all 56 files used, with SHA-256, record number, sample
count, rpm and rpm source. `bearing-audit fetch` accepts a file only if its
checksum matches.

## Where the files came from

- I downloaded five files directly from the CWRU site: `Normal_0`, `Normal_3`,
  `IR007_0`, `IR021_0` and `OR007@6_0`.
- The other 51 come from the public mirror
  [XiongMeijing/CWRU-1](https://github.com/XiongMeijing/CWRU-1).
- The five files from the CWRU site are byte-identical to the mirror's copies:

  | file | SHA-256 (first 16) |
  |---|---|
  | Normal_0.mat | `16bf48babcf1c7ac` |
  | Normal_3.mat | `88a5990cb541320e` |
  | IR007_0.mat | `f80b0ea04fd06b37` |
  | IR021_0.mat | `9f723d6d9d2eba71` |
  | OR007@6_0.mat | `35a095307d097147` |

## What the files contain

- Every file contains the record number the CWRU catalogue gives it: its
  variables are named after it, e.g. `X105_DE_time` in `IR007_0.mat` (record
  105). `Normal_2.mat` (record 99) also holds record 98, byte-identical to
  `Normal_1.mat`, so the loader picks channels by record number.
- `Normal_1` and `Normal_2` store no rpm. The catalogue speeds (1772 and 1750
  rpm) are used instead, and the spectra put the shaft line within 0.03 Hz of
  both (`reports/results/sampling_rate_check.csv`).
- The 0.028" files are excluded:
  - CWRU used NTN bearings for that size, not the SKF 6205.
  - They store no rpm.
  - Their internal record numbers (048–051 for ball, 056–059 for inner race)
    don't match the catalogue numbers (3001–3008).
