"""Check that the regenerated results match the committed ones.

Labels, predictions, decisions and every other piece of text must match
exactly. Numbers must match to a relative tolerance of 1e-9 (absolute 1e-12
near zero). Different processors, even two x86 ones, can round the last digits
of a long calculation differently, so a byte-for-byte comparison fails on some
CI machines when nothing real has changed. A real change in the code moves the
numbers by far more than 1e-9.

Run it from the repo root after `bearing-audit all`:  python scripts/compare_results.py
"""

import io
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = "reports/results"
RTOL, ATOL = 1e-9, 1e-12


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], capture_output=True, check=True).stdout


def rel_diff(a: np.ndarray, b: np.ndarray) -> float:
    finite = np.isfinite(a) & np.isfinite(b)
    if not finite.any():
        return 0.0
    a, b = a[finite], b[finite]
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


def compare_csv(old: bytes, new: Path) -> tuple[list[str], float]:
    a, b = pd.read_csv(io.BytesIO(old)), pd.read_csv(new)
    if list(a.columns) != list(b.columns):
        return ["columns changed"], 0.0
    if len(a) != len(b):
        return [f"{len(a)} rows committed, {len(b)} now"], 0.0
    problems, worst = [], 0.0
    for col in a.columns:
        x, y = a[col], b[col]
        numeric = all(pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s) for s in (x, y))
        if numeric:
            xv, yv = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
            ok = np.isclose(yv, xv, rtol=RTOL, atol=ATOL, equal_nan=True)
            worst = max(worst, rel_diff(yv, xv))
            if not ok.all():
                problems.append(f"column {col}: {int((~ok).sum())} numbers differ by more than the tolerance")
        elif not x.astype(str).equals(y.astype(str)):
            problems.append(f"column {col}: {int((x.astype(str) != y.astype(str)).sum())} entries changed")
    return problems, worst


def compare_json(old, new, where: str = "") -> tuple[list[str], float]:
    here = where or "top level"
    if isinstance(old, bool) or isinstance(new, bool) or old is None or new is None or isinstance(old, str):
        return ([] if old == new else [f"{here}: {old!r} committed, {new!r} now"]), 0.0
    if isinstance(old, (int, float)) and isinstance(new, (int, float)):
        a, b = float(old), float(new)
        if math.isnan(a) or math.isnan(b):
            return ([] if math.isnan(a) and math.isnan(b) else [f"{here}: {old} committed, {new} now"]), 0.0
        worst = abs(b - a) / max(abs(a), 1e-300)
        ok = math.isclose(b, a, rel_tol=RTOL, abs_tol=ATOL)
        return ([] if ok else [f"{here}: {old} committed, {new} now"]), worst
    if isinstance(old, dict) and isinstance(new, dict):
        if old.keys() != new.keys():
            return [f"{here}: keys changed"], 0.0
        problems, worst = [], 0.0
        for k in old:
            p, w = compare_json(old[k], new[k], f"{where}.{k}" if where else k)
            problems += p
            worst = max(worst, w)
        return problems, worst
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            return [f"{here}: {len(old)} items committed, {len(new)} now"], 0.0
        problems, worst = [], 0.0
        for i, (o, n) in enumerate(zip(old, new, strict=True)):
            p, w = compare_json(o, n, f"{where}[{i}]")
            problems += p
            worst = max(worst, w)
        return problems, worst
    return [f"{here}: type changed"], 0.0


def main() -> int:
    committed = git("ls-files", "--", RESULTS).decode().split()
    problems, identical, close, worst = [], 0, [], 0.0
    for name in committed:
        path = Path(name)
        if not path.exists():
            problems.append(f"{name}: missing")
            continue
        old = git("show", f"HEAD:{name}")
        if old == path.read_bytes():
            identical += 1
            continue
        if path.suffix == ".csv":
            found, w = compare_csv(old, path)
        elif path.suffix == ".json":
            found, w = compare_json(json.loads(old), json.loads(path.read_text()))
        else:
            found, w = ["changed, and it's neither CSV nor JSON"], 0.0
        problems += [f"{name}: {p}" for p in found]
        if not found:
            close.append(name)
            worst = max(worst, w)
    for path in sorted(p for p in Path(RESULTS).rglob("*") if p.is_file()):
        if path.as_posix() not in committed:
            problems.append(f"{path.as_posix()}: made by the pipeline but not committed")

    for line in problems:
        print(line)
    print(f"{len(committed)} result files checked: {identical} byte-identical, {len(close)} within tolerance")
    if close:
        print(f"  within tolerance: {', '.join(Path(n).name for n in close)} (largest relative difference {worst:.1e})")
    print(f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
