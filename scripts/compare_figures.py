"""Check that the regenerated figures look exactly like the committed ones.

The figures are compared pixel by pixel, not byte by byte. The same image
compresses to different PNG bytes on different machines (an Apple Silicon Mac
and an x86 Linux runner, for example), so a byte comparison would fail even
when nothing visible has changed. The result files (CSV, JSON) are still
compared byte by byte, with git diff.

Run it from the repo root after `bearing-audit all`:  python scripts/compare_figures.py
"""

import io
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

FIGURES = "reports/figures"


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], capture_output=True, check=True).stdout


def pixels(png: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"))


def main() -> int:
    committed = git("ls-files", "--", f"{FIGURES}/*.png").decode().split()
    problems = []
    for name in committed:
        path = Path(name)
        if not path.exists():
            problems.append(f"{name}: missing")
            continue
        old, new = pixels(git("show", f"HEAD:{name}")), pixels(path.read_bytes())
        if old.shape != new.shape:
            problems.append(f"{name}: size changed from {old.shape[1]}x{old.shape[0]} to {new.shape[1]}x{new.shape[0]}")
        elif not np.array_equal(old, new):
            changed = int(np.any(old != new, axis=2).sum())
            problems.append(f"{name}: {changed} pixels changed")
    for path in sorted(Path(FIGURES).glob("*.png")):
        if path.as_posix() not in committed:
            problems.append(f"{path.as_posix()}: made by the pipeline but not committed")

    for line in problems:
        print(line)
    print(f"{len(committed)} figures checked, {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
