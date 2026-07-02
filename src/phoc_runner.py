"""
Phonetic-feature step: runs the bundled `bin/phoc` Rust CLI over an already
assembled pair CSV.

phoc reads a CSV with x_1/x_2 (plus optional t_1/t_2, label, and any other
columns), preserves every input column verbatim, and appends one similarity
feature column per .toml config in PHOC_CONFIG_DIR — the column name is the
config file's stem, the algorithm is chosen by that file's `algorithm` key.

Because walter.py writes D.csv / D_rank.csv before calling into here, a phoc
failure never destroys those base datasets: it just aborts the run with a
clear error and leaves the base CSVs on disk.
"""

import csv
import errno
import os
import stat
import subprocess
import time
from pathlib import Path

from config import PHOC_BIN, PHOC_CONFIG_DIR


def _header(csv_path: Path) -> list[str]:
    with csv_path.open(newline="") as f:
        return next(csv.reader(f), [])


def run_phoc(
    input_csv: Path,
    output_csv: Path,
    config_dir: Path = PHOC_CONFIG_DIR,
) -> list[str]:
    """
    Run phoc on ``input_csv`` and write the feature-augmented CSV to
    ``output_csv``. Returns the list of feature columns phoc appended
    (output header minus input header).

    Raises FileNotFoundError if the phoc binary or config dir is missing,
    and RuntimeError (with phoc's stderr) if phoc exits non-zero.
    """
    if not PHOC_BIN.exists():
        raise FileNotFoundError(
            f"phoc binary not found at {PHOC_BIN}. "
            "It ships in the repo under bin/phoc — check it out or rebuild it."
        )
    # bin/phoc is gitignored (194 MB) and gets copied between machines, which
    # can strip the execute bit. Restore it rather than fail with Errno 13.
    if not os.access(PHOC_BIN, os.X_OK):
        try:
            mode = PHOC_BIN.stat().st_mode
            PHOC_BIN.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        except OSError as e:
            raise PermissionError(
                f"phoc binary at {PHOC_BIN} is not executable and could not be "
                f"made executable ({e}). Run: chmod +x {PHOC_BIN}"
            ) from e
    if not config_dir.is_dir():
        raise FileNotFoundError(
            f"phoc config dir not found at {config_dir}. "
            "It must contain one .toml per feature column."
        )

    input_cols = _header(input_csv)

    cmd = [
        str(PHOC_BIN),
        "--input",
        str(input_csv),
        "--output",
        str(output_csv),
        "--config-dir",
        str(config_dir),
    ]
    # ETXTBSY (Errno 26) means the binary is still open for writing elsewhere
    # (e.g. bin/phoc is 194 MB and may still be syncing/copying when the run
    # starts). It clears on its own, so retry a few times with a short backoff.
    for attempt in range(5):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            break
        except OSError as e:
            if e.errno == errno.ETXTBSY and attempt < 4:
                time.sleep(0.5 * (attempt + 1))
                continue
            raise
    if result.returncode != 0:
        raise RuntimeError(
            f"phoc failed (exit {result.returncode}) on {input_csv}:\n"
            f"{result.stderr.strip()}"
        )

    output_cols = _header(output_csv)
    return [c for c in output_cols if c not in input_cols]
    
