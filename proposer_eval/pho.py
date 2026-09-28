"""Thin wrappers around the two tools under test: `bin/phoc` and TagaBaybay."""

import itertools
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from config import COL_X1, COL_X2, PHOC_BIN

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_PHO_CONF = PACKAGE_DIR / "pho_conf"
NATIVIZED_CACHE = PACKAGE_DIR / ".cache" / "nativized.json"


def _run_phoc(pairs: pd.DataFrame, config_dir: Path, flags: tuple[str, ...]) -> pd.DataFrame:
    if not PHOC_BIN.exists():
        raise FileNotFoundError(f"phoc binary not found at {PHOC_BIN} (run from the repo root)")
    with tempfile.TemporaryDirectory() as td:
        inp, out = Path(td) / "in.csv", Path(td) / "out.csv"
        pairs.to_csv(inp, index=False)
        proc = subprocess.run(
            [str(PHOC_BIN), "csv", "--input", str(inp), "--output", str(out),
             "--config-dir", str(config_dir), *flags],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"phoc failed ({proc.returncode}):\n{proc.stderr[-2000:]}")
        header = pd.read_csv(out, nrows=0).columns.tolist()
        feats = [c for c in header if c not in (COL_X1, COL_X2)]
        res = pd.read_csv(out, usecols=feats, dtype=np.float32)
    if len(res) != len(pairs):
        raise RuntimeError(f"phoc returned {len(res)} rows for {len(pairs)} pairs")
    return res


def phoc_features(
    left: list[str],
    right: list[str],
    config_dir: Path,
    flags: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Score every (left[i], right[j]) pair with phoc, anchor-major.

    One row per pair (len(left) * len(right) rows), one float32 column per
    feature phoc appends: a `.toml` in `config_dir` gives a column named after
    its stem, and `flags` such as --include-fil-features add more. Row k is
    pair (left[k // len(right)], right[k % len(right)]).
    """
    pairs = pd.DataFrame(list(itertools.product(left, right)), columns=[COL_X1, COL_X2])
    return _run_phoc(pairs, config_dir, flags)


def phoc_zipped(
    left: list[str],
    right: list[str],
    config_dir: Path,
    flags: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Like phoc_features but scores (left[i], right[i]) only."""
    pairs = pd.DataFrame({COL_X1: left, COL_X2: right})
    return _run_phoc(pairs, config_dir, flags)


def nativize_all(names: list[str]) -> dict[str, str]:
    """name -> nativized spelling, via walter's tbb-cli adapter.

    Uses `.nativized` (the plain adapted spelling) rather than the
    stress-marked form; Soundex uppercases everything anyway, and this keeps
    the input to phoc letters-only. Cached on disk because nativizing the whole
    registry is the one slow step and the answer never changes.
    """
    from src.adapters.tbb import adapt

    cache: dict[str, str] = {}
    if NATIVIZED_CACHE.exists():
        cache = json.loads(NATIVIZED_CACHE.read_text(encoding="utf-8"))
    todo = [n for n in dict.fromkeys(names) if n not in cache]
    if todo:
        print(f"[nativize] {len(todo)} new names ({len(cache)} cached) ...", flush=True)
        for i, n in enumerate(todo, 1):
            cache[n] = adapt(n).nativized
            if i % 2000 == 0:
                print(f"[nativize]   {i}/{len(todo)}", flush=True)
        # Never cache an inert run: without G2P, tbb-cli hands back raw letters,
        # and caching those would silently poison every later run.
        if not _warn_if_inert({n: cache[n] for n in todo}):
            NATIVIZED_CACHE.parent.mkdir(parents=True, exist_ok=True)
            NATIVIZED_CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return {n: cache[n] for n in names}


def _warn_if_inert(nat: dict[str, str]) -> bool:
    """tbb-cli degrades silently to the raw letters when its G2P backend is
    missing (eSpeak-NG / uv), which would make every nat:* score a copy of the
    pho:* score. Catch that instead of reporting a meaningless comparison."""
    import re

    letters = lambda s: re.sub(r"[^a-zñ]", "", s.lower())
    words = [n for n in nat if letters(n)]
    if len(words) < 50:
        return False
    changed = sum(letters(n) != letters(v) for n, v in nat.items() if letters(n))
    frac = changed / len(words)
    print(f"[nativize] nativizer changed the spelling of {frac:.0%} of names")
    if frac < 0.05:
        print("[nativize] WARNING: almost nothing was nativized. tbb-cli probably has no G2P "
              "backend (install eSpeak-NG); nat:*/max:*/mean:* results below are NOT meaningful.")
        return True
    return False
