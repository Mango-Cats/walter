"""Filipino loanword adaptation using the `tbb-cli` tool.

What this file does:
Many drug names are originally English or Latin words. When spoken by Filipino
speakers, their pronunciation and spelling naturally adapt to Filipino phonetics
(e.g., 'chocolate' becomes 'tsokoleyt').

This module runs `bin/tbb-cli` (a fast background worker) to:
1. Adapt English drug names into their natural Filipino spelling ('nativized').
2. Break names into syllables (e.g., 'tso-ko-leyt').
3. Identify which syllable gets the vocal stress (e.g., 'tsokOleyt').

This helps Walter detect drug pairs that sound confusable specifically to Filipino
speakers, even if their standard English spellings look different.
"""

import atexit
import errno
import json
import os
import re
import stat
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from config import TBB_BIN

_NON_LETTER = re.compile(r"[^a-zñ]")


def _sanitize(word: str) -> str:
    """Strip non-alphabetical characters from a word.

    Args:
        word: Input word string.

    Returns:
        Sanitized lowercase word string containing only letters and ñ.

    """
    return _NON_LETTER.sub("", word.lower())


@dataclass(frozen=True, slots=True)
class Adaptation:
    """Filipino phonetic adaptation results for a single word.

    Attributes:
        nativized: Adapted spelling in Filipino orthography.
        syllabified: Hyphen-separated syllable breakdown.
        stressed: Spelling with capitalized vowel showing vocal stress.
        stressed_syllabified: Syllabified spelling including vocal stress.
        english_stress_on_penult: True if English source word has stress on the penult.

    """

    nativized: str
    syllabified: str
    stressed: str
    stressed_syllabified: str
    english_stress_on_penult: bool | None


class _TbbWorker:
    """Persistent background worker process for running tbb-cli with caching."""

    def __init__(self, bin_path: Path = TBB_BIN) -> None:
        """Spawn the tbb-cli process and configure it for stress marking.

        Args:
            bin_path: File path to the tbb-cli binary.

        Raises:
            FileNotFoundError: If the binary does not exist.
            PermissionError: If the binary cannot be made executable.

        """
        if not bin_path.exists():
            raise FileNotFoundError(
                f"tbb-cli binary not found at {bin_path}. It ships in the repo "
                "under bin/tbb-cli - check it out or rebuild it from the "
                "tagabaybay crate."
            )
        if not os.access(bin_path, os.X_OK):
            try:
                mode = bin_path.stat().st_mode
                bin_path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            except OSError as e:
                raise PermissionError(
                    f"tbb-cli at {bin_path} is not executable and could not be "
                    f"made executable ({e}). Run: chmod +x {bin_path}"
                ) from e

        proc = None
        for attempt in range(5):
            try:
                proc = subprocess.Popen(
                    [str(bin_path)],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    bufsize=1,
                )
                break
            except OSError as e:
                if e.errno == errno.ETXTBSY and attempt < 4:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise
        assert proc is not None and proc.stdin is not None and proc.stdout is not None
        self._proc = proc

        self._lock = threading.Lock()
        self._next_id = 0
        self._cache: dict[str, Adaptation] = {}

        self._read_json()

        assert self._proc.stdin is not None
        self._proc.stdin.write(
            json.dumps({"cmd": "config", "assign_prominence": True}) + "\n"
        )
        self._proc.stdin.flush()
        self._read_json()

    def _read_json(self, want_id: int | None = None) -> dict:
        """Read stdout lines until a valid JSON response is found.

        Args:
            want_id: Expected response ID to correlate requests and responses.

        Returns:
            Parsed JSON dictionary from the worker.

        Raises:
            RuntimeError: If the worker process terminates unexpectedly.

        """
        assert self._proc.stdout is not None
        while True:
            line = self._proc.stdout.readline()
            if line == "":
                raise RuntimeError("tbb-cli worker exited unexpectedly")
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            if want_id is not None and obj.get("id") != want_id:
                continue
            return obj

    def adapt(self, word: str) -> Adaptation:
        """Adapt a word into Filipino spellings, syllables, and vocal stress.

        Args:
            word: The word to adapt.

        Returns:
            An Adaptation object containing the adapted spellings.

        """
        key = _sanitize(word)
        if not key:
            return Adaptation("", "", "", "", None)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached

            self._next_id += 1
            req_id = self._next_id
            assert self._proc.stdin is not None
            self._proc.stdin.write(
                json.dumps({"id": req_id, "cmd": "adapt", "word": word}) + "\n"
            )
            self._proc.stdin.flush()

            resp = self._read_json(want_id=req_id)
            if resp.get("type") == "result" and resp.get("adapted"):
                nativized = resp["adapted"]
                syllabified = resp.get("syllables") or nativized
                stressed = resp.get("with_stress") or nativized
                stressed_syllabified = (
                    resp.get("with_stress_and_syllabified") or syllabified
                )
                english_stress_on_penult = resp.get("english_stress_on_penult")
                result = Adaptation(
                    nativized,
                    syllabified,
                    stressed,
                    stressed_syllabified,
                    english_stress_on_penult,
                )
            else:
                result = Adaptation(key, key, key, key, None)
            self._cache[key] = result
            return result

    def close(self) -> None:
        """Shut down the tbb-cli worker process cleanly."""
        proc = self._proc
        if proc.poll() is not None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                proc.stdin.flush()
                proc.stdin.close()
        except (OSError, ValueError):
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


_worker: _TbbWorker | None = None
_worker_lock = threading.Lock()


def _get_worker() -> _TbbWorker:
    """Return the shared background tbb-cli worker, spawning it on first use.

    Returns:
        The active _TbbWorker instance.

    """
    global _worker
    if _worker is None:
        with _worker_lock:
            if _worker is None:
                _worker = _TbbWorker()
                atexit.register(_worker.close)
    return _worker


def adapt(word: str) -> Adaptation:
    """Adapt a word into Filipino spellings, syllables, and stress markers.

    Args:
        word: Drug name or English word to adapt.

    Returns:
        An Adaptation dataclass containing all four Filipino spellings.

    """
    return _get_worker().adapt(word)


def nativize(word: str) -> str:
    """Adapt a word into its stress-marked Filipino spelling.

    Args:
        word: Drug name or English word to adapt.

    Returns:
        Stress-marked Filipino spelling string.

    """
    return _get_worker().adapt(word).stressed
