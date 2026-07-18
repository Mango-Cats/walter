"""
Client for the bundled ``bin/tbb-cli`` TagaBaybay stream worker.

tbb-cli is a long-lived Rust worker that adapts loanwords into Filipino
orthography over a newline-delimited JSON (JSONL) protocol on stdin/stdout: the
parent writes one JSON request per line, the child replies with one JSON
response per line (see the binary's module docstring for the full protocol).

We spawn the worker once, keep it warm, and reuse it for every word, so the
per-word cost is a single request/response round trip. Results are memoized, so
each distinct name is only ever adapted once no matter how many pairs it appears
in. The worker is shut down cleanly at interpreter exit.

``nativize(word)`` returns the worker's ``adapted`` spelling (e.g. "chocolate"
-> "tsokoleyt") — the Filipino phonemic form that the phonetic feature functions
in ``feature_engineering.py`` derive onset / coda / vowel-skeleton indicators
from. This replaces the hand-rolled orthographic substitution rules that used to
live in that module.
"""

from __future__ import annotations

import atexit
import errno
import json
import os
import re
import stat
import subprocess
import threading
import time
from pathlib import Path

from config import TBB_BIN

# Fallback sanitizer used only when the worker declines a specific word: keep
# Latin letters (and ñ) so a single un-adaptable name degrades to a plain
# lowercased skeleton instead of aborting the whole feature build.
_NON_LETTER = re.compile(r"[^a-zñ]")


def _sanitize(word: str) -> str:
    return _NON_LETTER.sub("", word.lower())


class _TbbWorker:
    """A single persistent tbb-cli subprocess with a nativization cache."""

    def __init__(self, bin_path: Path = TBB_BIN):
        if not bin_path.exists():
            raise FileNotFoundError(
                f"tbb-cli binary not found at {bin_path}. It ships in the repo "
                "under bin/tbb-cli — check it out or rebuild it from the "
                "tagabaybay crate."
            )
        # Like bin/phoc, the binary is copied between machines, which can strip
        # the execute bit. Restore it rather than fail with Errno 13 (EACCES).
        if not os.access(bin_path, os.X_OK):
            try:
                mode = bin_path.stat().st_mode
                bin_path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            except OSError as e:
                raise PermissionError(
                    f"tbb-cli at {bin_path} is not executable and could not be "
                    f"made executable ({e}). Run: chmod +x {bin_path}"
                ) from e

        # ETXTBSY (Errno 26): the binary may still be open for writing (freshly
        # copied/synced). It clears on its own, so retry with a short backoff.
        proc = None
        for attempt in range(5):
            try:
                proc = subprocess.Popen(
                    [str(bin_path)],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    # The worker prints IPA-alignment debug lines to stderr and,
                    # for adapt calls, interleaves them on stdout too; we ignore
                    # stderr entirely and skip non-JSON stdout lines when reading.
                    stderr=subprocess.DEVNULL,
                    text=True,
                    bufsize=1,  # line-buffered so requests reach the child promptly
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
        self._cache: dict[str, str] = {}

        # Drain the startup "ready" event before the first request.
        self._read_json()

    def _read_json(self, want_id: int | None = None) -> dict:
        """Read stdout lines until one parses as a JSON object.

        The worker interleaves human-readable alignment lines (``0: ch -> tʃ``)
        with its JSON responses on stdout, so we skip anything that isn't JSON.
        When ``want_id`` is given we also skip JSON whose ``id`` doesn't match,
        which keeps us correlated even if an unexpected line slips through.
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
                continue  # debug/alignment noise
            if not isinstance(obj, dict):
                continue
            if want_id is not None and obj.get("id") != want_id:
                continue
            return obj

    def nativize(self, word: str) -> str:
        """Return the Filipino ``adapted`` spelling for ``word`` (cached).

        Falls back to a sanitized (letters-only, lowercased) skeleton if the
        worker reports it cannot adapt the word, so one odd name never aborts
        the whole run.
        """
        key = _sanitize(word)
        if not key:
            return ""
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        with self._lock:
            # Re-check under the lock in case another thread just filled it in.
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
                adapted = resp["adapted"]
            else:
                adapted = key  # error / empty — degrade to the plain skeleton
            self._cache[key] = adapted
            return adapted

    def close(self) -> None:
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
    """Lazily spawn the shared worker on first use."""
    global _worker
    if _worker is None:
        with _worker_lock:
            if _worker is None:
                _worker = _TbbWorker()
                atexit.register(_worker.close)
    return _worker


def nativize(word: str) -> str:
    """Adapt ``word`` into its Filipino orthographic form via bin/tbb-cli."""
    return _get_worker().nativize(word)
