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

``adapt(word)`` returns an ``Adaptation`` with all four spellings the worker
can produce for a word:

- ``nativized``            - plain adapted spelling, e.g. "tsokoleyt"
- ``syllabified``          - hyphenated by syllable, e.g. "tso-ko-leyt"
- ``stressed``             - ``nativized`` with the prominent syllable's vowel
                              capitalized in place, not syllabified, e.g.
                              "tsokOleyt"
- ``stressed_syllabified`` - both: hyphenated and stress-capitalized, e.g.
                              "tso-kO-leyt"
- ``english_stress_on_penult`` - ``bool | None``; whether the source word's
                              English primary stress falls on the penult (or,
                              for a monosyllable, its only syllable). ``None``
                              when the source word's English stress couldn't be
                              looked up at all. This is the worker's
                              ``english_stress_on_penult`` field, passed through
                              unchanged - it describes the *English* word, not
                              the Filipino adaptation, so it's the same
                              regardless of whether the Filipino penult ends up
                              marked.

``nativize(word)`` is a convenience wrapper returning just ``stressed`` - the
Filipino phonemic form that the phonetic feature functions in
``src/pipeline/features.py`` derive onset / coda / vowel-skeleton indicators
from. This replaces the hand-rolled orthographic substitution rules that used
to live in that module.

Stress marking is enabled on the worker at startup via a ``config`` command
(``assign_prominence: true``). When the source word's English stress can't be
found, the worker omits ``with_stress``/``with_stress_and_syllabified`` from
the result and we fall back to the unmarked ``nativized``/``syllabified``
spelling for those fields.
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
    return _NON_LETTER.sub("", word.lower())


@dataclass(frozen=True, slots=True)
class Adaptation:
    """The four spellings tbb-cli can produce for one word, plus the English
    stress fact the marking rule was gated on."""

    nativized: str
    syllabified: str
    stressed: str
    stressed_syllabified: str
    english_stress_on_penult: bool | None


class _TbbWorker:
    """A single persistent tbb-cli subprocess with a nativization cache."""

    def __init__(self, bin_path: Path = TBB_BIN):
        if not bin_path.exists():
            raise FileNotFoundError(
                f"tbb-cli binary not found at {bin_path}. It ships in the repo "
                "under bin/tbb-cli - check it out or rebuild it from the "
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
        self._cache: dict[str, Adaptation] = {}

        # Drain the startup "ready" event before the first request.
        self._read_json()

        # Stress marking is off by default (extra English-stress lookup cost);
        # turn it on so results carry the capitalized `with_stress` spelling.
        assert self._proc.stdin is not None
        self._proc.stdin.write(
            json.dumps({"cmd": "config", "assign_prominence": True}) + "\n"
        )
        self._proc.stdin.flush()
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

    def adapt(self, word: str) -> Adaptation:
        """Return all four Filipino spellings for ``word`` (cached).

        Falls back to a sanitized (letters-only, lowercased) skeleton for all
        four spelling fields if the worker reports it cannot adapt the word at
        all, so one odd name never aborts the whole run. When the word adapts
        but its English stress can't be looked up, the stress-marked fields
        fall back to their unmarked counterparts (``stressed`` ->
        ``nativized``, ``stressed_syllabified`` -> ``syllabified``) and
        ``english_stress_on_penult`` is ``None``.
        """
        key = _sanitize(word)
        if not key:
            return Adaptation("", "", "", "", None)
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
                # error / empty - degrade to the plain skeleton everywhere
                result = Adaptation(key, key, key, key, None)
            self._cache[key] = result
            return result

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


def adapt(word: str) -> Adaptation:
    """Adapt ``word`` into all four Filipino spellings via bin/tbb-cli."""
    return _get_worker().adapt(word)


def nativize(word: str) -> str:
    """Adapt ``word`` into its stress-marked, non-syllabified spelling."""
    return _get_worker().adapt(word).stressed
