"""
Filipino (Tagalog) G2P: adds IPA phonetic transcriptions to a drug-pair
DataFrame using a Phonetisaurus WFST.

The model is the one trained in the Taglog-G2P repo's notebook/Wik_eval.ipynb
on Tagalog Wiktionary (clean_tgl_wik.csv). Always takes the best (top-1)
pronunciation.

Requirements:
    the `phonetisaurus` wheel (a project dependency; `uv sync`)
    the trained .fst checked out at config.FIL_G2P_MODEL

Shape mirrors src/adapters/g2p/eng.py: same transcribe_dataframe() entrypoint, same
concatenated-IPA output format. Unlike eSpeak, the decoder is a subprocess and
works one word at a time, so names are split into runs of model-alphabet
graphemes and every unique run across the whole batch is decoded in a single
--wordlist invocation.

Names carry digits (preprocessing.clean_name keeps them: "b12", "stabigran
150"), which the WFST cannot decode. Undecodable runs are passed through
verbatim into the IPA string rather than dropped.

Note that phoc's ALINE reader silently discards digits, whitespace, punctuation,
and the length/stress marks (ː ˈ ˌ), so the passthrough preserves them in the IPA
column but NOT in any phonetic-similarity score computed from it: "stabigran 150"
and "stabigran 75" transcribe differently and still score aline == 1.0. That
tolerance does not extend to every non-inventory character -- tone letters, for
one, hard-error -- so anything the WFST can emit that ALINE lacks must be
normalised away here (see _IPA_FIXUPS, _TONE_LETTERS).
"""

import importlib.util
import os
import platform
import re
import shutil
import subprocess
import tempfile
import unicodedata
from functools import lru_cache
from pathlib import Path

import pandas as pd

from config import (
    COL_T_FIL_1,
    COL_T_FIL_2,
    FIL_G2P_BATCH_SIZE,
    FIL_G2P_BIN,
    FIL_G2P_MODEL,
)
from src.adapters.g2p.client import transcribe_dataframe as _transcribe_dataframe

_TAG = "fil_g2p"

# Phonetisaurus prints this to stderr (not stdout) for each input character it
# cannot map; we scrape it to detect out-of-alphabet input. The decoder drops
# such characters silently, so its output for the affected word is unreliable.
_UNKNOWN_SYM = re.compile(r"Symbol: '(.+?)' not found in input symbols table")

# The model's grapheme alphabet, after NFC + lowercasing. clean_name() already
# restricts names to [a-z0-9 ], so digits are the only undecodable runs we
# expect, but the split is defined by what the model accepts, not by what we
# expect to meet.
_DECODABLE = re.compile(r"[a-z]+")

# Rewrite the WFST's phones to the spellings bin/pho_conf's ALINE reader accepts.
# It lists both spellings of each affricate but only tokenizes the precomposed
# one, and keys the velar stop on the ASCII lookalike. Order matters: the
# affricates must be replaced before any single-character rule could touch them.
_IPA_FIXUPS = {
    "d͡ʒ": "ʤ",
    "t͡ʃ": "ʧ",  # U+0074 U+0361 U+0283 -> U+02A7
    "ɡ": "g",  # ɡ (IPA velar stop, U+0261) -> g (U+0067)
}

# The WFST's output alphabet includes all five IPA tone letters (U+02E5..U+02E9),
# inherited from tone-marked Wiktionary entries; the decoder emits them rarely
# and unpredictably (~1 name in 85k). Tagalog is not tonal and ALINE has no tone
# feature, so dropping them is lossless for scoring. This is not the same as the
# length/stress marks (ː ˈ ˌ) or digits, which phoc's ALINE reader ignores on its
# own -- tone letters are absent from that ignore set and hard-error as
# UnknownToken instead, so they must be stripped here.
_TONE_LETTERS = re.compile(r"[˥-˩]")


def _resolve_model() -> Path:
    """Confirm the WFST exists on disk and return its path."""
    if not FIL_G2P_MODEL.is_file():
        raise FileNotFoundError(
            f"Filipino G2P model not found at {FIL_G2P_MODEL}. Copy it from "
            "Taglog-G2P (notebook/train/cwik_model.fst), keeping the filename."
        )
    return FIL_G2P_MODEL


@lru_cache(maxsize=1)
def _resolve_decoder() -> tuple[str, dict[str, str]]:
    """
    Locate phonetisaurus-g2pfst and build the environment it needs to run.

    Prefers the binary bundled in the `phonetisaurus` wheel. That binary links
    against OpenFst (libfst.so.13) which the wheel also bundles, but it carries
    no RPATH, so the bundled lib dir is prepended to LD_LIBRARY_PATH here.

    Returns (binary path, env). Honours config.FIL_G2P_BIN as an override, in
    which case the caller's environment is used unchanged.
    """
    if FIL_G2P_BIN:
        binary = shutil.which(FIL_G2P_BIN) or FIL_G2P_BIN
        return binary, dict(os.environ)

    spec = importlib.util.find_spec("phonetisaurus")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(
            "The `phonetisaurus` package is not installed. Run `uv sync` (it is "
            "a project dependency), or set config.FIL_G2P_BIN to an external "
            "phonetisaurus-g2pfst binary."
        )

    root = Path(next(iter(spec.submodule_search_locations)))
    arch = platform.machine()
    binary = root / "bin" / arch / "phonetisaurus-g2pfst"
    lib_dir = root / "lib" / arch

    if not binary.is_file():
        available = sorted(p.name for p in (root / "bin").glob("*")) or ["<none>"]
        raise RuntimeError(
            f"The installed `phonetisaurus` wheel ships no binary for this "
            f"architecture ({arch}); it has: {', '.join(available)}. Set "
            "config.FIL_G2P_BIN to a phonetisaurus-g2pfst you built yourself."
        )

    env = dict(os.environ)
    if lib_dir.is_dir():
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [str(lib_dir), env["LD_LIBRARY_PATH"]]
            if env.get("LD_LIBRARY_PATH")
            else [str(lib_dir)]
        )
    return str(binary), env


def _decode(tokens: list[str], model: Path) -> dict[str, str]:
    """
    Decode every token in one phonetisaurus call, via a --wordlist temp file.

    Returns {token: concatenated IPA}. Tokens the decoder produced no output
    for are absent from the mapping. Raises RuntimeError if the binary is
    missing, exits non-zero, or hit a character outside the model alphabet.
    """
    binary, env = _resolve_decoder()

    with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8") as wl:
        wl.write("\n".join(tokens) + "\n")
        wl.flush()
        try:
            proc = subprocess.run(
                [binary, f"--model={model}", f"--wordlist={wl.name}", "--nbest=1"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                env=env,
                check=True,
            )
        except FileNotFoundError as e:
            raise RuntimeError(f"phonetisaurus decoder not found at {binary}") from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"{binary} failed: {e.stderr.strip()}") from e

    unknown = sorted(set(_UNKNOWN_SYM.findall(proc.stderr)))
    if unknown:
        # Fail loudly: the decoder drops these silently, so whichever words
        # contained them got a plausible-looking but wrong transcription.
        raise RuntimeError(
            f"character(s) not in the model alphabet: "
            f"{', '.join(map(repr, unknown))}. Drug names must be cleaned to the "
            "model's grapheme set (lowercase Tagalog letters) before transcription; "
            "digits and punctuation are not decodable."
        )

    # Each stdout line is: word\tscore\tspace-separated phones. Keep the first
    # well-formed line per word (nbest=1, so there should only be one).
    out: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) == 3 and parts[2].strip():
            word, _score, phones = parts
            # Concatenate phones to match eng_g2p's format. phoc's aline parser
            # ignores whitespace, so this is lossless for downstream scoring.
            ipa = "".join(phones.split())
            for src, tgt in _IPA_FIXUPS.items():
                ipa = ipa.replace(src, tgt)
            ipa = _TONE_LETTERS.sub("", ipa)
            out.setdefault(word, ipa)
    return out


def _chunk(name: str) -> list[tuple[str, bool]]:
    """
    Split a name into consecutive (text, decodable) runs, in order.

    "stabigran 150" -> [("stabigran", True), (" 150", False)]
    "b12"           -> [("b", True), ("12", False)]

    Reassembling the texts reproduces `name` exactly.
    """
    chunks: list[tuple[str, bool]] = []
    pos = 0
    for m in _DECODABLE.finditer(name):
        if m.start() > pos:
            chunks.append((name[pos : m.start()], False))
        chunks.append((m.group(), True))
        pos = m.end()
    if pos < len(name):
        chunks.append((name[pos:], False))
    return chunks


def _transcribe_batch(names: list[str]) -> list[str]:
    """
    Transcribe a batch of drug names to IPA using the Phonetisaurus WFST.
    Returns one IPA string per name, aligned to the input order.

    A name may be multi-word ("vitamin c") and may carry digits ("b12"); the
    model is word-level and letters-only, so each decodable run is decoded and
    the results are concatenated with the undecodable runs left in place. A
    decodable run the model produced no output for contributes "".
    """
    model = _resolve_model()

    # The model's grapheme alphabet is lowercase and NFC; match it.
    normalized = [unicodedata.normalize("NFC", n).lower() for n in names]
    chunked = [_chunk(n) for n in normalized]

    tokens = sorted({text for cs in chunked for text, ok in cs if ok})
    decoded = _decode(tokens, model) if tokens else {}

    out = []
    for cs in chunked:
        parts = []
        for text, ok in cs:
            if ok:
                parts.append(decoded.get(text, ""))
            else:
                # Whitespace is a separator, not content: eng_g2p emits phones
                # unseparated, so drop it and keep only the literal graphemes.
                literal = "".join(text.split())
                if literal:
                    parts.append(literal)
        out.append("".join(parts))
    return out


def transcribe_dataframe(
    df: pd.DataFrame,
    batch_size: int = FIL_G2P_BATCH_SIZE,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Add Filipino IPA transcription columns COL_T_FIL_1 / COL_T_FIL_2 to a
    drug-pair DataFrame.

    Args:
        df:         DataFrame with columns COL_X1 and COL_X2.
        batch_size: Number of unique names per phonetisaurus call.
        verbose:    Print progress.
    Returns:
        Copy of df with added columns COL_T_FIL_1 and COL_T_FIL_2.
    """
    return _transcribe_dataframe(
        df,
        batch_fn=_transcribe_batch,
        out_cols=(COL_T_FIL_1, COL_T_FIL_2),
        tag=_TAG,
        batch_size=batch_size,
        verbose=verbose,
    )
