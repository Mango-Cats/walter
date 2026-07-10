"""
Filipino (Tagalog) G2P: adds IPA phonetic transcriptions to a drug-pair
DataFrame using a Phonetisaurus WFST.

The model is the one trained in the Taglog-G2P repo's notebook/Wik_eval.ipynb
on Tagalog Wiktionary (clean_tgl_wik.csv). Always takes the best (top-1)
pronunciation.

Requirements:
    the `phonetisaurus` wheel (a project dependency; `uv sync`)
    the trained .fst checked out at config.FIL_G2P_MODEL

Shape mirrors src/eng_g2p.py: same transcribe_dataframe() entrypoint, same
concatenated-IPA output format. Unlike eSpeak, the decoder is a subprocess and
works one word at a time, so names are tokenized on whitespace and every unique
token across the whole batch is decoded in a single --wordlist invocation.
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
from src.g2p_common import transcribe_dataframe as _transcribe_dataframe

_TAG = "fil_g2p"

# Phonetisaurus prints this to stderr (not stdout) for each input character it
# cannot map; we scrape it to detect out-of-alphabet input. The decoder drops
# such characters silently, so its output for the affected word is unreliable.
_UNKNOWN_SYM = re.compile(r"Symbol: '(.+?)' not found in input symbols table")


def _resolve_model() -> Path:
    """Confirm the WFST exists on disk and return its path."""
    if not FIL_G2P_MODEL.is_file():
        raise FileNotFoundError(
            f"Filipino G2P model not found at {FIL_G2P_MODEL}. Copy "
            "cwik_model.fst (Taglog-G2P: notebook/train/cwik_model.fst) there."
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
            [str(lib_dir), env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH")
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
            out.setdefault(word, "".join(phones.split()))
    return out


def _transcribe_batch(names: list[str]) -> list[str]:
    """
    Transcribe a batch of drug names to IPA using the Phonetisaurus WFST.
    Returns one IPA string per name, aligned to the input order.

    A name may be multi-word ("vitamin c"); the model is word-level, so each
    whitespace token is decoded and the results concatenated. A name whose
    tokens all fail to decode yields "".
    """
    model = _resolve_model()

    # The model's grapheme alphabet is lowercase and NFC; match it.
    normalized = [unicodedata.normalize("NFC", n).lower() for n in names]
    tokens = sorted({tok for n in normalized for tok in n.split()})
    if not tokens:
        return ["" for _ in names]

    decoded = _decode(tokens, model)
    return ["".join(decoded.get(tok, "") for tok in n.split()) for n in normalized]


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
