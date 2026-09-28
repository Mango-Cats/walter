"""Filipino (Tagalog) grapheme-to-phoneme (G2P) transcription.

This module uses a trained Phonetisaurus model to convert Filipino drug names
into International Phonetic Alphabet (IPA) pronunciations.
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

_UNKNOWN_SYM = re.compile(r"Symbol: '(.+?)' not found in input symbols table")
_DECODABLE = re.compile(r"[a-z]+")

_IPA_FIXUPS = {
    "d͡ʒ": "ʤ",
    "t͡ʃ": "ʧ",
    "ɡ": "g",
}

_TONE_LETTERS = re.compile(r"[˥-˩]")


def _resolve_model() -> Path:
    """Confirm that the Filipino G2P pronunciation model exists on disk.

    Returns:
        Path to the model file.

    Raises:
        FileNotFoundError: If the model file is missing.

    """
    if not FIL_G2P_MODEL.is_file():
        raise FileNotFoundError(
            f"Filipino G2P model not found at {FIL_G2P_MODEL}. Copy it from "
            "Taglog-G2P (notebook/train/cwik_model.fst), keeping the filename."
        )
    return FIL_G2P_MODEL


@lru_cache(maxsize=1)
def _resolve_decoder() -> tuple[str, dict[str, str]]:
    """Locate the phonetisaurus binary and prepare its execution environment.

    Returns:
        Tuple of (binary_path, environment_dictionary).

    Raises:
        RuntimeError: If phonetisaurus is not installed or unsupported.

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
    """Convert a list of unique word tokens into IPA pronunciations using Phonetisaurus.

    Args:
        tokens: List of unique word tokens to decode.
        model: Path to the pronunciation model file.

    Returns:
        Dictionary mapping each word token to its IPA pronunciation.

    Raises:
        RuntimeError: If the decoder fails or encounters invalid characters.

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
        raise RuntimeError(
            f"character(s) not in the model alphabet: "
            f"{', '.join(map(repr, unknown))}. Drug names must be cleaned to the "
            "model's grapheme set (lowercase Tagalog letters) before transcription; "
            "digits and punctuation are not decodable."
        )

    out: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) == 3 and parts[2].strip():
            word, _score, phones = parts
            ipa = "".join(phones.split())
            for src, tgt in _IPA_FIXUPS.items():
                ipa = ipa.replace(src, tgt)
            ipa = _TONE_LETTERS.sub("", ipa)
            out.setdefault(word, ipa)
    return out


def _chunk(name: str) -> list[tuple[str, bool]]:
    """Split a drug name into decodable alphabetic chunks and non-decodable tokens.

    Args:
        name: Drug name string.

    Returns:
        List of (text_segment, is_decodable) tuples.

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
    """Convert a batch of drug names into Filipino IPA pronunciations.

    Args:
        names: List of drug names.

    Returns:
        List of IPA pronunciation strings.

    """
    model = _resolve_model()

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
    """Add Filipino IPA pronunciation columns to a drug-pair DataFrame.

    Args:
        df: DataFrame containing drug pair columns COL_X1 and COL_X2.
        batch_size: Number of unique drug names per Phonetisaurus batch.
        verbose: Whether to print progress messages.

    Returns:
        Copy of DataFrame with Filipino pronunciation columns added.

    """
    return _transcribe_dataframe(
        df,
        batch_fn=_transcribe_batch,
        out_cols=(COL_T_FIL_1, COL_T_FIL_2),
        tag=_TAG,
        batch_size=batch_size,
        verbose=verbose,
    )
