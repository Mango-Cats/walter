"""Combine CSV files in a directory into scrambled and class-partitioned datasets.

This script reads all CSV files from a directory, combines their entries, and outputs:
1. `D.csv` - All entries combined and randomly scrambled (shuffled).
2. `D_<class>.csv` - Separate files containing all entries for each unique class/label.

Usage:
    python scripts/combine_csvs.py path/to/dir
    python scripts/combine_csvs.py path/to/dir --class-col label --seed 42
    python scripts/combine_csvs.py path/to/dir --output-dir path/to/output
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd


CANDIDATE_CLASS_COLS = ["label", "class", "target", "category"]


def sanitize_filename_class(cls_val: Any) -> str:
    """Format and sanitize class values for use in output filenames (D_<class>.csv)."""
    if pd.isna(cls_val):
        return "nan"
    # If float represents an integer (e.g. 1.0 -> '1')
    if isinstance(cls_val, float) and cls_val.is_integer():
        cls_val = int(cls_val)
    cls_str = str(cls_val).strip()
    # Replace slashes, spaces, and filesystem-reserved characters with underscores
    sanitized = re.sub(r'[^\w\-.]', '_', cls_str)
    return sanitized or "unknown"


def find_class_column(df: pd.DataFrame, requested_col: str | None) -> str:
    """Find the target class column or auto-detect from common candidates."""
    if requested_col:
        if requested_col in df.columns:
            return requested_col
        raise ValueError(
            f"Requested class column '{requested_col}' not found in dataset. "
            f"Available columns: {df.columns.tolist()}"
        )

    for candidate in CANDIDATE_CLASS_COLS:
        if candidate in df.columns:
            return candidate

    raise ValueError(
        f"Could not auto-detect a class column. Checked: {CANDIDATE_CLASS_COLS}. "
        f"Available columns: {df.columns.tolist()}. Please specify --class-col."
    )


def combine_csv_files(
    input_dir: Path,
    output_dir: Path | None = None,
    class_col: str | None = None,
    pattern: str = "*.csv",
    seed: int | None = 42,
    shuffle: bool = True,
    deduplicate: bool = False,
    include_d: bool = False,
) -> dict[str, Path]:
    """Combine all CSV files in input_dir into D.csv and D_<class>.csv files.

    Args:
        input_dir: Directory containing input CSV files.
        output_dir: Directory where combined CSVs will be saved (defaults to input_dir).
        class_col: Name of column to partition classes by (auto-detected if None).
        pattern: Glob pattern to match CSV files.
        seed: Random seed for scrambling/shuffling.
        shuffle: Whether to scramble the rows.
        deduplicate: Whether to remove duplicate rows before saving.
        include_d: Whether to include existing D.csv / D_*.csv files in input_dir.

    Returns:
        Dictionary mapping output file keys to their Path objects.
    """
    input_path = Path(input_dir).resolve()
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input directory does not exist or is not a directory: {input_path}")

    out_path = Path(output_dir).resolve() if output_dir else input_path
    out_path.mkdir(parents=True, exist_ok=True)

    # Discover CSV files in input_dir
    matched_files = sorted(input_path.glob(pattern))
    if not include_d:
        # Exclude existing output files to prevent compounding on repeated runs
        csv_files = [
            f for f in matched_files
            if f.is_file() and not (f.name == "D.csv" or (f.name.startswith("D_") and f.suffix == ".csv"))
        ]
    else:
        csv_files = [f for f in matched_files if f.is_file()]

    if not csv_files:
        raise FileNotFoundError(
            f"No source CSV files found in '{input_path}' matching pattern '{pattern}'"
            + (" (excluding D*.csv)" if not include_d else "")
        )

    print(f"Loading {len(csv_files)} CSV file(s) from {input_path}:")
    dfs: list[pd.DataFrame] = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            print(f"  • {f.name} ({len(df):,} rows)")
            dfs.append(df)
        except Exception as exc:
            print(f"  ⚠ Failed to read {f.name}: {exc}", file=sys.stderr)

    if not dfs:
        raise ValueError("No data could be loaded from the matched CSV files.")

    combined = pd.concat(dfs, ignore_index=True)
    total_loaded = len(combined)
    print(f"\nTotal rows loaded: {total_loaded:,}")

    if deduplicate:
        combined = combined.drop_duplicates().reset_index(drop=True)
        dropped = total_loaded - len(combined)
        if dropped > 0:
            print(f"Deduplication removed {dropped:,} duplicate row(s); {len(combined):,} remaining.")

    # Scramble / Shuffle
    if shuffle:
        combined = combined.sample(frac=1, random_state=seed).reset_index(drop=True)
        print(f"Scrambled rows using random seed: {seed}")

    # Determine class column
    target_class_col = find_class_column(combined, class_col)
    print(f"Using class column: '{target_class_col}'")

    output_files: dict[str, Path] = {}

    # 1. Save all entries scrambled -> D.csv
    d_path = out_path / "D.csv"
    combined.to_csv(d_path, index=False)
    output_files["D"] = d_path
    print(f"\nSaved all scrambled entries ({len(combined):,} rows) → {d_path}")

    # 2. Save each class -> D_<class>.csv
    grouped = combined.groupby(target_class_col, dropna=False)
    print(f"\nPartitioning into {len(grouped)} class file(s):")

    for cls_val, group in grouped:
        cls_sanitized = sanitize_filename_class(cls_val)
        cls_filename = f"D_{cls_sanitized}.csv"
        cls_path = out_path / cls_filename
        group.to_csv(cls_path, index=False)
        output_files[f"D_{cls_sanitized}"] = cls_path
        print(f"  • Class '{cls_val}' ({len(group):,} rows) → {cls_path}")

    return output_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Combine CSV files in a directory into scrambled D.csv and partitioned D_<class>.csv files."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        type=Path,
        help="Directory containing the input CSV files (default: current directory).",
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        dest="opt_input_dir",
        type=Path,
        default=None,
        help="Optional alias for input directory.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save output files (default: same as input directory).",
    )
    parser.add_argument(
        "-c",
        "--class-col",
        type=str,
        default=None,
        help="Name of the class/label column to partition by (default: auto-detect 'label', 'class', etc.).",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Random seed for scrambling rows (default: 42).",
    )
    parser.add_argument(
        "-p",
        "--pattern",
        type=str,
        default="*.csv",
        help="Glob pattern to match input CSV files (default: '*.csv').",
    )
    parser.add_argument(
        "--no-shuffle",
        dest="shuffle",
        action="store_false",
        help="Do not scramble rows; maintain input row order.",
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="Remove exact duplicate rows across combined files.",
    )
    parser.add_argument(
        "--include-d",
        action="store_true",
        help="Include existing D.csv / D_*.csv files from the input directory.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_dir = args.opt_input_dir or args.directory
    try:
        combine_csv_files(
            input_dir=input_dir,
            output_dir=args.output_dir,
            class_col=args.class_col,
            pattern=args.pattern,
            seed=args.seed,
            shuffle=args.shuffle,
            deduplicate=args.deduplicate,
            include_d=args.include_d,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

