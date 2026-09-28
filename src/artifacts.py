"""File path helpers for connecting pipeline stages.

Each pipeline stage reads and writes files using standard names inside directories.
This module ensures required input files exist before a stage starts, and creates
output directories automatically before writing.
"""

from pathlib import Path


def require_file(path: Path, produced_by: str) -> Path:
    """Verify that an input file exists on disk.

    Args:
        path: Path to the expected file.
        produced_by: Name of the Walter command that produces this file.

    Returns:
        The resolved Path object.

    Raises:
        FileNotFoundError: If the file does not exist.

    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `{produced_by}` first.")
    return path


def in_file(directory: Path, filename: str, produced_by: str) -> Path:
    """Resolve an input file path inside a directory and verify it exists.

    Args:
        directory: Folder holding the file.
        filename: Name of the file to read.
        produced_by: Name of the Walter command that generates this file.

    Returns:
        The resolved Path object.

    Raises:
        FileNotFoundError: If the file does not exist.

    """
    return require_file(Path(directory) / filename, produced_by)


def out_file(directory: Path, filename: str) -> Path:
    """Resolve an output file path inside a directory, creating folders as needed.

    Args:
        directory: Folder to write the file into.
        filename: Name of the output file.

    Returns:
        The resolved Path object for writing.

    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / filename


def seed_file(path: Path, what: str) -> Path:
    """Validate and resolve a user-supplied input file.

    Args:
        path: File path supplied by the user.
        what: Description of the expected file (for error reporting).

    Returns:
        The validated Path object.

    Raises:
        FileNotFoundError: If the path does not exist or points to a directory.

    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Provide the {what} there.")
    if path.is_dir():
        raise FileNotFoundError(f"{path} is a directory, expected the {what} file.")
    return path
