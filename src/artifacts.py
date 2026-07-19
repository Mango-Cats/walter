"""
Resolving a stage's input and output to actual files.

Every stage takes directories, not files: an artifact has one canonical
filename (config.paths), and a stage always reads that name out of its input
directory and writes that name into its output directory. Fixing the names
means the file a stage writes is already the file the next stage looks for,
so chaining stages is a matter of pointing them at the same directory.

The proposer is the one exception -- it takes a path to a specific CSV of
predefined LASA pairs -- so it uses seed_file() rather than in_file().

walter.py owns the CLI; nothing here parses arguments.
"""

from pathlib import Path


def require_file(path: Path, produced_by: str) -> Path:
    """
    Check that an input artifact exists, naming the command that writes it.

    produced_by is a walter command, since a missing input almost always
    means an earlier stage has not been run rather than a typo.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `{produced_by}` first.")
    return path


def in_file(directory: Path, filename: str, produced_by: str) -> Path:
    """Resolve <directory>/<filename> for reading."""
    return require_file(Path(directory) / filename, produced_by)


def out_file(directory: Path, filename: str) -> Path:
    """Resolve <directory>/<filename> for writing, creating the directory."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / filename


def seed_file(path: Path, what: str) -> Path:
    """
    Resolve a user-supplied input file.

    Unlike in_file(), no walter command produces this, so the error asks for
    the file instead of naming a stage to run.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Provide the {what} there.")
    if path.is_dir():
        raise FileNotFoundError(f"{path} is a directory, expected the {what} file.")
    return path
