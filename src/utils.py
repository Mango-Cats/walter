"""
General utility functions.
"""

import pathlib as pl


def finder(fname: str) -> pl.Path:
    """
    Finds the a file using its filename starting from the project root
    folder. Returns either the pl.Path or None.
    """

    current_dir: pl.Path = pl.Path(__file__).resolve().parent
    root_dir: pl.Path = current_dir.parent.parent
    path = next(root_dir.rglob(fname), None)

    if path is None:
        raise FileNotFoundError(
            f"Ensure the file with the filename {fname}, exists "
            " anywhere in the project's root folder."
        )

    return path
