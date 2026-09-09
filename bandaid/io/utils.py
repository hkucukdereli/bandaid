"""
File and directory utilities for lab naming conventions.
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path

__all__ = ["parse_filename", "get_files", "get_dirs", "get_subdir"]

DEFAULT_INDEX_PATTERN = r"(\d{3,})"


def _index_regex(index_pattern: str) -> re.Pattern:
    rx = re.compile(index_pattern)
    if rx.groups == 0:
        rx = re.compile(f"({index_pattern})")
    return rx


def parse_filename(filename: str, index_pattern: str = DEFAULT_INDEX_PATTERN) -> dict:
    """
    Break a lab filename into mouse, date, index, and leftover parts.

    Expects: ``<MOUSE>_<YYYYMMDD>_[extra_]<INDEX>[_extra].ext``
    where MOUSE is 2-4 letters optionally followed by digits.

    Parameters
    ----------
    filename : str
        Filename or full path. Path and extension are stripped.
    index_pattern : str, optional
        Regex matching the index segment, applied to each underscore-separated
        part. The first capture group (or the whole match) is the index.
        Default matches 3+ digits.

    Returns
    -------
    dict
        ``'mouse'`` : str or None — e.g. ``'HK13'``.
        ``'date'``  : str or None — e.g. ``'20260814'``.
        ``'index'`` : int or None — e.g. ``2``.
        ``'rest'``  : str — remaining parts joined with ``'_'``.

    Examples
    --------
    >>> parse_filename("HK13_20260814_50pw_00002.tif")
    {'mouse': 'HK13', 'date': '20260814', 'index': 2, 'rest': '50pw'}
    """
    stem = Path(filename).stem
    parts = stem.split("_")

    mouse = None
    date = None
    rest_start = 0

    if parts and re.fullmatch(r"[A-Za-z]{2,4}\d*", parts[0]):
        mouse = parts[0]
        rest_start = 1
        if len(parts) > 1 and re.fullmatch(r"\d{8}", parts[1]):
            date = parts[1]
            rest_start = 2

    remaining = parts[rest_start:]
    idx_re = _index_regex(index_pattern)
    date_re = re.compile(r"\d{8}")
    index = None
    index_pos = None

    for i, part in enumerate(remaining):  # last match wins
        m = idx_re.fullmatch(part)
        if m and not date_re.fullmatch(part):
            index = int(m.group(1))
            index_pos = i

    rest = "_".join(p for i, p in enumerate(remaining) if i != index_pos)
    return {"mouse": mouse, "date": date, "index": index, "rest": rest}


def get_files(
    directory: str | Path,
    pattern: str | None = None,
    extension: str = "tif",
    index_pattern: str | None = DEFAULT_INDEX_PATTERN,
    verbose: bool = False,
) -> list[tuple[str, int | None]]:
    """
    Find files in a directory, sorted by their filename index.

    Parameters
    ----------
    directory : str or Path
    pattern : str, optional
        Glob pattern (e.g. ``"*.tif"``). If ``None``, uses *extension*.
    extension : str, optional
        File extension without the dot. ``"tif"`` (default) matches both
        ``.tif`` and ``.tiff``.
    index_pattern : str or None, optional
        Regex for index extraction (see :func:`parse_filename`). Files without
        an index are dropped. If ``None``, all files are returned with index
        ``None``, sorted by name.
    verbose : bool, optional
        Print the file count.

    Returns
    -------
    list of (str, int or None)
        ``(filepath, index)`` sorted by index, then name.

    Examples
    --------
    >>> get_files("data/HK13/HK13_20260814")
    [('data/HK13/HK13_20260814/HK13_20260814_50pw_00002.tif', 2),
     ('data/HK13/HK13_20260814/HK13_20260814_90pw_00006.tif', 6)]
    """
    directory = str(directory)
    if pattern is not None:
        patterns = [os.path.join(directory, pattern)]
    elif extension.lower() in ("tif", "tiff"):
        patterns = [os.path.join(directory, "*.tif"), os.path.join(directory, "*.tiff")]
    else:
        patterns = [os.path.join(directory, f"*.{extension}")]

    matched = sorted({f for p in patterns for f in glob.glob(p)})
    if not matched:
        raise FileNotFoundError(f"No files found matching: {', '.join(patterns)}")

    if index_pattern is None:
        files: list[tuple[str, int | None]] = [(f, None) for f in matched]
    else:
        files = []
        for filepath in matched:
            idx = parse_filename(filepath, index_pattern=index_pattern)["index"]
            if idx is not None:
                files.append((filepath, idx))
        if not files:
            raise ValueError(f"No files in {directory} matched index_pattern '{index_pattern}'")
        files.sort(key=lambda x: (x[1], x[0]))

    if verbose:
        print(f"Found {len(files)} files")
    return files


def get_dirs(parent_dir: str | Path) -> list[tuple[str, str, str]]:
    """
    Find session directories matching the ``<MOUSE>_<YYYYMMDD>`` convention.

    Returns ``(path, date, mouse)`` tuples sorted by date then mouse.

    Examples
    --------
    >>> get_dirs("data/HK13")
    [('data/HK13/HK13_20260814', '20260814', 'HK13')]
    """
    result = []
    for entry in os.scandir(str(parent_dir)):
        if not entry.is_dir():
            continue
        parsed = parse_filename(entry.name)
        if parsed["mouse"] is not None and parsed["date"] is not None:
            result.append((entry.path, parsed["date"], parsed["mouse"]))
    result.sort(key=lambda x: (x[1], x[2]))
    return result


def get_subdir(data_dir: str, subdir: str | None = None) -> str:
    """Join *subdir* onto *data_dir*; returns *data_dir* unchanged if *subdir* is ``None``."""
    if subdir is None:
        return data_dir
    return os.path.join(data_dir, subdir)
