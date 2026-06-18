"""
File and directory utilities for lab naming conventions.
"""

import os
import re
import glob
from pathlib import Path
from typing import Optional, Union, List, Tuple


def parse_filename(filename: str, index_pattern: str = r'(\d{3,})') -> dict:
    """
    Break a lab filename into mouse, date, index, and leftover parts.

    Expects: ``<MOUSE>_<YYYYMMDD>_[extra_]<INDEX>[_extra].ext``
    where MOUSE is 2-4 letters optionally followed by digits.

    Parameters
    ----------
    filename : str
        Filename or full path. Path and extension are stripped.
    index_pattern : str, optional
        Regex to match the index segment. Applied to each
        underscore-separated part. Default matches 3+ digits.

    Returns
    -------
    dict
        ``'mouse'`` : str or None — e.g. ``'AMD4'``.
        ``'date'``  : str or None — e.g. ``'20260211'``.
        ``'index'`` : int or None — e.g. ``2``.
        ``'rest'``  : str — remaining parts joined with ``'_'``.

    Examples
    --------
    >>> parse_filename("AMD4_20260211_00002.tif")
    {'mouse': 'AMD4', 'date': '20260211', 'index': 2, 'rest': ''}

    >>> parse_filename("HK123_20260301_4364_tag_0001.tif")
    {'mouse': 'HK123', 'date': '20260301', 'index': 1, 'rest': '4364_tag'}
    """
    stem = Path(filename).stem
    parts = stem.split('_')

    mouse = None
    date = None
    rest_start = 0

    if parts and re.fullmatch(r'[A-Za-z]{2,4}\d*', parts[0]):
        mouse = parts[0]
        rest_start = 1

        if len(parts) > 1 and re.fullmatch(r'\d{8}', parts[1]):
            date = parts[1]
            rest_start = 2

    # Find the index among remaining parts (last match wins)
    remaining = parts[rest_start:]
    idx_re = re.compile(index_pattern)
    date_re = re.compile(r'\d{8}')
    index = None
    index_pos = None

    for i, part in enumerate(remaining):
        m = idx_re.fullmatch(part)
        if m and not date_re.fullmatch(part):
            index = int(m.group(1))
            index_pos = i

    rest_parts = [p for i, p in enumerate(remaining) if i != index_pos]
    rest = '_'.join(rest_parts)

    return {"mouse": mouse, "date": date, "index": index, "rest": rest}


def get_files(
    directory: str,
    pattern: Optional[str] = None,
    extension: str = "tif",
    index_pattern: Optional[str] = r'(\d{3,})',
    verbose: bool = False,
) -> Union[List[Tuple[str, int]], List[str]]:
    """
    Find files in a directory, optionally sorted by their filename index.

    Parameters
    ----------
    directory : str
        Path to the directory.
    pattern : str, optional
        Glob pattern (e.g. ``"*.tif"``). If ``None``, uses *extension*.
    extension : str, optional
        File extension without the dot. Default ``"tif"`` (matches
        both ``.tif`` and ``.tiff``).
    index_pattern : str or None, optional
        Regex for index extraction (see :func:`parse_filename`).
        If ``None``, returns a plain list of paths without sorting.
    verbose : bool, optional
        Print file count. Default ``False``.

    Returns
    -------
    list of (str, int) — ``(filepath, index)`` sorted by index, or
    list of str — file paths (unsorted) if *index_pattern* is ``None``.

    Examples
    --------
    >>> get_files("/data/AMD4_20260211")
    [('/data/.../AMD4_20260211_00002.tif', 2),
     ('/data/.../AMD4_20260211_00003.tif', 3),
     ('/data/.../AMD4_20260211_00004.tif', 4)]
    """
    if pattern is not None:
        patterns = [os.path.join(directory, pattern)]
    elif extension.lower() in ("tif", "tiff"):
        patterns = [
            os.path.join(directory, "*.tif"),
            os.path.join(directory, "*.tiff"),
        ]
    else:
        patterns = [os.path.join(directory, f"*.{extension}")]

    matched_files = []
    for p in patterns:
        matched_files.extend(glob.glob(p))

    if not matched_files:
        raise FileNotFoundError(
            f"No files found matching: {', '.join(patterns)}"
        )

    if index_pattern is None:
        if verbose:
            print(f"Found {len(matched_files)} files")
        return matched_files

    file_list = []
    for filepath in matched_files:
        parsed = parse_filename(filepath, index_pattern=index_pattern)
        if parsed["index"] is not None:
            file_list.append((filepath, parsed["index"]))

    if not file_list:
        raise ValueError(
            f"No files in {directory} matched index_pattern '{index_pattern}'"
        )

    file_list.sort(key=lambda x: x[1])

    if verbose:
        print(f"Found {len(file_list)} files")

    return file_list


def get_dirs(parent_dir: str) -> List[Tuple[str, str, str]]:
    """
    Find session directories matching the ``<MOUSE>_<YYYYMMDD>`` convention.

    Parameters
    ----------
    parent_dir : str
        Path to search.

    Returns
    -------
    list of (str, str, str)
        ``(path, date, mouse)`` tuples sorted by date then mouse.

    Examples
    --------
    >>> get_dirs("/data/AMD4")
    [('/data/AMD4/AMD4_20260211', '20260211', 'AMD4')]
    """
    result = []
    for entry in os.scandir(parent_dir):
        if not entry.is_dir():
            continue
        parsed = parse_filename(entry.name)
        if parsed["mouse"] is not None and parsed["date"] is not None:
            result.append((entry.path, parsed["date"], parsed["mouse"]))

    result.sort(key=lambda x: (x[1], x[2]))
    return result


def get_subdir(data_dir: str, subdir: Optional[str] = None) -> str:
    """
    Join a subdirectory onto a data path. Returns *data_dir* as-is if
    *subdir* is ``None``.

    Parameters
    ----------
    data_dir : str
        Parent directory.
    subdir : str or None, optional
        Subdirectory name (e.g. ``'suite2p'``). Default ``None``.

    Returns
    -------
    str
    """
    if subdir is None:
        return data_dir
    return os.path.join(data_dir, subdir)
