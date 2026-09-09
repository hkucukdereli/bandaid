"""Small helpers shared across bandaid modules."""

from __future__ import annotations

import numpy as np


def package_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("bandaid")
    except PackageNotFoundError:  # pragma: no cover - source checkout without install
        return "0.0.0"


def cast_like(arr: np.ndarray, dtype) -> np.ndarray:
    """
    Cast *arr* to *dtype*. Integer targets are rounded to nearest and clipped to
    the representable range, so values never wrap around.
    """
    dtype = np.dtype(dtype)
    arr = np.asarray(arr)
    if arr.dtype == dtype:
        return arr
    if dtype.kind in "iu":
        info = np.iinfo(dtype)
        out = np.rint(arr)
        np.clip(out, info.min, info.max, out=out)
        return out.astype(dtype)
    return arr.astype(dtype)
