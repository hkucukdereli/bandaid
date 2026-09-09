"""
Spatially blind destriping baselines: subtract a per-frame column (or row)
profile from every frame.

These do not know the scan geometry, so they also remove any real image
structure that is constant along a column (or row) in a frame, and they cannot
follow the edge-to-center change in stripe width. Prefer
:func:`bandaid.ripple.remove_ripple` for ScanImage data; keep these for quick
looks and comparisons.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.ndimage import uniform_filter1d

from bandaid._util import cast_like

__all__ = [
    "destripe_profile",
    "destripe_edge",
    "destripe_edge_highpass",
    "destripe_per_column",
    "destripe_per_row",
]


def _profile(region: np.ndarray, stat, axis: int) -> np.ndarray:
    if isinstance(stat, str):
        if stat == "mean":
            return region.mean(axis=axis)
        if stat == "median":
            return np.median(region, axis=axis)
        raise ValueError(f"stat must be 'mean', 'median' or a percentile (0-100), got {stat!r}")
    q = float(stat)
    if not 0 <= q <= 100:
        raise ValueError(f"percentile must be within 0..100, got {q}")
    return np.percentile(region, q, axis=axis)


def destripe_profile(
    data: np.ndarray,
    axis: str = "columns",
    stat="median",
    n_edge: int | None = None,
    highpass_size: int | None = None,
    preserve_mean: bool = True,
    dtype="same",
) -> np.ndarray:
    """
    Remove banding by subtracting a per-frame profile.

    Parameters
    ----------
    data : np.ndarray
        ``(T, H, W)`` stack or a single ``(H, W)`` frame.
    axis : ``'columns'`` or ``'rows'``
        ``'columns'`` removes vertical bands (one value per column),
        ``'rows'`` removes horizontal bands (one value per row).
    stat : ``'mean'``, ``'median'`` or a percentile in 0..100
        Statistic used to estimate the profile. Low percentiles avoid bright
        signal contamination.
    n_edge : int, optional
        Estimate the profile from only the first *n_edge* rows (for
        ``'columns'``) or columns (for ``'rows'``), e.g. a signal-free margin.
    highpass_size : int, optional
        Subtract a running mean of this many samples from the profile so that
        slow illumination gradients are kept and only the stripes are removed.
    preserve_mean : bool
        Restore each frame's mean intensity after subtraction.
    dtype : ``'same'`` or dtype
        Output dtype. Integer targets are rounded and clipped (no wraparound).

    Returns
    -------
    np.ndarray
        Same shape as *data*.
    """
    if axis not in ("columns", "rows"):
        raise ValueError("axis must be 'columns' or 'rows'")
    _profile(np.zeros((2, 2), np.float32), stat, 0)  # validate stat early

    data = np.asarray(data)
    single = data.ndim == 2
    if single:
        data = data[None]
    if data.ndim != 3:
        raise ValueError(f"expected (T, H, W) or (H, W), got shape {data.shape}")
    out_dtype = data.dtype if (isinstance(dtype, str) and dtype == "same") else np.dtype(dtype)
    work = np.empty(data.shape, dtype=np.float32)

    reduce_axis = 0 if axis == "columns" else 1
    for i in range(data.shape[0]):
        frame = data[i].astype(np.float32)
        if n_edge is None:
            region = frame
        elif axis == "columns":
            region = frame[:n_edge]
        else:
            region = frame[:, :n_edge]
        profile = _profile(region, stat, reduce_axis)
        if highpass_size:
            profile = profile - uniform_filter1d(profile, size=int(highpass_size), mode="nearest")
        corrected = frame - (profile[None, :] if axis == "columns" else profile[:, None])
        if preserve_mean:
            corrected += frame.mean() - corrected.mean()
        work[i] = corrected

    result = cast_like(work, out_dtype)
    return result[0] if single else result


# ---------------------------------------------------------------------------
# Deprecated names (thin wrappers)
# ---------------------------------------------------------------------------

def _deprecated(name: str) -> None:
    warnings.warn(
        f"{name} is deprecated; use destripe_profile() instead",
        DeprecationWarning,
        stacklevel=3,
    )


def _axis_name(axis: int) -> str:
    return "columns" if axis == 0 else "rows"


def destripe_edge(data, n_edge=10, axis=0, method="mean", percentile=None, preserve_mean=True):
    """Deprecated: ``destripe_profile(data, axis, stat, n_edge=n_edge)``."""
    _deprecated("destripe_edge")
    stat = percentile if percentile is not None else method
    return destripe_profile(data, _axis_name(axis), stat, n_edge=n_edge, preserve_mean=preserve_mean)


def destripe_edge_highpass(data, n_edge=10, axis=0, method="median", highpass_size=30, preserve_mean=True):
    """Deprecated: ``destripe_profile(..., n_edge=n_edge, highpass_size=highpass_size)``."""
    _deprecated("destripe_edge_highpass")
    return destripe_profile(data, _axis_name(axis), method, n_edge=n_edge,
                            highpass_size=highpass_size, preserve_mean=preserve_mean)


def destripe_per_column(data, method="median", percentile=None, preserve_mean=True):
    """Deprecated: ``destripe_profile(data, 'columns', stat)``."""
    _deprecated("destripe_per_column")
    stat = percentile if percentile is not None else method
    return destripe_profile(data, "columns", stat, preserve_mean=preserve_mean)


def destripe_per_row(data, method="median", preserve_mean=True):
    """Deprecated: ``destripe_profile(data, 'rows', stat)``."""
    _deprecated("destripe_per_row")
    return destripe_profile(data, "rows", method, preserve_mean=preserve_mean)
