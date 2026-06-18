"""
Simple destriping for banding artifacts in calcium imaging data.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d


def destripe_edge(
    data: np.ndarray,
    n_edge: int = 10,
    axis: int = 0,
    method: str = 'mean',
    percentile: float = None,
    preserve_mean: bool = True
) -> np.ndarray:
    """
    Remove banding by subtracting the profile from edge rows/columns.

    Parameters
    ----------
    data : np.ndarray
        Input data, shape (frames, height, width) or (height, width)
    n_edge : int
        Number of edge rows/columns to use. Default 10.
    axis : int
        0 = use top N rows to get stripe profile (for vertical bands)
        1 = use left N columns to get stripe profile (for horizontal bands)
    method : str
        'mean' or 'median'. Ignored if percentile is set. Default 'mean'.
    percentile : float, optional
        If set, use this percentile (0-100) instead of mean/median.
    preserve_mean : bool
        If True, preserve original frame mean intensity. Default True.

    Returns
    -------
    np.ndarray
        Corrected data, same shape and dtype as input
    """
    is_single = data.ndim == 2
    if is_single:
        data = data[np.newaxis,:,:] # if it's a single frame

    n_frames = data.shape[0]
    result = np.empty_like(data, dtype=np.float32)

    for i in range(n_frames):
        frame = data[i].astype(np.float32)
        orig_mean = frame.mean() if preserve_mean else 0

        if axis == 0:
            # Vertical bands: use top rows, subtract column profile
            edge_region = frame[:n_edge, :]
            if percentile is not None:
                profile = np.percentile(edge_region, percentile, axis=0)
            elif method == 'median':
                profile = np.median(edge_region, axis=0)
            else:
                profile = np.mean(edge_region, axis=0)
            corrected = frame - profile[np.newaxis, :]
        else:
            # Horizontal bands: use left columns, subtract row profile
            edge_region = frame[:, :n_edge]
            if percentile is not None:
                profile = np.percentile(edge_region, percentile, axis=1)
            elif method == 'median':
                profile = np.median(edge_region, axis=1)
            else:
                profile = np.mean(edge_region, axis=1)
            corrected = frame - profile[:, np.newaxis]

        if preserve_mean:
            corrected += orig_mean - corrected.mean()

        result[i] = corrected

    if is_single:
        return result[0]
    return result


def destripe_edge_highpass(
    data: np.ndarray,
    n_edge: int = 10,
    axis: int = 0,
    method: str = 'median',
    highpass_size: int = 30,
    preserve_mean: bool = True
) -> np.ndarray:
    """
    Remove banding using high-pass filtered edge profile.

    Same as destripe_edge, but applies a high-pass filter to the edge profile
    to remove low-frequency gradients (uneven illumination) and keep only
    the stripe pattern.

    Parameters
    ----------
    data : np.ndarray
        Input data, shape (frames, height, width) or (height, width)
    n_edge : int
        Number of edge rows/columns to use. Default 10.
    axis : int
        0 = use top N rows to get stripe profile (for vertical bands)
        1 = use left N columns to get stripe profile (for horizontal bands)
    method : str
        'mean' or 'median' for computing edge profile. Default 'median'.
    highpass_size : int
        Size of the smoothing window for high-pass filter. Larger = removes
        lower frequencies. Default 30.
    preserve_mean : bool
        If True, preserve original frame mean intensity. Default True.

    Returns
    -------
    np.ndarray
        Corrected data, same shape and dtype as input
    """
    is_single = data.ndim == 2
    if is_single:
        data = data[np.newaxis, ...]

    n_frames = data.shape[0]
    result = np.empty_like(data, dtype=np.float32)
    func = np.median if method == 'median' else np.mean

    for i in range(n_frames):
        frame = data[i].astype(np.float32)
        orig_mean = frame.mean() if preserve_mean else 0

        if axis == 0:
            # Vertical bands: use top rows, get column profile
            profile = func(frame[:n_edge, :], axis=0)
            # High-pass: subtract smoothed version
            lowpass = uniform_filter1d(profile, size=highpass_size, mode='nearest')
            highpass_profile = profile - lowpass
            corrected = frame - highpass_profile[np.newaxis, :]
        else:
            # Horizontal bands: use left columns, get row profile
            profile = func(frame[:, :n_edge], axis=1)
            lowpass = uniform_filter1d(profile, size=highpass_size, mode='nearest')
            highpass_profile = profile - lowpass
            corrected = frame - highpass_profile[:, np.newaxis]

        if preserve_mean:
            corrected += orig_mean - corrected.mean()

        result[i] = corrected

    if is_single:
        return result[0].astype(data.dtype)
    return result.astype(data.dtype)


def destripe_per_column(
    data: np.ndarray,
    method: str = 'median',
    percentile: float = None,
    preserve_mean: bool = True
) -> np.ndarray:
    """
    Remove vertical banding by subtracting each column's median/mean/percentile.

    For vertical bands (running top-to-bottom), the noise is constant
    along each column. This subtracts the column statistic independently.

    Parameters
    ----------
    data : np.ndarray
        Input data, shape (frames, height, width) or (height, width)
    method : str
        'median' or 'mean'. Ignored if percentile is set.
    percentile : float, optional
        If set, use this percentile (0-100) instead of median/mean.
        Lower values (e.g., 25) avoid bright signal contamination.
    preserve_mean : bool
        If True, preserve original frame mean intensity. Default True.

    Returns
    -------
    np.ndarray
        Corrected data, same shape and dtype as input
    """
    is_single = data.ndim == 2
    if is_single:
        data = data[np.newaxis, ...]

    n_frames = data.shape[0]
    result = np.empty_like(data, dtype=np.float32)

    for i in range(n_frames):
        frame = data[i].astype(np.float32)
        orig_mean = frame.mean() if preserve_mean else 0

        # Get each column's statistic (across rows)
        if percentile is not None:
            col_profile = np.percentile(frame, percentile, axis=0)
        elif method == 'median':
            col_profile = np.median(frame, axis=0)
        else:
            col_profile = np.mean(frame, axis=0)

        corrected = frame - col_profile[np.newaxis, :]

        if preserve_mean:
            corrected += orig_mean - corrected.mean()

        result[i] = corrected

    if is_single:
        return result[0].astype(data.dtype)
    return result.astype(data.dtype)


def destripe_per_row(
    data: np.ndarray,
    method: str = 'median',
    preserve_mean: bool = True
) -> np.ndarray:
    """
    Remove horizontal banding by subtracting each row's median/mean.

    For horizontal bands (running left-to-right), the noise is constant
    along each row. This subtracts the row median/mean independently.

    Parameters
    ----------
    data : np.ndarray
        Input data, shape (frames, height, width) or (height, width)
    method : str
        'median' or 'mean'. Median is more robust to outliers.
    preserve_mean : bool
        If True, preserve original frame mean intensity. Default True.

    Returns
    -------
    np.ndarray
        Corrected data, same shape and dtype as input
    """
    is_single = data.ndim == 2
    if is_single:
        data = data[np.newaxis, ...]

    n_frames = data.shape[0]
    result = np.empty_like(data, dtype=np.float32)

    func = np.median if method == 'median' else np.mean

    for i in range(n_frames):

        frame = data[i].astype(np.float32)
        orig_mean = frame.mean() if preserve_mean else 0

        # Get each row's median/mean (across columns)
        row_profile = func(frame, axis=1)  # shape: (height,)
        corrected = frame - row_profile[:, np.newaxis]

        if preserve_mean:
            corrected += orig_mean - corrected.mean()

        result[i] = corrected

    if is_single:
        return result[0].astype(data.dtype)
    return result.astype(data.dtype)
