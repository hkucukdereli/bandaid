"""
TIFF pixel I/O: whole-stack loading, chunked streaming, and dtype-safe writing.

ScanImage metadata parsing lives in :mod:`bandaid.io.scanimage`. tifffile reads
ScanImage BigTIFFs directly, so no other reader is needed.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from tifffile import TiffFile, TiffWriter

from bandaid._util import cast_like, package_version

__all__ = [
    "open_tiff",
    "load_tiff_stack",
    "iter_frames",
    "save_tiff",
    "TiffChunkWriter",
    "n_channels_in_tiff",
    "get_tiff_reader",
]

log = logging.getLogger("bandaid.io")

BIGTIFF_THRESHOLD_BYTES = 3_500_000_000
DEFAULT_MAX_CHUNK_BYTES = 256_000_000
MAX_CHUNK_FRAMES = 1024


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def n_channels_in_tiff(tif: TiffFile) -> int:
    """Number of interleaved channels per frame (from ScanImage metadata; 1 otherwise)."""
    if not tif.is_scanimage:
        return 1
    try:
        from bandaid.io.scanimage import n_channels_from_si

        return n_channels_from_si((tif.scanimage_metadata or {}).get("FrameData") or {})
    except Exception:  # pragma: no cover - defensive
        return 1


@contextmanager
def open_tiff(source: str | Path | TiffFile) -> Iterator[TiffFile]:
    """Context manager yielding a :class:`tifffile.TiffFile`; accepts a path or an open file."""
    if isinstance(source, TiffFile):
        yield source
        return
    with TiffFile(str(source)) as tif:
        yield tif


def _frame_range(n_frames: int, frames: range | slice | None) -> range:
    if frames is None:
        return range(n_frames)
    if isinstance(frames, slice):
        rng = range(*frames.indices(n_frames))
    elif isinstance(frames, range):
        rng = range(max(frames.start, 0), min(frames.stop, n_frames), frames.step)
    else:
        raise TypeError("frames must be None, a slice, or a range")
    if rng.step <= 0:
        raise ValueError("frames must have a positive step")
    return rng


def _page_keys(frames: range, n_ch: int, channel: int | None) -> list[int]:
    if n_ch == 1:
        return list(frames)
    if channel is None:
        return [f * n_ch + c for f in frames for c in range(n_ch)]
    if not 0 <= channel < n_ch:
        raise ValueError(f"channel {channel} out of range for a {n_ch}-channel file")
    return [f * n_ch + channel for f in frames]


def _read_pages(tif: TiffFile, frames: range, n_ch: int, channel: int | None) -> np.ndarray:
    page0 = tif.pages[0]
    keys = _page_keys(frames, n_ch, channel)
    if not keys:
        shape = (0, n_ch, *page0.shape) if (n_ch > 1 and channel is None) else (0, *page0.shape)
        return np.empty(shape, dtype=page0.dtype)
    out = tif.asarray(key=keys)
    if out.ndim == 2:
        out = out[None]
    if n_ch > 1 and channel is None:
        out = out.reshape(len(frames), n_ch, *out.shape[-2:])
    return out


def load_tiff_stack(
    filepath: str | Path | TiffFile,
    frames: range | slice | None = None,
    channel: int | None = None,
    **legacy,
) -> np.ndarray:
    """
    Load a TIFF stack (or part of it) into memory.

    Parameters
    ----------
    filepath : str, Path, or TiffFile
    frames : range or slice, optional
        Frames to load (default: all).
    channel : int, optional
        For multi-channel ScanImage files, the channel to load. If ``None`` and
        the file has several channels, the result has shape ``(T, C, H, W)``.

    Returns
    -------
    numpy.ndarray
        Shape ``(T, H, W)`` (a single-frame file gives ``T == 1``).

    Examples
    --------
    >>> mov = load_tiff_stack("trial.tif")
    >>> first = load_tiff_stack("trial.tif", frames=range(0, 100))
    """
    if legacy:
        warnings.warn(
            f"load_tiff_stack: ignoring deprecated arguments {sorted(legacy)}",
            DeprecationWarning,
            stacklevel=2,
        )
    with open_tiff(filepath) as tif:
        n_ch = n_channels_in_tiff(tif)
        n_frames = len(tif.pages) // n_ch
        if frames is None and n_ch == 1:
            out = tif.asarray()
            return out[None] if out.ndim == 2 else out
        return _read_pages(tif, _frame_range(n_frames, frames), n_ch, channel)


def iter_frames(
    source: str | Path | TiffFile,
    chunk_frames: int | None = None,
    max_chunk_bytes: int = DEFAULT_MAX_CHUNK_BYTES,
    frames: range | slice | None = None,
    channel: int | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """
    Stream a TIFF stack in chunks of frames without loading it whole.

    Yields ``(start_frame, chunk)`` with ``chunk`` of shape ``(n, H, W)``.
    The chunk size defaults to about *max_chunk_bytes* of pixel data.

    Examples
    --------
    >>> for start, chunk in iter_frames("trial.tif", chunk_frames=500):
    ...     process(chunk)
    """
    with open_tiff(source) as tif:
        n_ch = n_channels_in_tiff(tif)
        page0 = tif.pages[0]
        n_frames = len(tif.pages) // n_ch
        rng = _frame_range(n_frames, frames)
        if chunk_frames is None:
            per_frame = page0.nbytes * (n_ch if channel is None else 1)
            chunk_frames = int(max(1, min(MAX_CHUNK_FRAMES, max_chunk_bytes // max(per_frame, 1))))
        idx = list(rng)
        for s in range(0, len(idx), chunk_frames):
            sub = idx[s : s + chunk_frames]
            sub_rng = range(sub[0], sub[-1] + rng.step, rng.step)
            yield sub[0], _read_pages(tif, sub_rng, n_ch, channel)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

class TiffChunkWriter:
    """
    Write a ``(T, H, W)`` stack to a TIFF one chunk at a time.

    Frames are cast to *dtype* with rounding and clipping (never wraparound).
    BigTIFF is used when the expected size is unknown or exceeds ~3.5 GB.

    Examples
    --------
    >>> with TiffChunkWriter("out.tif", dtype="int16") as w:
    ...     for start, chunk in iter_frames("in.tif"):
    ...         w.write_chunk(chunk.astype("float32") - 3.2)
    """

    def __init__(
        self,
        path: str | Path,
        dtype,
        bigtiff: bool | None = None,
        expected_nbytes: int | None = None,
        software: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.dtype = np.dtype(dtype)
        if bigtiff is None:
            bigtiff = expected_nbytes is None or expected_nbytes > BIGTIFF_THRESHOLD_BYTES
        self.bigtiff = bool(bigtiff)
        self.software = software or f"bandaid {package_version()}"
        self._tif: TiffWriter | None = None
        self.n_frames = 0

    def __enter__(self) -> TiffChunkWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._tif = TiffWriter(str(self.path), bigtiff=self.bigtiff)
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._tif is not None:
            self._tif.close()
            self._tif = None

    def write_chunk(self, chunk: np.ndarray) -> None:
        if self._tif is None:
            raise RuntimeError("TiffChunkWriter must be used as a context manager")
        chunk = np.asarray(chunk)
        if chunk.ndim == 2:
            chunk = chunk[None]
        if chunk.ndim != 3:
            raise ValueError(f"expected (n, H, W) chunk, got shape {chunk.shape}")
        if chunk.shape[0] == 0:
            return
        chunk = np.ascontiguousarray(cast_like(chunk, self.dtype))
        kwargs = dict(photometric="minisblack", contiguous=True, metadata=None)
        if self.n_frames == 0:
            kwargs["software"] = self.software
        self._tif.write(chunk, **kwargs)
        self.n_frames += chunk.shape[0]


def save_tiff(
    data: np.ndarray,
    path: str | Path,
    dtype=None,
    bigtiff: bool | None = None,
    chunk_frames: int | None = None,
    software: str | None = None,
) -> Path:
    """
    Save a ``(T, H, W)`` or ``(H, W)`` array to a TIFF.

    The input dtype is kept unless *dtype* is given. Integer targets are
    rounded and clipped rather than truncated or wrapped.

    Returns the output path.
    """
    data = np.asarray(data)
    if data.ndim == 2:
        data = data[None]
    if data.ndim != 3:
        raise ValueError(f"expected (T, H, W) or (H, W), got shape {data.shape}")
    target = np.dtype(dtype) if dtype is not None else data.dtype
    nbytes = int(np.prod(data.shape)) * target.itemsize
    if chunk_frames is None:
        chunk_frames = int(max(1, DEFAULT_MAX_CHUNK_BYTES // max(data[0].nbytes, 1)))
    with TiffChunkWriter(path, target, bigtiff=bigtiff, expected_nbytes=nbytes, software=software) as w:
        for s in range(0, data.shape[0], chunk_frames):
            w.write_chunk(data[s : s + chunk_frames])
    return Path(path)


# ---------------------------------------------------------------------------
# Deprecated
# ---------------------------------------------------------------------------

def get_tiff_reader(filepath: str | Path, **legacy) -> TiffFile:
    """Deprecated: use :func:`open_tiff` or :class:`tifffile.TiffFile` directly."""
    warnings.warn(
        "get_tiff_reader is deprecated; use open_tiff() or tifffile.TiffFile",
        DeprecationWarning,
        stacklevel=2,
    )
    return TiffFile(str(filepath))
