"""
Pixel-to-time mapping for resonant and linear scan geometries.

The banding artifact that ``bandaid`` removes is periodic in *time*. Under
resonant scanning the mirror velocity is sinusoidal, so pixels that are uniform
in space are not uniform in time: pixels near the left and right edges of a line
integrate over more ADC samples than pixels at the center. A temporal ripple
therefore appears as bands that get narrower toward the edges of the image.

ScanImage records the number of ADC samples per pixel in ``SI.hScan2D.mask``.
:class:`ScanGeometry` reconstructs the acquisition time of every pixel from that
mask, or from the fill fraction when the mask is unavailable, so that the ripple
can be modelled as a plain harmonic series in scan time.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

__all__ = ["ScanGeometry"]


def _as_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=float).ravel()
        if arr.size == 0:
            return default
        value = arr[0]
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return default if np.isnan(out) else out


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes")
    return bool(value)


def _as_mask(value: Any) -> np.ndarray | None:
    """Coerce a ScanImage mask (list, nested list, MATLAB string, array) to a flat int array."""
    if value is None:
        return None
    if isinstance(value, str):
        from tifffile import matlabstr2py

        value = matlabstr2py(value)
    arr = np.asarray(value, dtype=float).ravel()
    if arr.size == 0:
        return None
    return np.rint(arr).astype(np.int64)


@dataclass(frozen=True, eq=False)
class ScanGeometry:
    """
    Timing of every pixel within a frame.

    Parameters
    ----------
    n_pixels : int
        Pixels per line (image width).
    n_lines : int
        Lines per frame (image height).
    line_period_s : float
        Time between the starts of consecutive lines.
    frame_period_s : float, optional
        Time between the starts of consecutive frames (includes flyback).
    sample_rate_hz : float, optional
        ADC sample rate. Needed to convert *mask* to seconds.
    mask : array-like, optional
        ADC samples per pixel (``SI.hScan2D.mask``), length *n_pixels*.
    scanner_frequency_hz : float, optional
        Resonant scanner frequency.
    fill_fraction_spatial, fill_fraction_temporal : float, optional
        ScanImage fill fractions. ``FFs = sin(pi * FFt / 2)``.
    bidirectional : bool
        Odd lines are scanned in the opposite direction.
    uniform_sampling : bool
        Pixels are uniform in time rather than space.
    line_phase_s : float
        ScanImage line phase. Stored for reference; a constant time offset is
        absorbed by the per-frame ripple phase and does not affect the model.
    scan_mode : str
        ``'resonant'`` or ``'linear'``.
    n_channels : int
        Number of channels saved (pages per frame).
    imaging_system : str, optional
        Name of the imaging system, for reports.
    """

    n_pixels: int
    n_lines: int
    line_period_s: float
    frame_period_s: float | None = None
    sample_rate_hz: float | None = None
    mask: np.ndarray | None = field(default=None, repr=False)
    scanner_frequency_hz: float | None = None
    fill_fraction_spatial: float | None = None
    fill_fraction_temporal: float | None = None
    bidirectional: bool = False
    uniform_sampling: bool = False
    line_phase_s: float = 0.0
    scan_mode: str = "resonant"
    n_channels: int = 1
    imaging_system: str | None = None

    # ------------------------------------------------------------------ init

    def __post_init__(self) -> None:
        if self.n_pixels <= 0 or self.n_lines <= 0:
            raise ValueError("n_pixels and n_lines must be positive")
        if self.line_period_s <= 0:
            raise ValueError("line_period_s must be positive")
        mask = _as_mask(self.mask)
        if mask is not None and mask.size != self.n_pixels:
            raise ValueError(
                f"mask has {mask.size} entries but n_pixels is {self.n_pixels}"
            )
        object.__setattr__(self, "mask", mask)
        ffs, fft = self.fill_fraction_spatial, self.fill_fraction_temporal
        if ffs is None and fft is not None:
            ffs = float(np.sin(np.pi * fft / 2))
        if fft is None and ffs is not None:
            fft = float(2 / np.pi * np.arcsin(ffs))
        object.__setattr__(self, "fill_fraction_spatial", ffs)
        object.__setattr__(self, "fill_fraction_temporal", fft)
        object.__setattr__(self, "scan_mode", str(self.scan_mode).lower())

    # -------------------------------------------------------- constructors

    @classmethod
    def from_si_params(cls, si: dict, **overrides: Any) -> ScanGeometry:
        """Build from a parsed ScanImage ``SI.*`` parameter dict."""
        g = si.get
        channel_save = g("SI.hChannels.channelSave", 1)
        n_channels = len(np.atleast_1d(channel_save))
        frame_period = _as_float(g("SI.hRoiManager.scanFramePeriod"))
        if frame_period is None:
            rate = _as_float(g("SI.hRoiManager.scanFrameRate"))
            frame_period = 1.0 / rate if rate else None
        kwargs: dict[str, Any] = dict(
            n_pixels=int(_as_float(g("SI.hRoiManager.pixelsPerLine"))),
            n_lines=int(_as_float(g("SI.hRoiManager.linesPerFrame"))),
            line_period_s=_as_float(g("SI.hRoiManager.linePeriod")),
            frame_period_s=frame_period,
            sample_rate_hz=_as_float(g("SI.hScan2D.sampleRate")),
            mask=g("SI.hScan2D.mask"),
            scanner_frequency_hz=_as_float(g("SI.hScan2D.scannerFrequency")),
            fill_fraction_spatial=_as_float(g("SI.hScan2D.fillFractionSpatial")),
            fill_fraction_temporal=_as_float(g("SI.hScan2D.fillFractionTemporal")),
            bidirectional=_as_bool(g("SI.hScan2D.bidirectional")),
            uniform_sampling=_as_bool(g("SI.hScan2D.uniformSampling")),
            line_phase_s=_as_float(g("SI.hScan2D.linePhase"), 0.0),
            scan_mode=str(g("SI.hScan2D.scanMode", "resonant")),
            n_channels=n_channels,
            imaging_system=g("SI.imagingSystem"),
        )
        kwargs.update(overrides)
        return cls(**kwargs)

    @classmethod
    def from_session_metadata(cls, meta: dict, file_index: int = 0, **overrides: Any) -> ScanGeometry:
        """Build from a ``session_metadata.json`` dict (format version 2 or later)."""
        si = meta.get("si_params")
        if not si:
            raise ValueError(
                "session metadata has no 'si_params'; regenerate it with "
                "bandaid.io.save_session_metadata (format_version >= 2)"
            )
        si = dict(si)
        files = meta.get("files") or []
        if 0 <= file_index < len(files):
            si.update(files[file_index].get("si_overrides") or {})
        return cls.from_si_params(si, **overrides)

    @classmethod
    def uniform(
        cls,
        n_pixels: int,
        n_lines: int,
        line_period_s: float,
        frame_period_s: float | None = None,
        acq_fraction: float = 1.0,
        bidirectional: bool = False,
        **kwargs: Any,
    ) -> ScanGeometry:
        """Geometry with pixels uniform in time (galvo scanning or unknown hardware)."""
        return cls(
            n_pixels=n_pixels,
            n_lines=n_lines,
            line_period_s=line_period_s,
            frame_period_s=frame_period_s,
            fill_fraction_spatial=acq_fraction,
            fill_fraction_temporal=acq_fraction,
            bidirectional=bidirectional,
            uniform_sampling=True,
            scan_mode="linear",
            **kwargs,
        )

    def replace(self, **changes: Any) -> ScanGeometry:
        return replace(self, **changes)

    # ------------------------------------------------------------ derived

    @property
    def is_resonant(self) -> bool:
        return self.scan_mode == "resonant" and not self.uniform_sampling

    @property
    def n_flyback_lines(self) -> int:
        """Line periods per frame not covered by image lines (flyback / flyto)."""
        if self.frame_period_s is None:
            return 0
        return max(int(round(self.frame_period_s / self.line_period_s)) - self.n_lines, 0)

    @property
    def lines_per_frame_period(self) -> int:
        """Total line periods per frame period, including flyback."""
        return self.n_lines + self.n_flyback_lines

    @property
    def line_acq_time_s(self) -> float:
        """Duration of the acquisition window of one line."""
        if self.mask is not None and self.sample_rate_hz:
            return float(self.mask.sum() / self.sample_rate_hz)
        if self.is_resonant and self.scanner_frequency_hz and self.fill_fraction_temporal:
            # one line = half a resonant period; the acquired part is FFt of it
            return self.fill_fraction_temporal / (2 * self.scanner_frequency_hz)
        frac = self.fill_fraction_temporal or self.fill_fraction_spatial or 1.0
        return frac * self.line_period_s

    @property
    def effective_sample_rate_hz(self) -> float:
        """Real sample rate, or a virtual one (100 samples per center pixel) when unknown."""
        if self.sample_rate_hz:
            return float(self.sample_rate_hz)
        return 100.0 * self.n_pixels / self.line_acq_time_s

    def _u(self) -> np.ndarray:
        """Normalised scanner position of each pixel center, in (-FFs, FFs)."""
        ffs = self.fill_fraction_spatial
        if ffs is None:
            raise ValueError(
                "resonant geometry needs fill_fraction_spatial or fill_fraction_temporal"
            )
        x = np.arange(self.n_pixels, dtype=float)
        return ffs * (2 * (x + 0.5) / self.n_pixels - 1)

    def dwell_samples(self) -> np.ndarray:
        """ADC samples integrated per pixel, shape ``(n_pixels,)`` (float)."""
        if self.mask is not None:
            return self.mask.astype(float)
        fs = self.effective_sample_rate_hz
        if not self.is_resonant:
            return np.full(self.n_pixels, fs * self.line_acq_time_s / self.n_pixels)
        if not self.scanner_frequency_hz:
            raise ValueError("resonant geometry without mask needs scanner_frequency_hz")
        u = self._u()
        ffs = self.fill_fraction_spatial
        dt_dx = (2 * ffs / self.n_pixels) / (2 * np.pi * self.scanner_frequency_hz * np.sqrt(1 - u**2))
        return fs * dt_dx

    def pixel_times(self) -> np.ndarray:
        """
        Acquisition time of each pixel center, in seconds from the start of the
        line's acquisition window. Shape ``(n_pixels,)``; strictly increasing.
        """
        if self.mask is not None:
            m = self.mask.astype(float)
            centers = np.cumsum(m) - m / 2
            if self.sample_rate_hz:
                return centers / self.sample_rate_hz
            return centers / m.sum() * self.line_acq_time_s
        if self.is_resonant:
            if not self.scanner_frequency_hz:
                raise ValueError("resonant geometry without mask needs scanner_frequency_hz")
            u = self._u()
            w = 2 * np.pi * self.scanner_frequency_hz
            return (np.arcsin(u) + np.arcsin(self.fill_fraction_spatial)) / w
        x = np.arange(self.n_pixels, dtype=float)
        return (x + 0.5) / self.n_pixels * self.line_acq_time_s

    def pixel_times_2d(self) -> np.ndarray:
        """
        Acquisition time of every pixel from the start of the frame, shape
        ``(n_lines, n_pixels)``. Odd lines are time-reversed for bidirectional scans.
        """
        t = self.pixel_times()
        rows = np.arange(self.n_lines, dtype=float)[:, None] * self.line_period_s
        out = rows + t[None, :]
        if self.bidirectional:
            out[1::2] = rows[1::2] + (self.line_acq_time_s - t)[None, :]
        return out

    def integration_gain(self, f_hz: float) -> np.ndarray:
        """
        Attenuation of a sinusoid at *f_hz* caused by averaging each pixel over
        its dwell samples (Dirichlet kernel), shape ``(n_pixels,)``.
        """
        fs = self.effective_sample_rate_hz
        m = self.dwell_samples()
        a = np.pi * f_hz / fs
        if a == 0:
            return np.ones(self.n_pixels)
        with np.errstate(divide="ignore", invalid="ignore"):
            gain = np.abs(np.sin(m * a) / (m * np.sin(a)))
        return np.nan_to_num(gain, nan=1.0)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "n_pixels": self.n_pixels,
            "n_lines": self.n_lines,
            "line_period_s": self.line_period_s,
            "frame_period_s": self.frame_period_s,
            "sample_rate_hz": self.sample_rate_hz,
            "has_mask": self.mask is not None,
            "scanner_frequency_hz": self.scanner_frequency_hz,
            "fill_fraction_spatial": self.fill_fraction_spatial,
            "fill_fraction_temporal": self.fill_fraction_temporal,
            "bidirectional": self.bidirectional,
            "uniform_sampling": self.uniform_sampling,
            "line_phase_s": self.line_phase_s,
            "scan_mode": self.scan_mode,
            "n_channels": self.n_channels,
            "imaging_system": self.imaging_system,
            "n_flyback_lines": self.n_flyback_lines,
            "line_acq_time_s": self.line_acq_time_s,
        }
        return d
