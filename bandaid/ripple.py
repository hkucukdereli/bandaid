"""
Detection and removal of moving periodic banding (PMT ripple) in scan time.

Model
-----
The artifact is an additive ripple that is periodic in *time* and sampled by
the scanner. For frame ``t``, row ``r`` and column ``x``::

    ripple(t, r, x) = sum_k A_k(t) g_k(x) cos(2 pi [k (f0 tau(x) + rho_t (r - r0))] + phi_k(t))

* ``tau(x)`` is the acquisition time of column ``x`` within a line
  (:meth:`ScanGeometry.pixel_times`). Under resonant scanning it is an arcsin of
  the column, which is why the bands get narrower toward the image edges.
* ``f0 = n / line_period``: the ripple frequency is a near-integer multiple ``n``
  of the line rate, which is what makes the bands vertical.
* ``rho_t`` is the small residual phase advance per line (cycles/row). It makes
  the pattern crawl across frames (``rho * lines_per_frame_period`` cycles per
  frame) and puts a slight phase ramp down each frame. It drifts slowly with the
  scanner, so it is tracked from the data.
* ``g_k`` is the attenuation from integrating each pixel over its dwell time.
* ``A_k(t)``, ``phi_k(t)`` are fitted per frame and harmonic.

Pipeline
--------
1. Reduce every frame to row-group mean profiles ``(T, G, W)`` (cheap, high SNR).
2. Detect: spectrum of the dynamic profiles in scan time, harmonic index ``n``,
   per-harmonic line power, phase advance, and a verdict.
3. Fit: per-frame complex coefficients by least squares in windows, with the
   row ramp; static image structure is the temporal mean of the profiles and is
   re-estimated once from the residual.
4. Remove: evaluate the model on the full frame grid and subtract it, streaming
   chunk by chunk so memory does not depend on movie length.
"""

from __future__ import annotations

import logging
import shutil
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from tifffile import TiffFile

from bandaid._util import cast_like, package_version
from bandaid.io.scanimage import write_json
from bandaid.io.tiff import TiffChunkWriter, iter_frames, load_tiff_stack, n_channels_in_tiff
from bandaid.scan_geometry import ScanGeometry

__all__ = [
    "RowGroups",
    "RippleDetection",
    "RippleFit",
    "RippleReport",
    "column_profiles",
    "ripple_basis",
    "tspace_spectrum",
    "line_power_ratios",
    "detect_ripple",
    "fit_ripple",
    "ripple_frames",
    "remove_ripple",
]

log = logging.getLogger("bandaid.ripple")


# ---------------------------------------------------------------------------
# Row groups and profiles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RowGroups:
    """Contiguous row blocks (split by parity for bidirectional scans)."""

    slices: tuple[slice, ...]
    centers: np.ndarray        # (G,) mean row index of each group
    parity: np.ndarray         # (G,) 0/1 scan direction of the group's rows
    counts: np.ndarray         # (G,) rows per group
    steps: np.ndarray          # (G,) row spacing within the group (1 or 2)
    n_lines: int
    bidirectional: bool

    @classmethod
    def from_geometry(cls, geometry: ScanGeometry, n_groups: int = 8) -> RowGroups:
        H = geometry.n_lines
        n_groups = int(max(1, min(n_groups, H // (2 if geometry.bidirectional else 1))))
        edges = np.linspace(0, H, n_groups + 1).astype(int)
        slices, centers, parity, counts, steps = [], [], [], [], []
        for a, b in zip(edges[:-1], edges[1:], strict=True):
            if geometry.bidirectional:
                for p in (0, 1):
                    start = a if a % 2 == p else a + 1
                    rows = np.arange(start, b, 2)
                    if rows.size:
                        slices.append(slice(start, b, 2))
                        centers.append(rows.mean())
                        parity.append(p)
                        counts.append(rows.size)
                        steps.append(2)
            else:
                rows = np.arange(a, b)
                if rows.size:
                    slices.append(slice(a, b))
                    centers.append(rows.mean())
                    parity.append(0)
                    counts.append(rows.size)
                    steps.append(1)
        return cls(tuple(slices), np.array(centers), np.array(parity), np.array(counts),
                   np.array(steps), H, geometry.bidirectional)

    @property
    def n(self) -> int:
        return len(self.slices)

    @property
    def n_parities(self) -> int:
        return 2 if self.bidirectional else 1

    def indices(self, parity: int) -> np.ndarray:
        return np.flatnonzero(self.parity == parity)

    @property
    def r0(self) -> float:
        return (self.n_lines - 1) / 2


def column_profiles(
    chunk: np.ndarray, groups: RowGroups, weights: np.ndarray | None = None
) -> np.ndarray:
    """
    Reduce frames ``(N, H, W)`` to row-group mean profiles ``(N, G, W)`` (float32).

    *weights* is an optional ``(H, W)`` array of per-pixel weights (e.g. 0 on
    cell bodies); columns whose weights vanish within a group fall back to the
    plain mean.
    """
    chunk = np.asarray(chunk)
    if chunk.ndim == 2:
        chunk = chunk[None]
    N, H, W = chunk.shape
    out = np.empty((N, groups.n, W), dtype=np.float32)
    for gi, sl in enumerate(groups.slices):
        block = chunk[:, sl]
        plain = block.mean(axis=1, dtype=np.float32)
        if weights is None:
            out[:, gi] = plain
            continue
        w = np.asarray(weights[sl], dtype=np.float32)
        wsum = w.sum(axis=0)
        weighted = (block.astype(np.float32) * w[None]).sum(axis=1) / np.maximum(wsum, 1e-6)[None]
        out[:, gi] = np.where(wsum[None] > 0.5, weighted, plain)
    return out


def auto_weights(mean_image: np.ndarray, percentile: float = 95.0) -> np.ndarray:
    """Binary weights excluding the brightest pixels (cell bodies) from the profile fit."""
    thr = np.percentile(mean_image, percentile)
    return (mean_image < thr).astype(np.float32)


# ---------------------------------------------------------------------------
# Basis and design
# ---------------------------------------------------------------------------

def ripple_basis(
    geometry: ScanGeometry, f_hz: float, n_harmonics: int, pixel_integration: bool = True
) -> np.ndarray:
    """
    Complex per-column basis ``b_k(x) = g_k(x) exp(2 pi i k f tau(x))``.

    Returns shape ``(P, K, W)`` with ``P = 2`` (even/odd line direction) for
    bidirectional scans, else ``P = 1``.
    """
    t2d = geometry.pixel_times_2d()
    taus = [t2d[0]]
    if geometry.bidirectional:
        taus.append(t2d[1] - geometry.line_period_s)
    out = np.empty((len(taus), n_harmonics, geometry.n_pixels), dtype=np.complex64)
    for p, tau in enumerate(taus):
        for k in range(1, n_harmonics + 1):
            gain = geometry.integration_gain(k * f_hz) if pixel_integration else 1.0
            out[p, k - 1] = gain * np.exp(2j * np.pi * k * f_hz * tau)
    return out


def _group_gain(k: np.ndarray, rho: float, count: int, step: int) -> np.ndarray:
    """Amplitude factor from averaging a ramped phase over *count* rows spaced *step*."""
    a = np.pi * k * rho * step
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.sin(a * count) / (count * np.sin(a))
    return np.where(np.abs(a) < 1e-12, 1.0, g)


def _group_phasors(groups: RowGroups, gidx: np.ndarray, rho: float, K: int) -> np.ndarray:
    """(len(gidx), K) complex factor of each group: gain * exp(2 pi i k rho (r_g - r0))."""
    k = np.arange(1, K + 1)
    out = np.empty((len(gidx), K), dtype=np.complex64)
    for j, g in enumerate(gidx):
        gain = _group_gain(k, rho, int(groups.counts[g]), int(groups.steps[g]))
        out[j] = gain * np.exp(2j * np.pi * k * rho * (groups.centers[g] - groups.r0))
    return out


def _design(basis_p: np.ndarray, phasors: np.ndarray) -> np.ndarray:
    """Real design matrix ``(G_p * W, 2K)`` for coefficients ``[Re c_k, Im c_k]``."""
    A = phasors[:, :, None] * basis_p[None]            # (G_p, K, W)
    A = np.transpose(A, (0, 2, 1)).reshape(-1, A.shape[1])  # (G_p*W, K)
    return np.concatenate([A.real, -A.imag], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Spectrum in scan time
# ---------------------------------------------------------------------------

def tspace_spectrum(
    profiles: np.ndarray,
    geometry: ScanGeometry,
    nfft: int | None = None,
    max_frames: int = 300,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Average power spectrum of column profiles ``(N, W)`` expressed in scan time.

    Each pixel value is expanded to its dwell samples, so the non-uniform pixel
    clock becomes a uniform sample clock. Returns ``(freqs_hz, power)``.
    """
    profiles = np.atleast_2d(np.asarray(profiles, dtype=np.float32))
    dwell = np.maximum(np.rint(geometry.dwell_samples()).astype(int), 1)
    idx = np.repeat(np.arange(geometry.n_pixels), dwell)
    fs = geometry.effective_sample_rate_hz
    n = idx.size
    if nfft is None:
        nfft = 1 << int(np.ceil(np.log2(8 * n)))
    if profiles.shape[0] > max_frames:
        sel = np.linspace(0, profiles.shape[0] - 1, max_frames).astype(int)
        profiles = profiles[sel]
    window = np.hanning(n).astype(np.float32)
    power = np.zeros(nfft // 2 + 1, dtype=np.float64)
    for s in range(0, profiles.shape[0], 32):
        block = profiles[s : s + 32][:, idx]
        block = (block - block.mean(axis=1, keepdims=True)) * window
        power += (np.abs(np.fft.rfft(block, n=nfft, axis=1)) ** 2).sum(axis=0)
    power /= profiles.shape[0]
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    return freqs, power


def _search_band(geometry: ScanGeometry, f_min_hz: float | None) -> tuple[float, float]:
    """Frequency band in which the fundamental is searched: a few cycles per line up to
    the Nyquist frequency of the median pixel dwell."""
    acq = geometry.line_acq_time_s
    f_lo = f_min_hz if f_min_hz is not None else 3.0 / acq
    f_hi = geometry.effective_sample_rate_hz / (2.0 * float(np.median(geometry.dwell_samples())))
    return f_lo, f_hi


def _max_line_frequency(geometry: ScanGeometry) -> float:
    """Highest frequency at which a spectral line is still meaningful: Nyquist of the
    shortest (center) pixel dwell."""
    return geometry.effective_sample_rate_hz / (2.0 * float(np.min(geometry.dwell_samples())))


def line_power_ratios(
    profiles: np.ndarray,
    geometry: ScanGeometry,
    f0_hz: float,
    n_harmonics: int,
    spectrum: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[int, float]:
    """
    Ratio of the spectral line power at ``k * f0`` to the local background, for
    ``k = 1..n_harmonics``. Values near 1 mean no detectable line.
    """
    freqs, power = spectrum if spectrum is not None else tspace_spectrum(profiles, geometry)
    acq = geometry.line_acq_time_s
    df = 1.0 / acq
    f_hi = _max_line_frequency(geometry)
    harmonics = np.arange(1, n_harmonics + 1) * f0_hz
    out: dict[int, float] = {}
    for k, fk in enumerate(harmonics, start=1):
        if fk > f_hi:
            out[k] = float("nan")
            continue
        line = power[np.abs(freqs - fk) < 0.5 * df].max()
        bg_mask = (np.abs(freqs - fk) > 3 * df) & (np.abs(freqs - fk) < 10 * df)
        for fj in harmonics:
            if fj != fk:
                bg_mask &= np.abs(freqs - fj) > 3 * df
        bg = np.median(power[bg_mask]) if bg_mask.any() else np.nan
        out[k] = float(line / bg) if bg and np.isfinite(bg) else float("nan")
    return out


# ---------------------------------------------------------------------------
# Least-squares fitting
# ---------------------------------------------------------------------------

def _fit_coeffs(
    dyn: np.ndarray,
    groups: RowGroups,
    basis: np.ndarray,
    rho_t: np.ndarray,
    window_frames: int,
) -> np.ndarray:
    """Per-frame complex coefficients ``(T, P, K)`` by windowed least squares."""
    T, G, W = dyn.shape
    P, K, _ = basis.shape
    coeffs = np.zeros((T, P, K), dtype=np.complex64)
    for w0 in range(0, T, window_frames):
        sl = slice(w0, min(w0 + window_frames, T))
        rho_w = float(np.median(rho_t[sl]))
        for p in range(P):
            gidx = groups.indices(p)
            D = _design(basis[p], _group_phasors(groups, gidx, rho_w, K))
            Y = dyn[sl][:, gidx].reshape(sl.stop - sl.start, -1)
            coef, *_ = np.linalg.lstsq(D, Y.T, rcond=None)
            coeffs[sl, p] = (coef[:K] + 1j * coef[K:]).T
    return coeffs


def _model_on_groups(
    coeffs: np.ndarray, rho_t: np.ndarray, basis: np.ndarray, groups: RowGroups
) -> np.ndarray:
    """Model evaluated on the row-group grid, shape ``(T, G, W)``."""
    T, P, K = coeffs.shape
    k = np.arange(1, K + 1)
    out = np.empty((T, groups.n, basis.shape[2]), dtype=np.float32)
    for g in range(groups.n):
        p = int(groups.parity[g])
        gain = np.stack([_group_gain(k, r, int(groups.counts[g]), int(groups.steps[g])) for r in rho_t])
        phase = np.exp(2j * np.pi * k[None] * rho_t[:, None] * (groups.centers[g] - groups.r0))
        E = coeffs[:, p] * gain * phase                     # (T, K)
        out[:, g] = (E @ basis[p]).real
    return out


def _rho_from_phase(c1: np.ndarray, lines_per_frame: int, smooth: int, valid: np.ndarray) -> np.ndarray:
    """Residual phase advance per row from the frame-to-frame phase of the fundamental."""
    T = c1.shape[0]
    if T < 2:
        return np.zeros(T)
    phi = np.angle(c1)
    adv = np.angle(np.exp(1j * np.diff(phi))) / (2 * np.pi)      # cycles per frame
    ok = valid[:-1] & valid[1:]
    if not ok.any():
        return np.zeros(T)
    adv = np.where(ok, adv, np.median(adv[ok]))
    rho_edge = adv / lines_per_frame
    rho = np.empty(T)
    rho[0], rho[-1] = rho_edge[0], rho_edge[-1]
    if T > 2:
        rho[1:-1] = 0.5 * (rho_edge[:-1] + rho_edge[1:])
    size = min(smooth | 1, T if T % 2 else T - 1)
    if size >= 3:
        rho = median_filter(rho, size=size, mode="nearest")
    return rho


def _interpolate_outliers(coeffs: np.ndarray, bad: np.ndarray) -> np.ndarray:
    if not bad.any():
        return coeffs
    good = np.flatnonzero(~bad)
    if good.size == 0:
        return np.zeros_like(coeffs)
    out = coeffs.copy()
    t = np.arange(coeffs.shape[0])
    T, P, K = coeffs.shape
    for p in range(P):
        for k in range(K):
            re = np.interp(t[bad], good, coeffs[good, p, k].real)
            im = np.interp(t[bad], good, coeffs[good, p, k].imag)
            out[bad, p, k] = re + 1j * im
    return out


@dataclass
class RippleFit:
    """Per-frame ripple model parameters (see :func:`fit_ripple`)."""

    f0_hz: float
    f_hz: float
    harmonic_index: int
    n_harmonics: int
    coeffs: np.ndarray            # (T, P, K) complex64
    rho: np.ndarray               # (T,) cycles per row
    static_profile: np.ndarray    # (G, W)
    basis: np.ndarray             # (P, K, W) complex64
    groups: RowGroups
    outlier_frames: np.ndarray
    explained_variance: float

    @property
    def n_frames(self) -> int:
        return self.coeffs.shape[0]

    @property
    def amplitude(self) -> np.ndarray:
        """(T, P, K) amplitude in counts."""
        return np.abs(self.coeffs)

    @property
    def phase(self) -> np.ndarray:
        """(T, P) phase of the fundamental in cycles."""
        return np.angle(self.coeffs[:, :, 0]) / (2 * np.pi)

    def phase_advance(self) -> np.ndarray:
        """(T-1,) frame-to-frame phase advance of the fundamental in cycles (parity 0)."""
        phi = np.angle(self.coeffs[:, 0, 0])
        return np.angle(np.exp(1j * np.diff(phi))) / (2 * np.pi)

    def summary(self) -> dict[str, Any]:
        amp = self.amplitude[:, 0, :]
        med = np.median(amp, axis=0)
        adv = self.phase_advance()
        return {
            "f0_hz": self.f0_hz,
            "f_hz": self.f_hz,
            "harmonic_index": self.harmonic_index,
            "n_harmonics": self.n_harmonics,
            "n_frames": self.n_frames,
            "amplitude_median_counts": med.tolist(),
            "amplitude_fundamental_min_max": [float(amp[:, 0].min()), float(amp[:, 0].max())],
            "rho_cycles_per_row_median": float(np.median(self.rho)),
            "rho_cycles_per_row_min_max": [float(self.rho.min()), float(self.rho.max())],
            "phase_advance_cycles_per_frame_median": float(np.median(adv)) if adv.size else 0.0,
            "phase_advance_mad": float(np.median(np.abs(adv - np.median(adv)))) if adv.size else 0.0,
            "n_outlier_frames": int(self.outlier_frames.size),
            "explained_variance": self.explained_variance,
            "bidirectional": self.groups.bidirectional,
        }


def fit_ripple(
    profiles: np.ndarray,
    groups: RowGroups,
    geometry: ScanGeometry,
    f0_hz: float,
    n_harmonics: int = 8,
    window_frames: int = 256,
    n_iter: int = 2,
    smooth_rho_frames: int = 51,
    outlier_mad: float = 5.0,
    rho: float | None = None,
    pixel_integration: bool = True,
    static_window_frames: int | None = 1024,
) -> RippleFit:
    """
    Fit the ripple model to row-group profiles ``(T, G, W)``.

    Parameters
    ----------
    profiles : ndarray
        Output of :func:`column_profiles` for every frame (consecutive frames).
    groups : RowGroups
    geometry : ScanGeometry
    f0_hz : float
        Harmonic frequency ``n / line_period`` from detection.
    n_harmonics : int
        Harmonics to fit. 8 reaches the noise floor on MINI2P data.
    window_frames : int
        Frames per least-squares window (the row ramp is constant per window).
    n_iter : int
        Rounds of (fit, ramp update, static re-estimate).
    smooth_rho_frames : int
        Median-filter length for the ramp track.
    outlier_mad : float
        Frames whose fundamental amplitude deviates more than this many MADs
        from the rolling median get interpolated coefficients.
    rho : float, optional
        Fix the ramp instead of tracking it.
    static_window_frames : int or None
        After the first round, the static (non-ripple) profile is a running
        mean of the residual over this many frames, so slow drift of the image
        does not leak into the fit. ``None`` uses the global mean.
    """
    T, G, W = profiles.shape
    valid = np.isfinite(profiles).all(axis=(1, 2)) & (np.abs(profiles).sum(axis=(1, 2)) > 0)
    if valid.sum() < 2:
        raise ValueError("need at least two valid frames to fit the ripple")
    P = np.where(np.isfinite(profiles), profiles, 0.0).astype(np.float32)
    harmonic_index = int(round(f0_hz * geometry.line_period_s))
    lines_per_frame = geometry.lines_per_frame_period

    static = P[valid].mean(axis=0)
    dyn = P - static
    dyn[~valid] = 0.0
    rho_t = np.zeros(T) if rho is None else np.full(T, float(rho))
    f_hz = f0_hz
    basis = ripple_basis(geometry, f_hz, n_harmonics, pixel_integration)

    for _ in range(n_iter):
        coeffs = _fit_coeffs(dyn, groups, basis, rho_t, window_frames)
        if rho is None:
            rho_t = _rho_from_phase(coeffs[:, 0, 0], lines_per_frame, smooth_rho_frames, valid)
            f_hz = f0_hz + float(np.median(rho_t)) / geometry.line_period_s
            basis = ripple_basis(geometry, f_hz, n_harmonics, pixel_integration)
        model_g = _model_on_groups(coeffs, rho_t, basis, groups)
        resid = P - model_g
        resid[~valid] = resid[valid].mean(axis=0)
        static = resid[valid].mean(axis=0)
        if static_window_frames and static_window_frames < T:
            size = int(min(static_window_frames, T))
            static_t = uniform_filter1d(resid, size=size, axis=0, mode="nearest")
        else:
            static_t = static
        dyn = P - static_t
        dyn[~valid] = 0.0

    coeffs = _fit_coeffs(dyn, groups, basis, rho_t, window_frames)

    amp = np.abs(coeffs[:, 0, 0])
    size = min(51, T if T % 2 else T - 1)
    med = median_filter(amp, size=size, mode="nearest") if size >= 3 else np.full(T, np.median(amp))
    mad = np.median(np.abs(amp - med)) + 1e-6
    bad = (np.abs(amp - med) > outlier_mad * mad) | ~valid
    coeffs = _interpolate_outliers(coeffs, bad)

    model_g = _model_on_groups(coeffs, rho_t, basis, groups)
    resid = dyn[valid] - model_g[valid]
    ev = float(1.0 - resid.var() / max(dyn[valid].var(), 1e-12))

    return RippleFit(
        f0_hz=f0_hz, f_hz=f_hz, harmonic_index=harmonic_index, n_harmonics=n_harmonics,
        coeffs=coeffs, rho=rho_t, static_profile=static, basis=basis, groups=groups,
        outlier_frames=np.flatnonzero(bad), explained_variance=ev,
    )


def ripple_frames(
    fit: RippleFit,
    frame_indices: np.ndarray | range,
    batch: int = 64,
    out: np.ndarray | None = None,
    subtract: bool = False,
) -> np.ndarray:
    """
    Evaluate the fitted ripple on the full frame grid: ``(N, H, W)`` float32.

    With *out* given, the model is written into it (or subtracted from it in
    place when ``subtract=True``), which keeps peak memory at one batch.
    """
    idx = np.asarray(list(frame_indices), dtype=int)
    H, W = fit.groups.n_lines, fit.basis.shape[2]
    K = fit.n_harmonics
    k = np.arange(1, K + 1)
    rows = np.arange(H)
    if out is None:
        out = np.empty((idx.size, H, W), dtype=np.float32)
        subtract = False
    for p in range(fit.basis.shape[0]):
        rows_p = rows[rows % 2 == p] if fit.groups.bidirectional else rows
        Bp = fit.basis[p]                                            # (K, W)
        for s in range(0, idx.size, batch):
            sub = idx[s : s + batch]
            c = fit.coeffs[sub, p]                                   # (n, K)
            ramp = np.exp(
                2j * np.pi * k[None, None, :] * fit.rho[sub][:, None, None]
                * (rows_p - fit.groups.r0)[None, :, None]
            )                                                        # (n, R, K)
            E = (c[:, None, :] * ramp).reshape(-1, K)                # (n*R, K)
            model = (E.real @ Bp.real - E.imag @ Bp.imag).reshape(len(sub), rows_p.size, W)
            if subtract:
                out[s : s + len(sub), rows_p] -= model
            else:
                out[s : s + len(sub), rows_p] = model
    return out


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

@dataclass
class RippleDetection:
    """Outcome of :func:`detect_ripple`."""

    detected: bool
    f0_hz: float | None
    harmonic_index: int | None
    f_peak_hz: float | None
    snr_db: float
    line_power_ratio: dict[int, float]
    n_harmonics_detected: int
    amplitude_counts: float
    harmonic_rel_amplitudes: list[float]
    phase_advance_cycles_per_frame: float
    rho_cycles_per_row: float
    rho_row_block: float | None
    explained_variance_k1: float
    explained_variance: float
    frames_used: list[tuple[int, int]]
    thresholds: dict[str, float]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["line_power_ratio"] = {str(k): v for k, v in self.line_power_ratio.items()}
        return d


def _choose_blocks(T: int, block_frames: int, n_blocks: int) -> list[tuple[int, int]]:
    if T <= block_frames * n_blocks:
        return [(0, T)]
    starts = np.linspace(0, T - block_frames, n_blocks).astype(int)
    return [(int(s), int(s) + block_frames) for s in starts]


def _ev_fit(dyn_mean: np.ndarray, geometry: ScanGeometry, f_hz: float, K: int, groups1: RowGroups):
    """Fit K harmonics (no ramp) to group-mean profiles (T, W); return coeffs (T,1,K) and EV."""
    basis = ripple_basis(geometry, f_hz, K)
    coeffs = _fit_coeffs(dyn_mean[:, None, :], groups1, basis, np.zeros(dyn_mean.shape[0]),
                         window_frames=dyn_mean.shape[0])
    model = _model_on_groups(coeffs, np.zeros(dyn_mean.shape[0]), basis, groups1)[:, 0]
    ev = float(1.0 - (dyn_mean - model).var() / max(dyn_mean.var(), 1e-12))
    return coeffs, ev


def _detect_from_profiles(
    profiles: np.ndarray,
    groups: RowGroups,
    geometry: ScanGeometry,
    blocks: list[tuple[int, int]] | None = None,
    n_harmonics_probe: int = 8,
    snr_threshold_db: float = 10.0,
    line_ratio_min: float = 10.0,
    ev_min: float = 0.1,
    f_min_hz: float | None = None,
    block_frames: int = 128,
    n_blocks: int = 3,
) -> RippleDetection:
    T = profiles.shape[0]
    blocks = blocks or _choose_blocks(T, block_frames, n_blocks)
    notes: list[str] = []
    thresholds = {"snr_db": snr_threshold_db, "line_ratio_min": line_ratio_min, "ev_min": ev_min}
    g0 = groups.indices(0)

    # dynamic group-mean profiles per block (static structure removed per block)
    dyn_blocks = []
    for a, b in blocks:
        P = profiles[a:b][:, g0].mean(axis=1)
        dyn_blocks.append(P - P.mean(axis=0))
    dyn = np.concatenate(dyn_blocks, axis=0)

    # whole-line single-group geometry for the no-ramp probe fits
    groups1 = RowGroups((slice(0, groups.n_lines),), np.array([groups.r0]), np.array([0]),
                        np.array([groups.n_lines]), np.array([1]), groups.n_lines, False)

    freqs, power = tspace_spectrum(dyn, geometry)
    f_lo, f_hi = _search_band(geometry, f_min_hz)
    band = (freqs >= f_lo) & (freqs <= f_hi)
    if not band.any():
        raise ValueError("empty search band; check the scan geometry")
    bp = power[band]
    f_peak = float(freqs[band][np.argmax(bp)])
    snr_db = float(10 * np.log10(bp.max() / max(np.median(bp), 1e-30)))

    # harmonic index: nearest integers to the peak, decided by explained variance
    n0 = max(1, int(round(f_peak * geometry.line_period_s)))
    cands = [n for n in range(n0 - 3, n0 + 4) if n >= 1 and f_lo <= n / geometry.line_period_s <= f_hi]
    if not cands:
        cands = [n0]
    evs = {n: _ev_fit(dyn, geometry, n / geometry.line_period_s, min(3, n_harmonics_probe), groups1)[1]
           for n in cands}
    n_best = max(evs, key=evs.get)
    f0 = n_best / geometry.line_period_s
    if abs(n_best - f_peak * geometry.line_period_s) > 1.5:
        notes.append(f"harmonic index {n_best} chosen by fit; spectral peak suggested {n0}")

    ratios = line_power_ratios(dyn, geometry, f0, n_harmonics_probe, spectrum=(freqs, power))
    n_det = max([k for k, v in ratios.items() if np.isfinite(v) and v > 3.0], default=0)

    coeffs1, ev1 = _ev_fit(dyn, geometry, f0, 1, groups1)
    coeffs, ev = _ev_fit(dyn, geometry, f0, n_harmonics_probe, groups1)
    amp = np.abs(coeffs[:, 0, :])
    med = np.median(amp, axis=0)
    amplitude = float(med[0])
    rel = (med / max(med[0], 1e-9)).tolist()

    advances = []
    off = 0
    for d in dyn_blocks:
        phi = np.angle(coeffs[off : off + len(d), 0, 0])
        if len(phi) > 1:
            advances.append(np.angle(np.exp(1j * np.diff(phi))) / (2 * np.pi))
        off += len(d)
    adv = float(np.median(np.concatenate(advances))) if advances else 0.0
    rho = adv / geometry.lines_per_frame_period

    # cross-check: within-frame ramp from row-block phases
    rho_rb = None
    if len(g0) >= 3:
        basis1 = ripple_basis(geometry, f0, 1)
        slopes = []
        for a, b in blocks:
            P = profiles[a:b][:, g0]
            dyn_g = P - P.mean(axis=0)
            phases = np.empty((dyn_g.shape[0], len(g0)))
            for j in range(len(g0)):
                c = _fit_coeffs(dyn_g[:, j : j + 1], groups1, basis1, np.zeros(len(dyn_g)),
                                window_frames=len(dyn_g))
                phases[:, j] = np.angle(c[:, 0, 0])
            phases = np.unwrap(phases, axis=1) / (2 * np.pi)
            x = groups.centers[g0]
            slopes.append(np.polyfit(x, phases.T, 1)[0])
        rho_rb = float(np.median(np.concatenate(slopes)))

    detected = bool(snr_db >= snr_threshold_db and ratios.get(1, 0) >= line_ratio_min and ev1 >= ev_min)
    if not detected:
        notes.append("no significant ripple line found; nothing will be removed unless force=True")
    return RippleDetection(
        detected=detected, f0_hz=f0, harmonic_index=n_best, f_peak_hz=f_peak, snr_db=snr_db,
        line_power_ratio=ratios, n_harmonics_detected=n_det, amplitude_counts=amplitude,
        harmonic_rel_amplitudes=rel, phase_advance_cycles_per_frame=adv, rho_cycles_per_row=rho,
        rho_row_block=rho_rb, explained_variance_k1=ev1, explained_variance=ev,
        frames_used=[(int(a), int(b)) for a, b in blocks], thresholds=thresholds, notes=notes,
    )


# ---------------------------------------------------------------------------
# Sources (arrays or TIFF files) and the public pipeline
# ---------------------------------------------------------------------------

class _Source:
    """Uniform access to a movie given as array, path or TiffFile: (N, C, H, W) chunks."""

    def __init__(self, source, channel: int | None = None):
        self.channel = channel
        self._own = False
        if isinstance(source, np.ndarray):
            arr = source
            if arr.ndim == 2:
                arr = arr[None]
            if arr.ndim == 3:
                arr = arr[:, None]
            if arr.ndim != 4:
                raise ValueError("array source must be (T, H, W) or (T, C, H, W)")
            self.array, self.tif, self.path = arr, None, None
            self.n_frames, self.n_channels, self.height, self.width = arr.shape
            self.dtype = arr.dtype
            self.name = "<array>"
        else:
            if isinstance(source, TiffFile):
                self.tif = source
            else:
                self.tif = TiffFile(str(source))
                self._own = True
            self.array = None
            self.path = Path(self.tif.filehandle.path)
            self.name = self.path.name
            n_ch = n_channels_in_tiff(self.tif)
            self.n_channels = 1 if channel is not None else n_ch
            self._file_channels = n_ch
            self.n_frames = len(self.tif.pages) // n_ch
            page0 = self.tif.pages[0]
            self.height, self.width = page0.shape
            self.dtype = page0.dtype
        self.frame_nbytes = self.height * self.width * np.dtype(self.dtype).itemsize * self.n_channels

    def close(self) -> None:
        if self._own and self.tif is not None:
            self.tif.close()

    def default_chunk(self, max_bytes: int = 64_000_000) -> int:
        return int(max(1, min(512, max_bytes // max(self.frame_nbytes, 1))))

    @property
    def is_bigtiff(self) -> bool | None:
        return bool(self.tif.is_bigtiff) if self.tif is not None else None

    def geometry(self) -> ScanGeometry:
        if self.tif is None or not self.tif.is_scanimage:
            raise ValueError(
                "cannot infer the scan geometry: pass geometry=ScanGeometry(...) "
                "(array input or non-ScanImage TIFF)"
            )
        si = (self.tif.scanimage_metadata or {}).get("FrameData") or {}
        return ScanGeometry.from_si_params(si)

    def iter(self, chunk_frames: int):
        if self.array is not None:
            for s in range(0, self.n_frames, chunk_frames):
                yield s, self.array[s : s + chunk_frames]
            return
        for start, chunk in iter_frames(self.tif, chunk_frames=chunk_frames, channel=self.channel):
            if chunk.ndim == 3:
                chunk = chunk[:, None]
            yield start, chunk

    def read(self, frames: range) -> np.ndarray:
        if self.array is not None:
            return self.array[frames.start : frames.stop : frames.step]
        out = load_tiff_stack(self.tif, frames=frames, channel=self.channel)
        return out[:, None] if out.ndim == 3 else out


@dataclass
class RippleReport:
    """Everything that happened in :func:`remove_ripple`, JSON-serialisable."""

    source: str
    output: str | None
    n_frames: int
    n_channels: int
    shape: tuple[int, int]
    dtype: str
    geometry: dict[str, Any]
    params: dict[str, Any]
    detection: list[dict[str, Any]]
    fit: list[dict[str, Any] | None]
    metrics: list[dict[str, Any] | None]
    runtime_s: dict[str, float]
    removed: bool
    bandaid_version: str = field(default_factory=package_version)
    fits: list[RippleFit | None] | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("fits", None)
        return d

    def to_json(self, path: str | Path) -> Path:
        return write_json(self.to_dict(), path)


def _profile_metrics(fit: RippleFit, profiles: np.ndarray, geometry: ScanGeometry) -> dict[str, Any]:
    g0 = fit.groups.indices(0)
    model_g = _model_on_groups(fit.coeffs, fit.rho, fit.basis, fit.groups)
    before_mean = profiles[:, g0].mean(axis=1)
    before_mean = before_mean - before_mean.mean(axis=0)
    after_mean = (profiles - model_g)[:, g0].mean(axis=1)
    after_mean = after_mean - after_mean.mean(axis=0)
    dyn, dyn_after = before_mean, after_mean
    return {
        "line_power_ratio_before": {str(k): v for k, v in
                                    line_power_ratios(before_mean, geometry, fit.f0_hz, fit.n_harmonics).items()},
        "line_power_ratio_after": {str(k): v for k, v in
                                   line_power_ratios(after_mean, geometry, fit.f0_hz, fit.n_harmonics).items()},
        "dynamic_variance_removed": float(1.0 - dyn_after.var() / max(dyn.var(), 1e-12)),
        "mean_image_change_max": float(np.abs(model_g.mean(axis=0)).max()),
    }


def detect_ripple(
    source,
    geometry: ScanGeometry | None = None,
    *,
    channel: int | None = None,
    n_row_groups: int = 8,
    fit_mask: str | np.ndarray | None = "auto",
    mask_percentile: float = 95.0,
    block_frames: int = 128,
    n_blocks: int = 3,
    n_harmonics_probe: int = 8,
    snr_threshold_db: float = 10.0,
    line_ratio_min: float = 10.0,
    ev_min: float = 0.1,
    f_min_hz: float | None = None,
    mask_frames: int = 256,
) -> RippleDetection:
    """
    Decide whether a movie has the ripple artifact and characterise it.

    Reads a few blocks of consecutive frames (start, middle, end), reduces them
    to row-group profiles and analyses their spectrum in scan time.

    Parameters
    ----------
    source : ndarray ``(T, H, W)``, path, or TiffFile
    geometry : ScanGeometry, optional
        Read from the ScanImage metadata when omitted.
    channel : int, optional
        Channel to analyse in a multi-channel file (default: first).

    Returns
    -------
    RippleDetection
    """
    src = _Source(source, channel=channel if channel is not None else 0 if not isinstance(source, np.ndarray) else None)
    try:
        geometry = geometry or src.geometry()
        _check_geometry(geometry, src)
        groups = RowGroups.from_geometry(geometry, n_row_groups)
        blocks = _choose_blocks(src.n_frames, block_frames, n_blocks)
        weights = _fit_weights(src, fit_mask, mask_percentile, mask_frames)[0]
        parts, rel_blocks, off = [], [], 0
        for a, b in blocks:
            chunk = src.read(range(a, b))[:, 0]
            parts.append(column_profiles(chunk, groups, weights))
            rel_blocks.append((off, off + (b - a)))
            off += b - a
        profiles = np.concatenate(parts, axis=0)
        det = _detect_from_profiles(
            profiles, groups, geometry, blocks=rel_blocks, n_harmonics_probe=n_harmonics_probe,
            snr_threshold_db=snr_threshold_db, line_ratio_min=line_ratio_min, ev_min=ev_min,
            f_min_hz=f_min_hz,
        )
        det.frames_used = [(int(a), int(b)) for a, b in blocks]
        return det
    finally:
        src.close()


def _fit_weights(src: _Source, fit_mask, percentile: float, mask_frames: int) -> list[np.ndarray | None]:
    """Per-channel pixel weights for the profile fit (independent of chunking)."""
    if isinstance(fit_mask, np.ndarray):
        if fit_mask.shape != (src.height, src.width):
            raise ValueError(f"fit_mask must have shape {(src.height, src.width)}")
        return [fit_mask.astype(np.float32)] * src.n_channels
    if isinstance(fit_mask, str) and fit_mask == "auto":
        probe = src.read(range(0, min(src.n_frames, int(mask_frames))))
        return [auto_weights(probe[:, c].mean(axis=0), percentile) for c in range(src.n_channels)]
    if fit_mask is None:
        return [None] * src.n_channels
    raise ValueError("fit_mask must be 'auto', an (H, W) array, or None")


def _check_geometry(geometry: ScanGeometry, src: _Source) -> None:
    if geometry.n_pixels != src.width or geometry.n_lines != src.height:
        raise ValueError(
            f"geometry is {geometry.n_lines}x{geometry.n_pixels} but frames are "
            f"{src.height}x{src.width}"
        )
    if geometry.bidirectional:
        warnings.warn(
            "bidirectional scan: even and odd lines are fitted separately; this path is "
            "not yet validated on real data",
            stacklevel=3,
        )


def remove_ripple(
    source,
    geometry: ScanGeometry | None = None,
    *,
    out: str | Path | None = None,
    n_harmonics: int = 8,
    n_row_groups: int = 8,
    fit_mask: str | np.ndarray | None = "auto",
    mask_percentile: float = 95.0,
    window_frames: int = 256,
    chunk_frames: int | None = None,
    force: bool = False,
    dtype="same",
    detection: RippleDetection | list[RippleDetection] | None = None,
    detection_kwargs: dict[str, Any] | None = None,
    report_path: str | Path | None = None,
    copy_if_clean: bool = False,
    channel: int | None = None,
    mask_frames: int = 256,
) -> tuple[np.ndarray | None, RippleReport]:
    """
    Detect and remove the ripple from a movie.

    Parameters
    ----------
    source : ndarray ``(T, H, W)`` / ``(T, C, H, W)``, path, or TiffFile
    geometry : ScanGeometry, optional
        Read from the ScanImage metadata when omitted (required for arrays).
    out : path, optional
        Write the cleaned movie here (plain BigTIFF, input dtype). When omitted
        the cleaned movie is returned as an array.
    n_harmonics : int
        Harmonics in the model (default 8).
    n_row_groups : int
        Row groups used for the profile fit (default 8).
    fit_mask : ``'auto'``, ndarray ``(H, W)``, or None
        Pixel weights for the profiles; ``'auto'`` excludes the brightest
        *mask_percentile* of the mean image (cell bodies).
    window_frames, chunk_frames : int
        Least-squares window and streaming chunk size.
    force : bool
        Remove even if detection is negative.
    dtype : ``'same'`` or dtype
        Output dtype; integer targets are rounded and clipped.
    detection : RippleDetection, optional
        Reuse a previous detection (per channel if a list).
    report_path : path, optional
        Also write the report as JSON.
    copy_if_clean : bool
        With *out*: copy the input unchanged when no ripple is detected.
    channel : int, optional
        Process only this channel of a multi-channel file.
    mask_frames : int
        Leading frames averaged to build the ``'auto'`` fit mask.

    Returns
    -------
    (cleaned, report)
        *cleaned* is the array (``out`` is None) or ``None`` (written to *out*).
        When nothing was removed, the input array is returned unchanged.
    """
    t_all = time.perf_counter()
    times: dict[str, float] = {}
    src = _Source(source, channel=channel)
    try:
        geometry = geometry or src.geometry()
        _check_geometry(geometry, src)
        out_dtype = src.dtype if (isinstance(dtype, str) and dtype == "same") else np.dtype(dtype)
        chunk_frames = chunk_frames or src.default_chunk()
        groups = RowGroups.from_geometry(geometry, n_row_groups)
        T, C, H, W = src.n_frames, src.n_channels, src.height, src.width
        log.info("%s: %d frames x %d channel(s) of %dx%d %s", src.name, T, C, H, W, src.dtype)

        # ---- pass 1: profiles -------------------------------------------------
        t0 = time.perf_counter()
        profiles = np.empty((T, C, groups.n, W), dtype=np.float32)
        weights = _fit_weights(src, fit_mask, mask_percentile, mask_frames)
        for start, chunk in src.iter(chunk_frames):
            for c in range(C):
                profiles[start : start + chunk.shape[0], c] = column_profiles(chunk[:, c], groups, weights[c])
        times["pass1_profiles"] = time.perf_counter() - t0

        # ---- detection -------------------------------------------------------
        t0 = time.perf_counter()
        dets: list[RippleDetection] = []
        given = detection if isinstance(detection, list) else ([detection] * C if detection else [])
        for c in range(C):
            if given:
                dets.append(given[c])
            else:
                dets.append(_detect_from_profiles(profiles[:, c], groups, geometry, **(detection_kwargs or {})))
            d = dets[-1]
            log.info(
                "channel %d: %s (n=%s, f0=%.1f kHz, SNR %.1f dB, %d harmonics, amp %.1f, advance %.4f cyc/frame)",
                c, "ripple detected" if d.detected else "no ripple", d.harmonic_index,
                (d.f0_hz or 0) / 1e3, d.snr_db, d.n_harmonics_detected, d.amplitude_counts,
                d.phase_advance_cycles_per_frame,
            )
        times["detection"] = time.perf_counter() - t0

        active = [c for c in range(C) if dets[c].detected or force]
        params = {
            "n_harmonics": n_harmonics, "n_row_groups": n_row_groups,
            "fit_mask": fit_mask if isinstance(fit_mask, str) or fit_mask is None else "array",
            "mask_percentile": mask_percentile, "window_frames": window_frames,
            "chunk_frames": chunk_frames, "force": force, "dtype": str(out_dtype),
        }

        def make_report(fits, metrics, removed, output):
            times["total"] = time.perf_counter() - t_all
            return RippleReport(
                source=str(src.path) if src.path else src.name, output=str(output) if output else None,
                n_frames=T, n_channels=C, shape=(H, W), dtype=str(src.dtype),
                geometry=geometry.to_dict(), params=params, detection=[d.to_dict() for d in dets],
                fit=[f.summary() if f else None for f in fits],
                metrics=metrics, runtime_s=dict(times), removed=removed,
            )

        if not active:
            output = None
            if out is not None and copy_if_clean and src.path is not None:
                Path(out).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src.path, out)
                output = out
            report = make_report([None] * C, [None] * C, False, output)
            if report_path:
                report.to_json(report_path)
            result = None if out is not None else (source if isinstance(source, np.ndarray) else None)
            return result, report

        # ---- fit --------------------------------------------------------------
        t0 = time.perf_counter()
        fits: list[RippleFit | None] = [None] * C
        for c in active:
            fits[c] = fit_ripple(profiles[:, c], groups, geometry, dets[c].f0_hz,
                                 n_harmonics=n_harmonics, window_frames=window_frames)
            s = fits[c].summary()
            log.info(
                "channel %d fit: amplitude %.1f counts, rho %.2e cyc/row, EV %.2f, %d outlier frames",
                c, s["amplitude_median_counts"][0], s["rho_cycles_per_row_median"],
                s["explained_variance"], s["n_outlier_frames"],
            )
        times["fit"] = time.perf_counter() - t0

        # ---- pass 2: subtract ---------------------------------------------------
        t0 = time.perf_counter()
        result = None
        writer = None
        if out is None:
            shape = (T, H, W) if C == 1 else (T, C, H, W)
            result = np.empty(shape, dtype=out_dtype)
        else:
            writer = TiffChunkWriter(
                out, out_dtype, bigtiff=True if src.is_bigtiff else None,
                expected_nbytes=T * C * H * W * out_dtype.itemsize,
            )
            writer.__enter__()
        try:
            for start, chunk in src.iter(chunk_frames):
                n = chunk.shape[0]
                cleaned = chunk.astype(np.float32)
                for c in active:
                    ripple_frames(fits[c], range(start, start + n), out=cleaned[:, c], subtract=True)
                cleaned = cast_like(cleaned, out_dtype)
                if writer is not None:
                    writer.write_chunk(cleaned.reshape(n * C, H, W))
                elif C == 1:
                    result[start : start + n] = cleaned[:, 0]
                else:
                    result[start : start + n] = cleaned
        finally:
            if writer is not None:
                writer.__exit__(None, None, None)
        times["pass2_subtract"] = time.perf_counter() - t0

        metrics = [_profile_metrics(fits[c], profiles[:, c], geometry) if fits[c] else None for c in range(C)]
        report = make_report(fits, metrics, True, out)
        report.fits = fits
        if report_path:
            report.to_json(report_path)
        return result, report
    finally:
        src.close()
