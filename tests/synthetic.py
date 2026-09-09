"""
Synthetic movies built from the exact forward model of the ripple artifact.

A temporal ripple ``s(time) = sum_k a_k cos(2 pi k f_rip time + theta_k)`` is
sampled at the acquisition time of every pixel, ``time = t*T_frame + r*T_line +
tau(x)``, with each pixel averaging the waveform over its dwell time. The image
underneath has static cells, slow calcium-like transients and shot noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bandaid.scan_geometry import ScanGeometry


@dataclass
class SyntheticMovie:
    movie: np.ndarray          # (T, H, W) with ripple, integer dtype
    truth: np.ndarray          # (T, H, W) float32, same frames without ripple (before rounding)
    ripple: np.ndarray         # (T, H, W) float32 ripple that was added
    geometry: ScanGeometry
    f_rip_hz: float
    harmonic_index: int
    rho_cycles_per_row: float
    amplitudes: np.ndarray     # absolute amplitude per harmonic (counts)
    phases: np.ndarray


def make_geometry(
    n_pixels: int = 128,
    n_lines: int = 128,
    f_res: float = 2000.0,
    fs: float = 125e6,
    ffs: float = 0.9,
    n_flyback: int = 10,
    bidirectional: bool = False,
) -> ScanGeometry:
    t_line = 1 / f_res
    base = ScanGeometry(
        n_pixels=n_pixels,
        n_lines=n_lines,
        line_period_s=t_line,
        frame_period_s=(n_lines + n_flyback) * t_line,
        sample_rate_hz=fs,
        scanner_frequency_hz=f_res,
        fill_fraction_spatial=ffs,
        bidirectional=bidirectional,
    )
    mask = np.rint(base.dwell_samples()).astype(int)
    return base.replace(mask=mask)


def make_image(H: int, W: int, rng: np.random.Generator, n_cells: int = 30):
    yy, xx = np.mgrid[0:H, 0:W]
    img = 20.0 + 15.0 * xx / W + 5.0 * yy / H
    cells = []
    for _ in range(n_cells):
        cy, cx = rng.uniform(6, H - 6), rng.uniform(6, W - 6)
        r = rng.uniform(2.5, 4.5)
        amp = rng.uniform(80, 250)
        blob = amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r**2))
        img += blob
        cells.append(blob / amp)
    return img.astype(np.float32), cells


def make_activity(T: int, cells, rng: np.random.Generator, n_active: int = 8, tau_frames: float = 12.0):
    H, W = cells[0].shape
    act = np.zeros((T, H, W), dtype=np.float32)
    t = np.arange(T)
    for blob in rng.choice(len(cells), size=min(n_active, len(cells)), replace=False):
        trace = np.zeros(T)
        for onset in rng.integers(0, T, size=rng.integers(1, 4)):
            trace += np.where(t >= onset, np.exp(-(t - onset) / tau_frames), 0.0)
        trace *= rng.uniform(60, 200)
        act += trace[:, None, None].astype(np.float32) * cells[blob][None]
    return act


def make_ripple(
    geometry: ScanGeometry,
    T: int,
    f_rip_hz: float,
    amplitudes,
    phases,
    integrate: bool = True,
) -> np.ndarray:
    """Evaluate the ripple waveform at every pixel's acquisition time, shape (T, H, W)."""
    H, W = geometry.n_lines, geometry.n_pixels
    t2d = geometry.pixel_times_2d()                                   # (H, W) within frame
    dwell_s = geometry.dwell_samples() / geometry.effective_sample_rate_hz
    frame_t = np.arange(T) * geometry.frame_period_s                  # (T,)
    out = np.zeros((T, H, W), dtype=np.float32)
    for k, (a, th) in enumerate(zip(amplitudes, phases, strict=True), start=1):
        f = k * f_rip_hz
        gain = np.sinc(f * dwell_s) if integrate else np.ones(W)      # continuous dwell average
        frame_phase = np.exp(2j * np.pi * f * frame_t + 1j * th)      # (T,)
        pix_phase = np.exp(2j * np.pi * f * t2d) * gain[None, :]     # (H, W)
        out += (a * (frame_phase[:, None, None] * pix_phase[None]).real).astype(np.float32)
    return out


def make_synthetic_movie(
    T: int = 240,
    n_pixels: int = 128,
    n_lines: int = 128,
    harmonic_index: int = 50,
    rho_cycles_per_row: float = 1.96e-4,
    fundamental_amplitude: float = 17.0,
    relative_amplitudes=(1.0, 0.40, 0.26, 0.42, 0.16),
    noise_gain: float = 2.0,
    offset: float = -10.0,
    dtype=np.int16,
    with_ripple: bool = True,
    bidirectional: bool = False,
    seed: int = 0,
) -> SyntheticMovie:
    rng = np.random.default_rng(seed)
    geometry = make_geometry(n_pixels=n_pixels, n_lines=n_lines, bidirectional=bidirectional)
    H, W = n_lines, n_pixels

    image, cells = make_image(H, W, rng)
    activity = make_activity(T, cells, rng)
    signal = image[None] + activity
    noise = rng.normal(0.0, 1.0, size=signal.shape).astype(np.float32) * np.sqrt(np.maximum(signal, 1.0)) * noise_gain
    truth = (signal + noise + offset).astype(np.float32)

    f_rip = harmonic_index / geometry.line_period_s + rho_cycles_per_row / geometry.line_period_s
    amplitudes = fundamental_amplitude * np.asarray(relative_amplitudes, dtype=float)
    phases = rng.uniform(0, 2 * np.pi, size=len(amplitudes))
    ripple = make_ripple(geometry, T, f_rip, amplitudes, phases) if with_ripple else np.zeros_like(truth)

    info = np.iinfo(dtype)
    movie = np.clip(np.rint(truth + ripple), info.min, info.max).astype(dtype)
    return SyntheticMovie(
        movie=movie,
        truth=truth,
        ripple=ripple,
        geometry=geometry,
        f_rip_hz=f_rip,
        harmonic_index=harmonic_index,
        rho_cycles_per_row=rho_cycles_per_row,
        amplitudes=amplitudes,
        phases=phases,
    )
