"""
Diagnostic figures for ripple detection and removal (matplotlib is imported lazily).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bandaid.ripple import RippleFit, RowGroups, column_profiles, tspace_spectrum
from bandaid.scan_geometry import ScanGeometry

__all__ = [
    "plot_tspace_spectra",
    "plot_fit_tracks",
    "plot_frame_comparison",
    "plot_profiles_over_time",
    "write_diagnostics",
]


def _plt():
    import matplotlib

    if not matplotlib.get_backend().lower().startswith(("agg", "module://")):
        try:
            matplotlib.use("Agg")
        except Exception:  # pragma: no cover
            pass
    import matplotlib.pyplot as plt

    return plt


def _dynamic(profiles: np.ndarray) -> np.ndarray:
    return profiles - profiles.mean(axis=0)


def plot_tspace_spectra(
    profiles_before: np.ndarray,
    profiles_after: np.ndarray | None,
    geometry: ScanGeometry,
    f0_hz: float | None = None,
    n_harmonics: int = 8,
    ax=None,
    f_max_hz: float | None = None,
):
    """Power spectra of dynamic column profiles in scan time, before and after cleaning."""
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    f, p = tspace_spectrum(_dynamic(profiles_before), geometry)
    f_max = f_max_hz or (n_harmonics + 1.5) * (f0_hz or f[np.argmax(p[1:]) + 1])
    sel = (f > 0) & (f <= f_max)
    ax.semilogy(f[sel] / 1e3, p[sel], color="0.25", lw=0.8, label="before")
    if profiles_after is not None:
        f2, p2 = tspace_spectrum(_dynamic(profiles_after), geometry)
        ax.semilogy(f2[sel] / 1e3, p2[sel], color="tab:green", lw=0.8, label="after")
    if f0_hz:
        for k in range(1, n_harmonics + 1):
            ax.axvline(k * f0_hz / 1e3, color="tab:red", lw=0.6, ls="--", alpha=0.6)
    ax.set_xlabel("frequency in scan time (kHz)")
    ax.set_ylabel("power")
    ax.set_title("column-profile spectrum; dashed = fitted harmonics")
    ax.legend(loc="upper right", fontsize=8)
    return ax


def plot_fit_tracks(fit: RippleFit, frame_period_s: float | None = None, axes=None):
    """Per-frame amplitude of each harmonic, fundamental phase, and the row-ramp track."""
    plt = _plt()
    if axes is None:
        _, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    t = np.arange(fit.n_frames)
    x = t * frame_period_s if frame_period_s else t
    xlabel = "time (s)" if frame_period_s else "frame"
    amp = fit.amplitude[:, 0, :]
    for k in range(fit.n_harmonics):
        axes[0].plot(x, amp[:, k], lw=0.7, label=f"h{k + 1}")
    if fit.outlier_frames.size:
        axes[0].plot(x[fit.outlier_frames], amp[fit.outlier_frames, 0], "rx", ms=4, label="outlier")
    axes[0].set_ylabel("amplitude (counts)")
    axes[0].legend(ncol=5, fontsize=7, loc="upper right")
    axes[1].plot(x, np.unwrap(2 * np.pi * fit.phase[:, 0]) / (2 * np.pi), lw=0.8)
    axes[1].set_ylabel("fundamental phase (cycles)")
    axes[2].plot(x, fit.rho * 1e4, lw=0.8)
    axes[2].set_ylabel("row ramp (1e-4 cycles/row)")
    axes[2].set_xlabel(xlabel)
    axes[0].set_title(
        f"n={fit.harmonic_index}, f0={fit.f0_hz / 1e3:.1f} kHz, K={fit.n_harmonics}, "
        f"EV={fit.explained_variance:.2f}"
    )
    return axes


def plot_frame_comparison(raw: np.ndarray, cleaned: np.ndarray, title: str = "", axes=None):
    """Raw frame, cleaned frame and the removed component side by side."""
    plt = _plt()
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    raw = raw.astype(np.float32)
    cleaned = cleaned.astype(np.float32)
    vmin, vmax = np.percentile(raw, [1, 99])
    axes[0].imshow(raw, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].set_title("raw")
    axes[1].imshow(cleaned, cmap="gray", vmin=vmin, vmax=vmax)
    axes[1].set_title("cleaned")
    diff = raw - cleaned
    lim = max(np.percentile(np.abs(diff), 99), 1e-6)
    axes[2].imshow(diff, cmap="RdBu_r", vmin=-lim, vmax=lim)
    axes[2].set_title("removed")
    for ax in axes:
        ax.axis("off")
    if title:
        axes[1].figure.suptitle(title)
    return axes


def plot_profiles_over_time(profiles: np.ndarray, ax=None, title: str = "dynamic column profiles"):
    """Image of de-meaned column profiles (frame x column); crawling stripes appear as diagonals."""
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    d = _dynamic(profiles)
    lim = max(np.percentile(np.abs(d), 99), 1e-6)
    ax.imshow(d, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
    ax.set_xlabel("column")
    ax.set_ylabel("frame")
    ax.set_title(title)
    return ax


def write_diagnostics(
    raw: np.ndarray,
    cleaned: np.ndarray,
    fit: RippleFit,
    geometry: ScanGeometry,
    out_dir: str | Path,
    stem: str,
    frame_offset: int = 0,
) -> list[Path]:
    """
    Save the standard set of figures for a block of consecutive frames
    (``raw`` and ``cleaned`` of shape ``(N, H, W)``). Returns the written paths.
    """
    plt = _plt()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = RowGroups.from_geometry(geometry, 1)
    before = column_profiles(raw, groups)[:, 0]
    after = column_profiles(cleaned, groups)[:, 0]
    paths = []

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    plot_tspace_spectra(before, after, geometry, fit.f0_hz, fit.n_harmonics, ax=axes[0])
    plot_profiles_over_time(before[: min(len(before), 200)], ax=axes[1],
                            title=f"dynamic column profiles (frames {frame_offset}..)")
    fig.tight_layout()
    paths.append(out_dir / f"{stem}_spectrum.png")
    fig.savefig(paths[-1], dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    plot_fit_tracks(fit, geometry.frame_period_s, axes=axes)
    fig.tight_layout()
    paths.append(out_dir / f"{stem}_tracks.png")
    fig.savefig(paths[-1], dpi=130)
    plt.close(fig)

    mid = raw.shape[0] // 2
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    plot_frame_comparison(raw[mid], cleaned[mid], title=f"{stem}: frame {frame_offset + mid}", axes=axes)
    fig.tight_layout()
    paths.append(out_dir / f"{stem}_frames.png")
    fig.savefig(paths[-1], dpi=130)
    plt.close(fig)
    return paths
