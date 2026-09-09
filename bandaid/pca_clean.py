"""
PCA-based ripple removal for two-photon calcium imaging stacks.

A spatially coherent periodic artifact tends to dominate the leading principal
components of a frame stack; subtracting those components suppresses it. This
is a spatially blind alternative to :mod:`bandaid.ripple`: it does not use the
scan geometry and it cannot separate a *crawling* pattern (which spans two
components per harmonic) from real signal in general, so it is kept for
comparison and for data without ScanImage metadata.

Usage
-----
As a module::

    from bandaid import remove_ripple_pca
    cleaned = remove_ripple_pca(stack, n_noise_pcs=2)      # stack: (T, H, W)

From the command line::

    bandaid-clean session_dir --method pca --n-noise-pcs 2
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np

from bandaid._util import cast_like

__all__ = ["detect_ripple_pcs", "remove_ripple_pca", "inspect_pca"]

log = logging.getLogger("bandaid.pca")


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def detect_ripple_pcs(
    components: np.ndarray,
    explained_variance_ratio: np.ndarray,
    shape: tuple[int, int] | None = None,
    freq_threshold: float = 0.15,
    var_threshold: float = 0.03,
    max_pcs: int = 4,
) -> list[int]:
    """
    Identify which PCs look like ripple by the fraction of their power at
    mid-to-high spatial frequencies.

    Parameters
    ----------
    components : (n_components, H*W)
    explained_variance_ratio : (n_components,)
    shape : (H, W) of the frames. Required when H != W.
    freq_threshold : fraction of Nyquist above which power counts as ripple band
    var_threshold : PCs explaining less variance than this are skipped
    max_pcs : never flag more than this many PCs

    Returns
    -------
    list of int
        Indices of PCs identified as ripple (possibly empty).
    """
    n, pixels = components.shape
    if shape is not None:
        H, W = shape
    else:
        side = int(np.sqrt(pixels))
        if side * side != pixels:
            raise ValueError("components are not square frames; pass shape=(H, W)")
        H = W = side

    # stop at a clear variance elbow: e.g. PC1=21%, PC2=17%, PC3=9% -> ratio 0.53 -> stop
    evr_gap_stop = None
    for i in range(1, min(n, max_pcs * 2)):
        if explained_variance_ratio[i] < var_threshold:
            break
        ratio = explained_variance_ratio[i] / (explained_variance_ratio[i - 1] + 1e-12)
        if ratio < 0.65 and i >= 2:
            evr_gap_stop = i
            break

    fy = np.abs(np.fft.fftfreq(H))
    fx = np.abs(np.fft.fftfreq(W))
    FX, FY = np.meshgrid(fx, fy)
    high_freq_mask = np.fft.fftshift(np.sqrt(FX**2 + FY**2) > freq_threshold)
    cy, cx = H // 2, W // 2

    ripple_pcs: list[int] = []
    for i in range(min(n, max_pcs * 2)):
        if evr_gap_stop is not None and i >= evr_gap_stop:
            break
        if explained_variance_ratio[i] < var_threshold:
            break
        mag = np.abs(np.fft.fftshift(np.fft.fft2(components[i].reshape(H, W))))
        mag[cy - 2 : cy + 3, cx - 2 : cx + 3] = 0  # mask DC
        high_power_frac = mag[high_freq_mask].sum() / (mag.sum() + 1e-12)
        if high_power_frac > 0.5:
            ripple_pcs.append(i)
        if len(ripple_pcs) >= max_pcs:
            break
    return ripple_pcs


def remove_ripple_pca(
    stack: np.ndarray,
    n_noise_pcs: int | None = None,
    auto_detect: bool = True,
    freq_threshold: float = 0.15,
    return_info: bool = False,
    fit_frames: int | None = None,
    n_components: int = 20,
    chunk_frames: int = 512,
    random_state: int = 0,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """
    Remove ripple from a ``(T, H, W)`` stack by subtracting leading PCs.

    Parameters
    ----------
    stack : np.ndarray, shape (T, H, W)
        Raw fluorescence stack (integer or float).
    n_noise_pcs : int or None
        Number of leading PCs to subtract. If ``None`` and *auto_detect* is
        true, PCs are selected by spatial frequency content; if none qualify
        the stack is returned unchanged (with a warning). If ``None`` and
        *auto_detect* is false, 2 PCs are removed.
    auto_detect : bool
        Use frequency-based selection when *n_noise_pcs* is not given.
    freq_threshold : float
        Fraction of Nyquist above which power counts as ripple (auto-detect).
    return_info : bool
        Also return a dict with ``noise_pcs``, ``explained_variance_ratio``,
        ``components``, ``scores`` and ``mean``.
    fit_frames : int, optional
        Fit the PCA on this many evenly spaced frames instead of all frames,
        then project every frame. Bounds memory for long movies.
    n_components : int
        PCs to compute (default 20).
    chunk_frames : int
        Frames per projection chunk.

    Returns
    -------
    cleaned : np.ndarray, same shape and dtype as *stack*
    info : dict (only if ``return_info``)
    """
    from sklearn.decomposition import PCA

    stack = np.asarray(stack)
    if stack.ndim != 3:
        raise ValueError(f"expected a (T, H, W) stack, got shape {stack.shape}")
    T, H, W = stack.shape
    n_components = int(min(n_components, T, H * W))

    if fit_frames is not None and fit_frames < T:
        fit_idx = np.linspace(0, T - 1, int(fit_frames)).astype(int)
    else:
        fit_idx = np.arange(T)
    X_fit = stack[fit_idx].reshape(len(fit_idx), H * W).astype(np.float32)
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(X_fit)
    del X_fit
    components = pca.components_.astype(np.float32)   # (n_components, H*W)
    mean = pca.mean_.astype(np.float32)
    evr = pca.explained_variance_ratio_

    if n_noise_pcs is not None:
        noise_pcs = list(range(int(n_noise_pcs)))
    elif auto_detect:
        noise_pcs = detect_ripple_pcs(components, evr, shape=(H, W), freq_threshold=freq_threshold)
        if not noise_pcs:
            warnings.warn(
                "remove_ripple_pca: auto-detect found no ripple-like PCs; returning the stack "
                "unchanged (pass n_noise_pcs to force removal)",
                stacklevel=2,
            )
    else:
        noise_pcs = [0, 1]

    log.info("removing PCs %s (%.1f%% of variance)", [i + 1 for i in noise_pcs],
             100 * float(evr[noise_pcs].sum()) if noise_pcs else 0.0)

    scores = np.empty((T, n_components), dtype=np.float32)
    cleaned = np.empty_like(stack)
    noise_comp = components[noise_pcs]                # (k, H*W)
    for s in range(0, T, chunk_frames):
        sl = slice(s, min(s + chunk_frames, T))
        Xc = stack[sl].reshape(sl.stop - sl.start, H * W).astype(np.float32)
        Xc -= mean
        scores[sl] = Xc @ components.T
        if noise_pcs:
            Xc -= scores[sl][:, noise_pcs] @ noise_comp
        Xc += mean
        cleaned[sl] = cast_like(Xc, stack.dtype).reshape(-1, H, W)

    if return_info:
        info = {
            "noise_pcs": noise_pcs,
            "explained_variance_ratio": evr,
            "components": components,
            "scores": scores,
            "mean": mean,
            "fit_frames": fit_idx,
        }
        return cleaned, info
    return cleaned


# ---------------------------------------------------------------------------
# Inspection / diagnostics
# ---------------------------------------------------------------------------

def inspect_pca(
    stack: np.ndarray,
    info: dict,
    save_path: str | Path | None = None,
    show: bool = False,
):
    """
    Plot the leading PCA components (removed ones outlined in red) and the
    before/after/removed images of the middle frame, reconstructed from *info*.
    """
    import matplotlib.pyplot as plt

    T, H, W = stack.shape
    components = info["components"]
    evr = info["explained_variance_ratio"]
    noise_pcs = list(info["noise_pcs"])
    n_show = min(len(evr), 12)

    cols = 4
    rows = int(np.ceil(n_show / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3))
    axes = np.atleast_1d(axes).flat
    for i in range(n_show):
        ax = axes[i]
        ax.imshow(components[i].reshape(H, W), cmap="RdBu_r")
        label = f"PC{i + 1} ({evr[i] * 100:.1f}%)"
        if i in noise_pcs:
            label += "  removed"
            for spine in ax.spines.values():
                spine.set_edgecolor("red")
                spine.set_linewidth(2)
        ax.set_title(label, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for j in range(n_show, rows * cols):
        axes[j].axis("off")
    plt.suptitle("PCA components (red border = removed)", fontsize=12)
    plt.tight_layout()
    if save_path:
        p = Path(save_path)
        fig.savefig(p.with_stem(p.stem + "_components"), dpi=150)
    if show:
        plt.show()
    plt.close(fig)

    mid = T // 2
    removed = np.zeros(H * W, dtype=np.float32)
    for i in noise_pcs:
        removed += info["scores"][mid, i] * components[i]
    removed = removed.reshape(H, W)
    raw = stack[mid].astype(np.float32)
    cleaned = raw - removed
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    vmin, vmax = np.percentile(raw, [1, 99])
    axes[0].imshow(raw, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Original (frame {mid})")
    axes[1].imshow(cleaned, cmap="gray", vmin=vmin, vmax=vmax)
    axes[1].set_title("PCA cleaned")
    lim = max(np.percentile(np.abs(removed), 99), 1e-6)
    axes[2].imshow(removed, cmap="RdBu_r", vmin=-lim, vmax=lim)
    axes[2].set_title(
        f"Removed (PCs {[i + 1 for i in noise_pcs]}, {evr[noise_pcs].sum() * 100:.1f}% var)"
        if noise_pcs else "Removed (nothing)"
    )
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    if save_path:
        fig.savefig(p.with_stem(p.stem + "_comparison"), dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    if save_path:
        log.info("inspection figures saved next to %s", save_path)


# ---------------------------------------------------------------------------
# CLI (kept for ``python -m bandaid.pca_clean``; ``bandaid-clean --method pca`` is preferred)
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    import tifffile

    parser = argparse.ArgumentParser(description="Remove ripple from a TIFF stack via PCA subtraction.")
    parser.add_argument("input", type=Path, help="Input TIFF stack (T, H, W)")
    parser.add_argument("output", type=Path, help="Output TIFF path")
    parser.add_argument("--n-noise-pcs", type=int, default=None, help="leading PCs to remove (default: auto)")
    parser.add_argument("--no-auto-detect", action="store_true", help="disable auto-detection; default 2 PCs")
    parser.add_argument("--freq-threshold", type=float, default=0.15)
    parser.add_argument("--fit-frames", type=int, default=None, help="frames used to fit the PCA")
    parser.add_argument("--inspect", action="store_true", help="save diagnostic figures next to the output")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    stack = tifffile.imread(args.input)
    if stack.ndim == 2:
        stack = stack[np.newaxis]
    log.info("loaded %s: %s %s", args.input, stack.shape, stack.dtype)
    cleaned, info = remove_ripple_pca(
        stack, n_noise_pcs=args.n_noise_pcs, auto_detect=not args.no_auto_detect,
        freq_threshold=args.freq_threshold, fit_frames=args.fit_frames, return_info=True,
    )
    from bandaid.io import save_tiff

    save_tiff(cleaned, args.output)
    if args.inspect:
        inspect_pca(stack, info, save_path=args.output)
    log.info("saved %s", args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
