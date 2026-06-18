"""
pca_clean.py
-----------------
PCA-based PMT ripple noise removal for two-photon calcium imaging data.

The PMT ripple artifact is a spatially coherent periodic pattern (diamond/chevron)
that dominates the first few PCs when decomposing a stack of frames. Removing
those PCs cleanly suppresses the noise without touching cell signal.

Usage
-----
As a script:
    python pmt_ripple_pca.py input.tif output.tif
    python pmt_ripple_pca.py input.tif output.tif --n-noise-pcs 2 --inspect

As a module:
    from pmt_ripple_pca import remove_ripple_pca
    cleaned = remove_ripple_pca(stack)          # stack: (T, H, W) numpy array
"""

import argparse
import numpy as np
from pathlib import Path


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
    Automatically identify which PCs are PMT ripple by measuring how much of
    their power sits at mid-to-high spatial frequencies (where ripple lives).

    Parameters
    ----------
    components : (n_components, H*W) — PCA components
    explained_variance_ratio : (n_components,)
    shape : (H, W) of the original frames. Required when H != W.
    freq_threshold : fraction of Nyquist above which power is counted as 'ripple-band'
    var_threshold : PCs explaining less than this fraction of variance are skipped
    max_pcs : never flag more than this many PCs regardless

    Returns
    -------
    List of PC indices identified as ripple.
    """
    from numpy.fft import fft2, fftshift
    ripple_pcs = []
    n, pixels = components.shape
    if shape is not None:
        H, W = shape
    else:
        side = int(np.sqrt(pixels))
        H = W = side

    # Stop at clear variance elbow: e.g. PC1=21%, PC2=17%, PC3=9% → ratio 0.53 → stop
    evr_gap_stop = None
    for i in range(1, min(n, max_pcs * 2)):
        if explained_variance_ratio[i] < var_threshold:
            break
        ratio = explained_variance_ratio[i] / (explained_variance_ratio[i - 1] + 1e-12)
        if ratio < 0.65 and i >= 2:
            evr_gap_stop = i
            break

    for i in range(min(n, max_pcs * 2)):
        if evr_gap_stop is not None and i >= evr_gap_stop:
            break
        if explained_variance_ratio[i] < var_threshold:
            break
        comp_2d = components[i].reshape(H, W)
        mag = np.abs(fftshift(fft2(comp_2d)))
        cy, cx = H // 2, W // 2
        # Mask DC
        mag[cy - 2:cy + 2, cx - 2:cx + 2] = 0
        # Fraction of power above frequency threshold
        fy = np.abs(np.fft.fftfreq(H))
        fx = np.abs(np.fft.fftfreq(W))
        FX, FY = np.meshgrid(fx, fy)
        FR = np.sqrt(FX**2 + FY**2)
        high_freq_mask = FR > freq_threshold
        high_power_frac = mag[high_freq_mask].sum() / (mag.sum() + 1e-12)
        if high_power_frac > 0.5:   # >50% power in ripple band → flag it
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
) -> np.ndarray | tuple[np.ndarray, dict]:
    """
    Remove PMT ripple from a (T, H, W) image stack using PCA subtraction.

    Parameters
    ----------
    stack : np.ndarray, shape (T, H, W)
        Raw fluorescence stack. Accepts uint8, uint16, float32.
    n_noise_pcs : int or None
        Number of leading PCs to subtract. If None and auto_detect=True,
        PCs are selected automatically by spatial frequency content.
        If None and auto_detect=False, defaults to 2.
    auto_detect : bool
        Use frequency-based auto-detection when n_noise_pcs is not given.
    freq_threshold : float
        Fraction of Nyquist above which power counts as ripple (auto_detect only).
    return_info : bool
        If True, also return a dict with PCA diagnostics.

    Returns
    -------
    cleaned : np.ndarray, same shape and dtype as stack
    info : dict (only if return_info=True)
        Keys: 'noise_pcs', 'explained_variance_ratio', 'components', 'scores'
    """
    from sklearn.decomposition import PCA

    orig_dtype = stack.dtype
    T, H, W = stack.shape
    X = stack.reshape(T, H * W).astype(np.float32)

    n_components = min(T, H * W, 20)   # cap at 20 for efficiency
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)       # (T, n_components)
    components = pca.components_        # (n_components, H*W)
    evr = pca.explained_variance_ratio_

    # Determine which PCs to remove
    if n_noise_pcs is not None:
        noise_pcs = list(range(n_noise_pcs))
    elif auto_detect:
        noise_pcs = detect_ripple_pcs(components, evr, shape=(H, W), freq_threshold=freq_threshold)
        if not noise_pcs:
            print("  Auto-detect found no clear ripple PCs; defaulting to PC1+PC2.")
            noise_pcs = [0, 1]
    else:
        noise_pcs = [0, 1]

    print(f"  Removing PCs: {[i+1 for i in noise_pcs]} "
          f"(total variance removed: {evr[noise_pcs].sum()*100:.1f}%)")

    # Subtract noise PC contributions
    X_clean = X.copy()
    for i in noise_pcs:
        contribution = np.outer(scores[:, i], components[i])
        X_clean -= contribution

    cleaned = np.clip(X_clean, np.iinfo(orig_dtype).min if np.issubdtype(orig_dtype, np.integer)
                      else X.min(), np.iinfo(orig_dtype).max if np.issubdtype(orig_dtype, np.integer)
                      else X.max())
    cleaned = cleaned.reshape(T, H, W).astype(orig_dtype)

    if return_info:
        info = {
            "noise_pcs": noise_pcs,
            "explained_variance_ratio": evr,
            "components": components,
            "scores": scores,
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
    Plot PCA components coloured by spatial frequency content, and compare
    original vs cleaned for a middle frame.
    """
    import matplotlib.pyplot as plt

    T, H, W = stack.shape
    components = info["components"]
    evr = info["explained_variance_ratio"]
    noise_pcs = info["noise_pcs"]
    n_show = min(len(evr), 12)

    # --- Component grid ---
    cols = 4
    rows = int(np.ceil(n_show / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3))
    axes = axes.flat
    for i in range(n_show):
        ax = axes[i]
        comp = components[i].reshape(H, W)
        ax.imshow(comp, cmap="RdBu_r")
        label = f"PC{i+1} ({evr[i]*100:.1f}%)"
        if i in noise_pcs:
            label += "  ← NOISE"
            for spine in ax.spines.values():
                spine.set_edgecolor("red")
                spine.set_linewidth(2)
        ax.set_title(label, fontsize=9)
        ax.axis("off")
    for j in range(n_show, rows * cols):
        axes[j].axis("off")
    plt.suptitle("PCA components  (red border = removed)", fontsize=12)
    plt.tight_layout()
    if save_path:
        p = Path(save_path)
        fig.savefig(p.with_stem(p.stem + "_components"), dpi=150)
    if show:
        plt.show()
    plt.close()

    # --- Before / after for middle frame ---
    cleaned = remove_ripple_pca(stack, n_noise_pcs=len(noise_pcs))
    mid = T // 2
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    vmin, vmax = stack[mid].min(), stack[mid].max()
    axes[0].imshow(stack[mid], cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Original (frame {mid})")
    axes[1].imshow(cleaned[mid], cmap="gray", vmin=vmin, vmax=vmax)
    axes[1].set_title("PCA cleaned")
    diff = stack[mid].astype(float) - cleaned[mid].astype(float)
    lim = np.percentile(np.abs(diff), 99)
    axes[2].imshow(diff, cmap="RdBu_r", vmin=-lim, vmax=lim)
    axes[2].set_title(f"Removed noise\n(PCs {[i+1 for i in noise_pcs]}, {evr[noise_pcs].sum()*100:.1f}% var)")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    if save_path:
        fig.savefig(p.with_stem(p.stem + "_comparison"), dpi=150)
    if show:
        plt.show()
    plt.close()
    print(f"  Inspection figures saved.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Remove PMT ripple noise from a 2P TIFF stack via PCA subtraction."
    )
    parser.add_argument("input", type=Path, help="Input TIFF stack (T, H, W)")
    parser.add_argument("output", type=Path, help="Output TIFF path")
    parser.add_argument(
        "--n-noise-pcs", type=int, default=None,
        help="Number of leading PCs to remove (default: auto-detect)"
    )
    parser.add_argument(
        "--no-auto-detect", action="store_true",
        help="Disable frequency-based auto-detection; use --n-noise-pcs or default of 2"
    )
    parser.add_argument(
        "--freq-threshold", type=float, default=0.15,
        help="Fraction of Nyquist above which power counts as ripple (default: 0.15)"
    )
    parser.add_argument(
        "--inspect", action="store_true",
        help="Save diagnostic figures alongside the output"
    )
    args = parser.parse_args()

    try:
        import tifffile
    except ImportError:
        raise ImportError("pip install tifffile")

    print(f"Loading {args.input} ...")
    stack = tifffile.imread(args.input)
    if stack.ndim == 2:
        stack = stack[np.newaxis]   # single frame → (1, H, W)
    print(f"  Shape: {stack.shape}  dtype: {stack.dtype}")

    print("Running PCA ripple removal ...")
    cleaned, info = remove_ripple_pca(
        stack,
        n_noise_pcs=args.n_noise_pcs,
        auto_detect=not args.no_auto_detect,
        freq_threshold=args.freq_threshold,
        return_info=True,
    )

    print(f"Saving to {args.output} ...")
    tifffile.imwrite(args.output, cleaned)

    if args.inspect:
        inspect_pca(stack, info, save_path=args.output)

    print("Done.")


if __name__ == "__main__":
    main()
