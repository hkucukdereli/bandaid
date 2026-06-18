"""
compare_denoise.py
------------------
Compare three approaches to PMT ripple removal:
  1. PCA subtraction
  2. PCA + ICA rotation
  3. Robust PCA (Candès)

Usage:
    python compare_denoise.py input.tif
    python compare_denoise.py input.tif --n-noise-pcs 2 --rpca-lambda 0.01
"""

import numpy as np
import tifffile
import matplotlib.pyplot as plt
import argparse
from sklearn.decomposition import PCA, FastICA


# ------------------------------------------------------------------ #
#  Method 1 — PCA subtraction
# ------------------------------------------------------------------ #

def denoise_pca(stack, n_noise_pcs=2):
    T, H, W = stack.shape
    X = stack.reshape(T, -1).astype(np.float32)

    pca = PCA(n_components=max(n_noise_pcs + 2, 10))
    scores = pca.fit_transform(X)           # (T, n_components)
    components = pca.components_            # (n_components, H*W)

    X_clean = X.copy()
    for i in range(n_noise_pcs):
        X_clean -= np.outer(scores[:, i], components[i])

    return X_clean.reshape(T, H, W).astype(stack.dtype)


# ------------------------------------------------------------------ #
#  Method 2 — PCA + ICA rotation
# ------------------------------------------------------------------ #

def denoise_pca_ica(stack, n_noise_pcs=2):
    T, H, W = stack.shape
    X = stack.reshape(T, -1).astype(np.float32)

    # Step 1: PCA to get the ripple subspace
    pca = PCA(n_components=max(n_noise_pcs + 2, 10))
    scores = pca.fit_transform(X)           # (T, n_components)
    components = pca.components_            # (n_components, H*W)

    # Step 2: ICA rotation within the ripple subspace only
    subspace_scores = scores[:, :n_noise_pcs]   # (T, n_noise_pcs)
    ica = FastICA(n_components=n_noise_pcs, random_state=0, max_iter=1000)
    ica_scores = ica.fit_transform(subspace_scores)         # (T, n_noise_pcs)
    # mixing_.T rotates components back to pixel space
    ica_components = ica.mixing_.T @ components[:n_noise_pcs]   # (n_noise_pcs, H*W)

    # Identify which ICA component is the ripple by spatial freq power
    ripple_idx = pick_ripple_component(ica_components, H, W)

    # Subtract only the ripple ICA component
    X_clean = X.copy()
    X_clean -= np.outer(ica_scores[:, ripple_idx], ica_components[ripple_idx])

    return X_clean.reshape(T, H, W).astype(stack.dtype)


def pick_ripple_component(components, H, W, freq_lo=0.15, freq_hi=0.6):
    """Return index of the component with most power in the ripple frequency band."""
    scores = []
    for comp in components:
        mag = np.abs(np.fft.fftshift(np.fft.fft2(comp.reshape(H, W))))
        fy = np.abs(np.fft.fftfreq(H))
        fx = np.abs(np.fft.fftfreq(W))
        FX, FY = np.meshgrid(fx, fy)
        FR = np.sqrt(FX**2 + FY**2)
        band = (FR > freq_lo) & (FR < freq_hi)
        scores.append(mag[band].sum() / (mag.sum() + 1e-12))
    return int(np.argmax(scores))


# ------------------------------------------------------------------ #
#  Method 3 — Robust PCA  (Candès et al.)
# ------------------------------------------------------------------ #

def denoise_rpca(stack, lam=None, max_iter=100, tol=1e-5):
    """
    Decompose X = L + S where L is low-rank (ripple) and S is sparse (signal).
    Uses the inexact ALM algorithm (Lin et al. 2010).
    lam controls sparsity vs low-rank trade-off; default 1/sqrt(max(T, pixels)).
    """
    T, H, W = stack.shape
    X = stack.reshape(T, -1).astype(np.float32)

    if lam is None:
        lam = 1.0 / np.sqrt(max(X.shape))

    # ALM initialisation
    mu = X.shape[0] * X.shape[1] / (4 * np.linalg.norm(X, 1))
    rho = 1.5
    Y = X / max(np.linalg.norm(X, 2), np.linalg.norm(X, np.inf) / lam)
    L = np.zeros_like(X)
    S = np.zeros_like(X)

    for i in range(max_iter):
        # Update L via singular value thresholding
        U, sigma, Vt = np.linalg.svd(X - S + Y / mu, full_matrices=False)
        sigma_thresh = np.maximum(sigma - 1.0 / mu, 0)
        L = (U * sigma_thresh) @ Vt

        # Update S via soft thresholding
        Z = X - L + Y / mu
        S = np.sign(Z) * np.maximum(np.abs(Z) - lam / mu, 0)

        # Update dual variable
        residual = X - L - S
        Y = Y + mu * residual
        mu *= rho

        err = np.linalg.norm(residual, 'fro') / (np.linalg.norm(X, 'fro') + 1e-12)
        if err < tol:
            print(f"  RPCA converged at iteration {i+1}, err={err:.2e}")
            break

    # S is the sparse signal; L is the low-rank ripple
    # Return S (signal) reshaped back, cast to original dtype
    S_out = (S + stack.reshape(T, -1).mean(axis=0)).reshape(T, H, W)
    return np.clip(S_out, 0, np.iinfo(stack.dtype).max).astype(stack.dtype)


# ------------------------------------------------------------------ #
#  Comparison figure
# ------------------------------------------------------------------ #

def compare(stack, results, labels, save_path="comparison.png"):
    mid = len(stack) // 2
    frame = stack[mid]
    n = len(results) + 1
    fig, axes = plt.subplots(2, n, figsize=(n * 3.5, 7))

    def show(ax, img, title):
        ax.imshow(img, cmap="gray", vmin=np.percentile(frame, 2), vmax=np.percentile(frame, 98))
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    show(axes[0, 0], frame, "Original")
    axes[1, 0].axis("off")

    for j, (result, label) in enumerate(zip(results, labels)):
        show(axes[0, j + 1], result[mid], label)
        diff = frame.astype(float) - result[mid].astype(float)
        lim = np.percentile(np.abs(diff), 99)
        axes[1, j + 1].imshow(diff, cmap="RdBu_r", vmin=-lim, vmax=lim)
        axes[1, j + 1].set_title(f"removed noise\n(red=+, blue=-)", fontsize=8)
        axes[1, j + 1].axis("off")

    plt.suptitle(f"PMT ripple removal comparison  (frame {mid})", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved {save_path}")


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input TIFF stack (T, H, W)")
    parser.add_argument("--n-noise-pcs", type=int, default=2)
    parser.add_argument("--rpca-lambda", type=float, default=None,
                        help="RPCA lambda (default: 1/sqrt(max(T,pixels)))")
    parser.add_argument("--max-iter", type=int, default=100)
    args = parser.parse_args()

    print(f"Loading {args.input}")
    stack = tifffile.imread(args.input)
    if stack.ndim == 2:
        stack = stack[np.newaxis]
    print(f"  shape: {stack.shape}  dtype: {stack.dtype}")

    print("Method 1: PCA")
    r_pca = denoise_pca(stack, n_noise_pcs=args.n_noise_pcs)

    print("Method 2: PCA + ICA")
    r_ica = denoise_pca_ica(stack, n_noise_pcs=args.n_noise_pcs)

    print("Method 3: Robust PCA")
    r_rpca = denoise_rpca(stack, lam=args.rpca_lambda, max_iter=args.max_iter)

    compare(
        stack,
        [r_pca, r_ica, r_rpca],
        ["PCA", "PCA + ICA", "Robust PCA"],
        save_path=args.input.replace(".tif", "_comparison.png")
    )

    for result, label in zip([r_pca, r_ica, r_rpca], ["pca", "pca_ica", "rpca"]):
        out = args.input.replace(".tif", f"_{label}.tif")
        tifffile.imwrite(out, result)
        print(f"Saved {out}")
