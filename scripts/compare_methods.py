"""
Compare ripple-removal methods on a block of frames with the same metrics.

Usage
-----
    python scripts/compare_methods.py data/HK13/HK13_20260814 [--frames 2000:2600] [--out table.md]
    python scripts/compare_methods.py some_file.tif --frames 0:600

Methods: scan-time harmonic model (K=4, K=8), PCA subtraction (2, 4, 8 PCs),
PCA + ICA rotation (2 PCs), and the per-column median baseline. The legacy
robust-PCA variant was dropped: its ALM loop terminated at iteration 1 on real
data, i.e. it never decomposed anything.

Metrics
-------
line_k1, line_max   spectral line power / background at the fundamental and the
                    worst harmonic (k <= 8) in scan time; ~1 means no line left
var_removed         fraction of dynamic column-profile variance removed
mean_change         max change of the temporal-mean image (counts)
std_cells           temporal-std ratio cleaned/raw on the brightest 5 % of pixels
                    (signal preservation proxy; the ripple itself contributes
                    little there, so values well below 1 mean activity was removed)
runtime             seconds
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from bandaid.destripe import destripe_profile
from bandaid.io import load_tiff_stack, read_scanimage_metadata
from bandaid.io.utils import get_files
from bandaid.pca_clean import remove_ripple_pca
from bandaid.ripple import (
    RowGroups,
    column_profiles,
    detect_ripple,
    line_power_ratios,
    remove_ripple,
)
from bandaid.scan_geometry import ScanGeometry


def pca_ica(stack: np.ndarray, n_noise_pcs: int = 2) -> np.ndarray:
    """PCA subspace, ICA rotation inside it, remove the most ripple-like component."""
    from sklearn.decomposition import PCA, FastICA

    T, H, W = stack.shape
    X = stack.reshape(T, -1).astype(np.float32)
    pca = PCA(n_components=max(n_noise_pcs + 2, 10))
    scores = pca.fit_transform(X)
    comps = pca.components_
    ica = FastICA(n_components=n_noise_pcs, random_state=0, max_iter=1000)
    ica_scores = ica.fit_transform(scores[:, :n_noise_pcs])
    ica_comps = ica.mixing_.T @ comps[:n_noise_pcs]
    fy = np.abs(np.fft.fftfreq(H))
    fx = np.abs(np.fft.fftfreq(W))
    FX, FY = np.meshgrid(fx, fy)
    band = np.fft.fftshift((np.sqrt(FX**2 + FY**2) > 0.15) & (np.sqrt(FX**2 + FY**2) < 0.6))
    frac = []
    for comp in ica_comps:
        mag = np.abs(np.fft.fftshift(np.fft.fft2(comp.reshape(H, W))))
        frac.append(mag[band].sum() / (mag.sum() + 1e-12))
    j = int(np.argmax(frac))
    X -= np.outer(ica_scores[:, j], ica_comps[j])
    return np.clip(np.rint(X), np.iinfo(stack.dtype).min, np.iinfo(stack.dtype).max).astype(stack.dtype).reshape(T, H, W)


def metrics(raw: np.ndarray, cleaned: np.ndarray, geometry: ScanGeometry, f0_hz: float) -> dict:
    groups = RowGroups.from_geometry(geometry, 1)
    pr = column_profiles(raw, groups)[:, 0]
    pc = column_profiles(cleaned, groups)[:, 0]
    dr, dc = pr - pr.mean(0), pc - pc.mean(0)
    ratios = line_power_ratios(dc, geometry, f0_hz, 8)
    finite = [v for v in ratios.values() if np.isfinite(v)]
    mean_raw = raw.mean(axis=0, dtype=np.float64)
    bright = mean_raw >= np.percentile(mean_raw, 95)
    std_raw = raw[:, bright].astype(np.float32).std(axis=0)
    std_cl = cleaned[:, bright].astype(np.float32).std(axis=0)
    return {
        "line_k1": ratios[1],
        "line_max": max(finite) if finite else float("nan"),
        "var_removed": 1 - dc.var() / dr.var(),
        "mean_change": float(np.abs(cleaned.mean(axis=0, dtype=np.float64) - mean_raw).max()),
        "std_cells": float(np.median(std_cl / np.maximum(std_raw, 1e-6))),
    }


def run(path: Path, frames: range) -> list[dict]:
    geometry = ScanGeometry.from_si_params(read_scanimage_metadata(path, frames=False)["si_params"])
    raw = load_tiff_stack(path, frames=frames)
    det = detect_ripple(raw, geometry)
    print(f"\n{path.name}: frames {frames.start}:{frames.stop}, ripple {'detected' if det.detected else 'NOT detected'} "
          f"(n={det.harmonic_index}, SNR {det.snr_db:.1f} dB, amp {det.amplitude_counts:.1f})", file=sys.stderr)
    f0 = det.f0_hz
    methods = [
        ("ripple K=4", lambda s: remove_ripple(s, geometry, n_harmonics=4, detection=det)[0]),
        ("ripple K=8", lambda s: remove_ripple(s, geometry, n_harmonics=8, detection=det)[0]),
        ("pca n=2", lambda s: remove_ripple_pca(s, n_noise_pcs=2)),
        ("pca n=4", lambda s: remove_ripple_pca(s, n_noise_pcs=4)),
        ("pca n=8", lambda s: remove_ripple_pca(s, n_noise_pcs=8)),
        ("pca+ica n=2", lambda s: pca_ica(s, 2)),
        ("column median", lambda s: destripe_profile(s, "columns", "median")),
    ]
    rows = []
    for name, fn in methods:
        t0 = time.perf_counter()
        try:
            cleaned = fn(raw)
        except Exception as exc:  # report and continue
            print(f"  {name}: failed ({exc})", file=sys.stderr)
            continue
        dt = time.perf_counter() - t0
        m = metrics(raw, cleaned, geometry, f0)
        m.update(file=path.name, method=name, runtime=dt)
        rows.append(m)
        print(f"  {name:14s} line_k1={m['line_k1']:8.1f} line_max={m['line_max']:8.1f} "
              f"var_removed={m['var_removed']:.3f} mean_change={m['mean_change']:.3f} "
              f"std_cells={m['std_cells']:.3f} {dt:.1f}s", file=sys.stderr)
    return rows


def to_markdown(rows: list[dict]) -> str:
    lines = ["| file | method | line k=1 | line max | var removed | mean change | std cells | time (s) |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['file']} | {r['method']} | {r['line_k1']:.1f} | {r['line_max']:.1f} | "
                     f"{r['var_removed']:.3f} | {r['mean_change']:.3f} | {r['std_cells']:.3f} | {r['runtime']:.1f} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="session folder or TIFF file")
    ap.add_argument("--frames", default="2000:2600", help="frame range start:stop (default 2000:2600)")
    ap.add_argument("--out", type=Path, default=None, help="write the markdown table here")
    args = ap.parse_args(argv)
    a, b = (int(v) for v in args.frames.split(":"))
    p = Path(args.input)
    files = [Path(f) for f, _ in get_files(p)] if p.is_dir() else [p]
    rows = []
    for f in files:
        n = read_scanimage_metadata(f, frames=False)["n_frames"]
        rows.extend(run(f, range(min(a, n - 1), min(b, n))))
    table = to_markdown(rows)
    print("\n" + table)
    if args.out:
        args.out.write_text(table + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
