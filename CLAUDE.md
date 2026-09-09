# bandaid

Python package that detects and removes moving periodic vertical banding (PMT
ripple) from ScanImage two-photon calcium imaging movies, as a preprocessing
step before suite2p / CaImAn.

## Layout

- `bandaid/ripple.py` — the core: `detect_ripple`, `fit_ripple`, `remove_ripple` (streaming two-pass), metrics, `RippleReport`.
- `bandaid/scan_geometry.py` — `ScanGeometry`: pixel-to-time mapping from ScanImage metadata (mask or arcsin fallback).
- `bandaid/io/scanimage.py` — metadata parsing via tifffile, `save_session_metadata` (JSON format v2, see `io/session_metadata_format.md`).
- `bandaid/io/tiff.py` — `load_tiff_stack`, `iter_frames`, `save_tiff`, `TiffChunkWriter` (dtype-safe: rint + clip).
- `bandaid/io/utils.py` — lab filename conventions (`<mouse>_<YYYYMMDD>_..._<index>.tif`).
- `bandaid/pca_clean.py` — PCA subtraction, kept as a first-class alternative for comparison.
- `bandaid/destripe.py` — `destripe_profile`, spatially blind baseline (old names are deprecated wrappers).
- `bandaid/diagnostics.py` — matplotlib figures; `bandaid/cli.py` — `bandaid-clean`, `bandaid-metadata`.
- `tests/` — pytest; `tests/synthetic.py` is the exact forward model used by the synthetic tests.
- `scripts/compare_methods.py` — same metrics for every method on a block of real frames.
- `notebooks/` — exploratory, outputs stripped (nbstripout); treat as scratch.

## Commands

```
pip install -e ".[dev]"
ruff check . && pytest -q            # synthetic + unit tests (no data needed)
pytest -q -m data                    # acceptance tests on data/HK13 (skipped when absent)
bandaid-clean data/HK13/HK13_20260814 --dry-run
bandaid-clean data/HK13/HK13_20260814 --diagnostics -v      # -> <session>/cleaned/
bandaid-metadata data/HK13/HK13_20260814                     # -> session_metadata.json
python scripts/compare_methods.py data/HK13/HK13_20260814
```

## Data

`data/` is gitignored. `data/HK13/HK13_20260814/` holds two 600 MB ScanImage
TIFFs (4512 frames, 256x256 int16, MINI2P, 2 kHz resonant, unidirectional, 50 %
and 90 % laser power). Session folder convention: `<mouse>/<mouse>_<date>[_<session>]/*.tif`.
No fixture is committed; data tests skip when the folder is missing.

## Physical model (measured on HK13; keep these in mind when editing ripple.py)

- Additive ripple at ~100.000 kHz = 50.00 x the 2 kHz line rate, harmonics to
  8f (relative amplitudes ~1, .4, .26, .42, .16, .09, .10, .07), fundamental ~17
  counts at both laser powers.
- Pixels are uniform in space, the resonant mirror velocity is sinusoidal, so
  bands are narrower at the image edges. `SI.hScan2D.mask` (ADC samples per
  pixel) gives the exact column-to-time map; the arcsin formula agrees to 5e-5.
- The tiny offset from 50x makes the pattern crawl 0.02-0.06 cycles/frame and
  ramp ~0.03 cycles down each frame; the offset drifts, so per-frame phase is
  fitted and the ramp is tracked (`rho`).
- Acceptance on HK13: harmonic index 50, SNR > 25 dB, >= 7 harmonics detected,
  all harmonic line ratios < 2 after cleaning (measured ~0.3-0.5), mean image
  change < 0.1 counts, whole file < 30 s and < 1 GB RSS.

## Conventions

- `logging`, never `print`, in library code.
- Preserve the input dtype; cast with `bandaid._util.cast_like` (rint + clip, never wraparound).
- Output movies are plain (Big)TIFF in the input dtype plus JSON sidecars; ScanImage tags are not carried over.
- Keep notebooks stripped; keep `data/` and `.DS_Store` out of git.
- Bidirectional scanning is implemented (even/odd lines fitted separately) but not validated on real data.
