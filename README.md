# 🩹 bandaid

Detect and remove moving periodic vertical banding (PMT ripple) from ScanImage
two-photon calcium imaging movies. Designed as a preprocessing step before
suite2p / CaImAn.

The banding is an additive electrical ripple that is periodic in *time*. Under
resonant scanning the mirror velocity is sinusoidal, so the ripple shows up as
vertical bands that get narrower toward the left and right edges of the image,
and because its frequency is a near-integer multiple of the line rate the
pattern is almost identical on every line and crawls slowly across frames.
bandaid maps every column to its acquisition time using the ScanImage metadata,
fits the ripple per frame as a harmonic series in that time coordinate (with
its slow drift), and subtracts it. On the example data this brings all eight
harmonic lines below the noise floor while changing the mean image by less
than 0.05 counts, in about 7 s per 4512-frame file.

## Install

```
pip install -e ".[dev]"      # numpy, scipy, scikit-learn, tifffile (+ pytest, ruff, matplotlib)
```

## Command line

```
bandaid-clean <session_dir or files> [--out-dir DIR] [--dry-run] [--diagnostics] [--method ripple|pca]
bandaid-metadata <session_dir>
```

`bandaid-clean` writes `cleaned/<same name>.tif` (plain BigTIFF, input dtype,
same shape) next to the session plus `cleaned/ripple_report.json` with the
detection, fit and residual metrics for every file. `--dry-run` only prints a
detection table. Files without detectable ripple are skipped (or copied with
`--copy-clean`). `--diagnostics` saves spectrum, fit-track and before/after
figures under `cleaned/diagnostics/`.

`bandaid-metadata` collects all ScanImage parameters and per-frame timestamps
of a session folder into `session_metadata.json`
(format in `bandaid/io/session_metadata_format.md`).

## Python

```python
from bandaid import detect_ripple, remove_ripple, ScanGeometry
from bandaid.io import load_tiff_stack, read_scanimage_metadata

det = detect_ripple("HK13_20260814_50pw_00002.tif")
det.detected, det.harmonic_index, det.snr_db, det.amplitude_counts

# stream a file to a cleaned copy
_, report = remove_ripple("HK13_20260814_50pw_00002.tif", out="cleaned/HK13_20260814_50pw_00002.tif")
report.metrics[0]["line_power_ratio_after"]

# or work in memory (geometry comes from the metadata, or build one by hand)
geometry = ScanGeometry.from_si_params(read_scanimage_metadata(path, frames=False)["si_params"])
mov = load_tiff_stack(path, frames=range(0, 600))
cleaned, report = remove_ripple(mov, geometry)
```

For data without ScanImage metadata pass `geometry=ScanGeometry.uniform(...)`
(galvo scanning) or a `ScanGeometry` built from your own fill fraction and
scanner frequency.

## How it works

1. Every frame is reduced to row-group mean profiles (high SNR, cheap).
2. The dynamic profiles are analysed in scan time: the fundamental frequency,
   its harmonic index relative to the line rate, per-harmonic line power, the
   phase advance per frame and a detection verdict (`RippleDetection`).
3. Per-frame complex amplitudes of `K` harmonics (default 8) are fitted by
   least squares in windows, including the small phase ramp down each frame;
   static image structure is removed as a running mean of the residual.
4. The model is evaluated on the full frame grid and subtracted, streaming
   chunk by chunk, so memory does not depend on movie length. Integer output is
   rounded and clipped, never wrapped.

Measured on the example data (MINI2P, 2 kHz resonant, 256x256): fundamental
100.000 kHz = 50.00 x line rate, harmonics to 8f, amplitude ~17 counts
independent of laser power, crawl 0.02-0.06 cycles/frame.

## Other methods

`bandaid.remove_ripple_pca` subtracts leading principal components; it needs no
scan geometry but cannot follow a crawling pattern without also touching real
signal, and it is kept for comparison and for non-ScanImage data.
`bandaid.destripe.destripe_profile` subtracts a per-frame column (or row)
statistic; it is spatially blind and also removes real column structure.
`python scripts/compare_methods.py <session>` prints the same metrics for all of
them on a block of frames.

### Method comparison on the example data

`python scripts/compare_methods.py data/HK13/HK13_20260814` on frames 2000-2600
of each file. "line" is the spectral line power over background at the
fundamental and at the worst harmonic (k <= 8) after cleaning; ~1 means gone.
"mean change" is the largest change of the temporal-mean image in counts.

| file | method | line k=1 | line max | var removed | mean change | time (s) |
|---|---|---:|---:|---:|---:|---:|
| 50pw | ripple K=8 | 0.4 | 0.5 | 0.86 | 0.05 | 1.7 |
| 50pw | ripple K=4 | 0.4 | 48.4 | 0.80 | 0.06 | 1.7 |
| 50pw | pca n=2 | 0.5 | 277.9 | 0.54 | 0.05 | 1.1 |
| 50pw | pca n=8 | 0.4 | 48.6 | 0.87 | 0.06 | 0.5 |
| 50pw | column median | 1.3 | 1.3 | 0.88 | 5.21 | 0.5 |
| 90pw | ripple K=8 | 0.5 | 0.5 | 0.80 | 0.06 | 1.7 |
| 90pw | pca n=2 | 2.3 | 154.6 | 0.50 | 0.05 | 0.5 |
| 90pw | pca n=8 | 0.4 | 27.6 | 0.86 | 0.05 | 0.5 |
| 90pw | column median | 1.4 | 1.4 | 0.83 | 7.27 | 0.5 |

Only the eight-harmonic scan-time model clears every harmonic line. PCA with
two components removes the fundamental but leaves the higher harmonics (each
crawling harmonic needs two components); with eight it still leaves the 4th.
The column-median baseline also removes stripes but shifts the mean image by
5-7 counts because it subtracts real column structure.

## Limitations

- Requires ScanImage metadata (or a hand-built `ScanGeometry`).
- Bidirectional scanning is implemented (even and odd lines fitted separately)
  but has not been validated on real data yet.
- The ripple frequency must be close to an integer multiple of the line rate,
  which is what makes the bands vertical in the first place.

## Development

```
ruff check . && pytest -q        # synthetic and unit tests
pytest -q -m data                # acceptance tests on data/HK13 (skipped when absent)
```
