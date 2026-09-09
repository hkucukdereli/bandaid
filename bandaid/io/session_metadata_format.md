# Session Metadata JSON Format (version 2)

`save_session_metadata()` reads all ScanImage TIFFs in a session folder and
writes a single `session_metadata.json`. `load_session_metadata()` reads it
back, and `ScanGeometry.from_session_metadata()` rebuilds the scan geometry
that the ripple remover needs from it.

## Source TIFF tags

ScanImage writes three metadata tags per TIFF:

| TIFF tag            | Tag # | Scope     | Content                                     |
|---------------------|-------|-----------|---------------------------------------------|
| **Software**        | 305   | Per-file  | `SI.*` key-value acquisition parameters     |
| **Artist**          | 315   | Per-file  | JSON with ROI / scanfield geometry           |
| **ImageDescription**| 270   | Per-page  | Frame number, timestamp, triggers, epoch     |

Values are parsed with `tifffile.matlabstr2py`, so MATLAB arrays (`[1 2 3]`),
cell arrays (`{'a' 'b'}`), booleans, `Inf` and `NaN` become JSON lists, booleans,
`"inf"`/`"-inf"` strings and `null`. The file is strict JSON (no `NaN` tokens).

---

## JSON structure

```
session_metadata.json
├── format_version — 2
├── session        — identifiers and counts
├── acquisition    — commonly used acquisition parameters (from the first file)
├── roi            — scanfield geometry (Artist tag, verbatim JSON)
├── si_params      — the complete parsed SI.* parameter set of the first file
└── files[]        — one entry per TIFF file
    ├── filename, n_frames, n_channels, acquisition_complete, n_dropped_frames, shape, dtype
    ├── si_overrides — SI.* parameters of this file that differ from the first file
    └── frames       — per-frame arrays (from ImageDescription)
```

---

### `session`

| Field          | Type      | Description                                              |
|----------------|-----------|----------------------------------------------------------|
| `mouse`        | str/null  | Mouse ID, inferred from folder name (e.g. `"HK13"`)      |
| `date`         | str/null  | Date string, inferred from folder name (`"20260814"`)    |
| `session_id`   | str/null  | Trailing part of the folder name after the date, if any  |
| `n_files`      | int       | Number of TIFF files in the session                      |
| `total_frames` | int       | Total frame count across all files                       |

Folder name convention: `<mouse>_<YYYYMMDD>[_<session_id>]`. Each identifier
is inferred independently and can be overridden by the corresponding argument.

---

### `acquisition`

Values come from the first TIFF file. Per-file differences are in `files[].si_overrides`.

| Field                    | Type        | SI source key                          | Description                                    |
|--------------------------|-------------|----------------------------------------|------------------------------------------------|
| `scanimage_version`      | str         | `SI.VERSION_MAJOR/MINOR/UPDATE`        | ScanImage version (e.g. `"2023.1.1"`)          |
| `imaging_system`         | str         | `SI.imagingSystem`                     | Imaging system name (e.g. `"MINI2P_001"`)      |
| `acq_state`              | str         | `SI.acqState`                          | Acquisition mode (`"grab"`, `"loop"`, ...)     |
| `frame_rate_hz`          | float       | `SI.hRoiManager.scanFrameRate`         | Frame rate in Hz                               |
| `frame_period_s`         | float       | `SI.hRoiManager.scanFramePeriod`       | Frame period in seconds (includes flyback)     |
| `pixels_per_line`        | int         | `SI.hRoiManager.pixelsPerLine`         | Image width                                    |
| `lines_per_frame`        | int         | `SI.hRoiManager.linesPerFrame`         | Image height                                   |
| `line_period_s`          | float       | `SI.hRoiManager.linePeriod`            | Time between line starts                       |
| `n_flyback_lines`        | int/null    | derived                                | `round(frame_period / line_period) - lines_per_frame` |
| `flyback_time_s`         | float       | `SI.hScan2D.flybackTimePerFrame`       | Frame flyback time                             |
| `zoom`                   | float       | `SI.hRoiManager.scanZoomFactor`        | Optical zoom factor                            |
| `fov_deg`, `fov_um`      | number[]    | `SI.hRoiManager.imagingFov*`           | Field of view corner coordinates               |
| `scan_mode`              | str         | `SI.hScan2D.scanMode`                  | `"resonant"` or `"linear"`                     |
| `bidirectional`          | bool        | `SI.hScan2D.bidirectional`             | Bidirectional scanning enabled                 |
| `uniform_sampling`       | bool        | `SI.hScan2D.uniformSampling`           | Pixels uniform in time rather than space       |
| `fill_fraction_spatial`  | float       | `SI.hScan2D.fillFractionSpatial`       | Spatial fill fraction (`sin(pi*FFt/2)`)        |
| `fill_fraction_temporal` | float       | `SI.hScan2D.fillFractionTemporal`      | Temporal fill fraction                         |
| `scanner_type`           | str         | `SI.hScan2D.scannerType`               | Scanner type (`"RG"`, `"RGG"`, ...)            |
| `scanner_frequency_hz`   | number      | `SI.hScan2D.scannerFrequency`          | Resonant scanner frequency                     |
| `sample_rate_hz`         | number      | `SI.hScan2D.sampleRate`                | ADC sample rate                                |
| `pixel_bin_factor`       | int         | `SI.hScan2D.pixelBinFactor`            | Samples per pixel (linear scanning)            |
| `line_phase_s`           | float       | `SI.hScan2D.linePhase`                 | Line phase adjustment                          |
| `mask_samples_per_pixel` | int[]       | `SI.hScan2D.mask`                      | ADC samples integrated per pixel, per column   |
| `n_channels_available`   | int         | `SI.hChannels.channelsAvailable`       | Number of available channels                   |
| `channels_saved`         | int/int[]   | `SI.hChannels.channelSave`             | Saved channel(s); pages are interleaved        |
| `channels_active`        | int/int[]   | `SI.hChannels.channelsActive`          | Displayed channel(s)                           |
| `channel_offsets`        | int[]       | `SI.hChannels.channelOffset`           | Per-channel ADC offsets                        |
| `laser_power_percent`    | number      | `SI.hBeams.powers`                     | Laser power (percent)                          |
| `laser_power_fraction`   | float       | `SI.hBeams.powerFractions`             | Laser power (fraction)                         |
| `motor_position`         | number[]    | `SI.hMotors.axesPosition`              | Stage position                                 |
| `fastz_enabled`          | bool        | `SI.hFastZ.enable`                     | Fast Z enabled                                 |
| `z_step_um`              | number      | `SI.hStackManager.actualStackZStepSize`| Z step size                                    |

The scan-geometry fields (`mask_samples_per_pixel`, `sample_rate_hz`,
`fill_fraction_*`, `scanner_frequency_hz`, `line_period_s`, `n_flyback_lines`,
`bidirectional`, `uniform_sampling`) are what `bandaid.ripple` uses to map image
columns to acquisition time.

---

### `roi`

The ROI geometry JSON from the Artist tag, stored verbatim. Useful paths:
`RoiGroups.imagingRoiGroup.rois.scanfields.{centerXY,sizeXY,pixelResolutionXY,pixelToRefTransform,affine}`
and `RoiGroups.imagingRoiGroup.rois.UserData.{imagingSystem,scanZoomFactor,pixelsPerLine,linesPerFrame}`.

### `si_params`

All `SI.*` parameters of the first file, parsed. About 330 keys / 20 kB.
Combined with `files[i].si_overrides` this reproduces the parameters of any file.

---

### `files[]`

One entry per TIFF file, ordered by filename.

| Field                  | Type      | Description                                                   |
|------------------------|-----------|---------------------------------------------------------------|
| `filename`             | str       | TIFF filename (not full path)                                 |
| `n_frames`             | int       | Frames in the file (`pages / n_channels`)                     |
| `n_channels`           | int       | Interleaved channels per frame                                |
| `acquisition_complete` | bool/null | `true` if the last frame has `endOfAcquisition = 1`; `null` when frame metadata was not parsed |
| `n_dropped_frames`     | int       | Frames missing according to timestamp gaps (> 1.5 frame periods) |
| `shape`                | [H, W]    | Frame dimensions in pixels                                    |
| `dtype`                | str       | Pixel data type (e.g. `"int16"`)                              |
| `si_overrides`         | dict      | `SI.*` keys whose value differs from the first file (empty for the first file) |
| `frames`               | dict      | Per-frame metadata arrays (see below)                         |

#### `files[].frames`

Arrays of per-frame values from the ImageDescription tag (one entry per frame;
for multi-channel files the first channel's page is used).

| Field                          | Type          | Description                                               |
|--------------------------------|---------------|-----------------------------------------------------------|
| `frame_numbers`                | int[]         | Global frame number assigned by ScanImage                 |
| `frame_numbers_acquisition`    | int[]         | Frame number within the acquisition (1-indexed)           |
| `timestamps_sec`               | float[]       | Frame timestamp in seconds relative to acquisition start  |
| `acq_trigger_timestamps_sec`   | float[]       | Acquisition trigger timestamp; `-1.0` if not applicable   |
| `epoch`                        | number[]      | Acquisition start time `[year, month, day, hour, minute, second]` |

---

## Validation

`save_session_metadata()` performs these checks:

1. **SI param consistency**: parameters that differ from the first file are
   stored in `si_overrides`. A warning is raised only for keys outside the
   allowlist of expected per-file differences (laser power, channel offsets,
   LUTs, acquisition state, log file counters).
2. **Frame count**: warns if `frameNumberAcquisition` does not span exactly
   `n_frames` values (interrupted or split recording).
3. **Dropped frames**: gaps in `timestamps_sec` larger than 1.5 frame periods
   are counted in `n_dropped_frames` and warned about.
4. **Acquisition completeness**: `acquisition_complete` per file.

---

## Example

```python
from bandaid.io import save_session_metadata, load_session_metadata
from bandaid.scan_geometry import ScanGeometry

path = save_session_metadata("data/HK13/HK13_20260814")
meta = load_session_metadata("data/HK13/HK13_20260814")

meta["session"]["total_frames"]                 # 9024
meta["acquisition"]["frame_rate_hz"]            # 7.5188
meta["files"][1]["si_overrides"]["SI.hBeams.powers"]   # 90
meta["files"][0]["frames"]["timestamps_sec"][-1]       # 599.963

geometry = ScanGeometry.from_session_metadata(meta, file_index=0)
geometry.pixel_times()                          # acquisition time of each column
```

## Changes from version 1

- Arrays are JSON lists instead of MATLAB strings; `epoch` is a list of numbers.
- `fill_fraction` renamed to `fill_fraction_spatial`; many scan-geometry fields added.
- New top-level `format_version` and `si_params`; new per-file `n_channels`,
  `n_dropped_frames`, `si_overrides`.
- `session_id` is inferred independently of `mouse` and `date`.
