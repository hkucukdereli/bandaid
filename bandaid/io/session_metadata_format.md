# Session Metadata JSON Format

`save_session_metadata()` reads all ScanImage TIFFs in a session folder and
writes a single `session_metadata.json`. This document describes the JSON
structure.

## Source TIFF tags

ScanImage writes three metadata tags per TIFF:

| TIFF tag            | Tag # | Scope     | Content                                     |
|---------------------|-------|-----------|---------------------------------------------|
| **Software**        | 305   | Per-file  | `SI.*` key-value acquisition parameters     |
| **Artist**          | 315   | Per-file  | JSON with ROI / scanfield geometry           |
| **ImageDescription**| 270   | Per-frame | Frame number, timestamp, triggers, epoch     |

Software and Artist are identical across files in a normal session (a warning
is raised if they differ). ImageDescription varies per frame.

---

## JSON structure

```
session_metadata.json
├── session        — identifiers and counts
├── acquisition    — shared acquisition parameters (from Software tag)
├── roi            — scanfield geometry (from Artist tag, verbatim JSON)
└── files[]        — one entry per TIFF file
    ├── filename, n_frames, acquisition_complete, shape, dtype
    └── frames     — per-frame arrays (from ImageDescription tag)
```

---

### `session`

Top-level identifiers for the recording session.

| Field          | Type      | Description                                              |
|----------------|-----------|----------------------------------------------------------|
| `mouse`        | str/null  | Mouse ID, inferred from folder name (e.g. `"AMD4"`)     |
| `date`         | str/null  | Date string, inferred from folder name (`"20260211"`)    |
| `session_id`   | str/null  | Session ID if present in folder name (e.g. `"002"`)     |
| `n_files`      | int       | Number of TIFF files in the session                      |
| `total_frames` | int       | Total frame count across all files                       |

Folder name convention: `<mouse>_<YYYYMMDD>[_<session_id>]`.
Fields can also be passed explicitly to `save_session_metadata()`.

---

### `acquisition`

Shared acquisition parameters extracted from the ScanImage `Software` tag.
These are the same across all files in a session. Values come from the first
TIFF file.

| Field                  | Type       | SI source key                        | Description                                    |
|------------------------|------------|--------------------------------------|------------------------------------------------|
| `scanimage_version`    | str        | `SI.VERSION_MAJOR/MINOR/UPDATE`      | ScanImage version (e.g. `"2023.1.1"`)         |
| `acq_state`            | str        | `SI.acqState`                        | Acquisition mode (`"grab"`, `"loop"`, etc.)    |
| `frame_rate_hz`        | float      | `SI.hRoiManager.scanFrameRate`       | Frame rate in Hz                               |
| `pixels_per_line`      | int        | `SI.hRoiManager.pixelsPerLine`       | Pixels per scan line (image width)             |
| `lines_per_frame`      | int        | `SI.hRoiManager.linesPerFrame`       | Lines per frame (image height)                 |
| `line_period_s`        | float      | `SI.hRoiManager.linePeriod`          | Time per scan line in seconds                  |
| `zoom`                 | float      | `SI.hRoiManager.scanZoomFactor`      | Optical zoom factor                            |
| `fov_deg`              | str        | `SI.hRoiManager.imagingFovDeg`       | Field of view in degrees (MATLAB matrix str)   |
| `fov_um`               | str        | `SI.hRoiManager.imagingFovUm`        | Field of view in microns (MATLAB matrix str)   |
| `bidirectional`        | bool       | `SI.hScan2D.bidirectional`           | Bidirectional scanning enabled                 |
| `fill_fraction`        | float      | `SI.hScan2D.fillFractionSpatial`     | Spatial fill fraction of scan                  |
| `scanner_type`         | str        | `SI.hScan2D.scannerType`             | Scanner type (`"RG"`, `"RGG"`, etc.)           |
| `scanner_frequency_hz` | int/float  | `SI.hScan2D.scannerFrequency`        | Resonant scanner frequency in Hz               |
| `n_channels_available` | int        | `SI.hChannels.channelsAvailable`     | Total number of available channels             |
| `channels_saved`       | int/str    | `SI.hChannels.channelSave`           | Which channel(s) are saved to disk             |
| `channel_offsets`      | str        | `SI.hChannels.channelOffset`         | Per-channel ADC offsets (MATLAB array str)      |
| `laser_power_percent`  | int/float  | `SI.hBeams.powers`                   | Laser power as percentage of max               |
| `laser_power_fraction` | float      | `SI.hBeams.powerFractions`           | Laser power as fraction (0-1)                  |
| `motor_position`       | str        | `SI.hMotors.axesPosition`            | Stage XYZ position (MATLAB array str)           |
| `fastz_enabled`        | bool       | `SI.hFastZ.enable`                   | Fast Z (piezo/ETL) enabled                     |
| `z_step_um`            | str/float  | `SI.hStackManager.actualStackZStepSize` | Z step size in microns                      |

Note: some fields remain as MATLAB-format strings (e.g. `fov_um`, `channel_offsets`,
`motor_position`). These are carried through as-is from ScanImage rather than
parsed into arrays, since their format can vary.

---

### `roi`

The full ROI geometry JSON from the ScanImage `Artist` tag, stored verbatim.
Key fields within the nested structure:

| Path (within `roi`)                                      | Description                              |
|----------------------------------------------------------|------------------------------------------|
| `RoiGroups.imagingRoiGroup.rois.UserData.imagingSystem`  | Scanner system name                      |
| `RoiGroups.imagingRoiGroup.rois.UserData.scanZoomFactor` | Zoom (matches `acquisition.zoom`)        |
| `RoiGroups.imagingRoiGroup.rois.UserData.pixelsPerLine`  | Pixels (matches `acquisition`)           |
| `RoiGroups.imagingRoiGroup.rois.scanfields.centerXY`     | Scanfield center `[x, y]`               |
| `RoiGroups.imagingRoiGroup.rois.scanfields.sizeXY`       | Scanfield size `[x, y]`                 |
| `RoiGroups.imagingRoiGroup.rois.scanfields.pixelToRefTransform` | 3x3 pixel-to-reference transform  |
| `RoiGroups.imagingRoiGroup.rois.scanfields.affine`       | 3x3 affine transform                    |

---

### `files[]`

One entry per TIFF file in the session, ordered by filename.

| Field                  | Type   | Description                                                   |
|------------------------|--------|---------------------------------------------------------------|
| `filename`             | str    | TIFF filename (not full path)                                 |
| `n_frames`             | int    | Actual number of frames (pages) in the file                   |
| `acquisition_complete` | bool   | `true` if the last frame has `endOfAcquisition=1`; `false` if the recording was interrupted |
| `shape`                | [H, W] | Frame dimensions in pixels                                    |
| `dtype`                | str    | Pixel data type (e.g. `"int16"`)                              |
| `frames`               | dict   | Per-frame metadata arrays (see below)                         |

#### `files[].frames`

Arrays of per-frame values extracted from the `ImageDescription` tag of each
TIFF page. All arrays have length `n_frames`.

| Field                          | Type          | Description                                               |
|--------------------------------|---------------|-----------------------------------------------------------|
| `frame_numbers`                | int[]         | Global frame number assigned by ScanImage                 |
| `frame_numbers_acquisition`    | int[]         | Frame number within this acquisition (1-indexed, resets per file) |
| `timestamps_sec`               | float[]       | Frame timestamp in seconds (relative to acquisition start)|
| `acq_trigger_timestamps_sec`   | float[]       | Acquisition trigger timestamp; `-1.0` if not applicable   |
| `epoch`                        | str           | Acquisition start time as MATLAB vector (e.g. `"[2026  2 11 13 18 59.046]"`) |

---

## Validation

`save_session_metadata()` performs these checks:

1. **SI param consistency** — warns if any `Software` tag parameters differ
   between files (e.g. power stepping, channel offset drift).

2. **Frame count** — warns if the last frame's `frameNumberAcquisition` does
   not match the actual page count (indicates an interrupted recording).

3. **Acquisition completeness** — sets `acquisition_complete=false` per file
   if the last frame's `endOfAcquisition` flag is not `1`.

---

## Example

```python
from bandaid.io import save_session_metadata, load_session_metadata

# Write
save_session_metadata("/data/AMD8/AMD8_20260618/AMD8_20260618_002")

# Read
meta = load_session_metadata("/data/AMD8/AMD8_20260618/AMD8_20260618_002")
meta['session']['total_frames']          # 13042
meta['acquisition']['frame_rate_hz']     # 14.49
meta['files'][0]['acquisition_complete']  # True
meta['files'][0]['frames']['timestamps_sec'][-1]  # 899.829
```
