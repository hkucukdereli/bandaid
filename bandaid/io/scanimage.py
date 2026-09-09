"""
ScanImage metadata: parsing TIFF tags and writing per-session JSON.

ScanImage stores three kinds of metadata in every TIFF:

- **Software** tag (305): ``SI.*`` acquisition parameters, one ``key = value``
  per line in MATLAB syntax. tifffile parses these (``TiffFile.scanimage_metadata``).
- **Artist** tag (315): ROI / scanfield geometry as JSON.
- **ImageDescription** tag (270): per-frame values (frame numbers, timestamps,
  triggers), also in ``key = value`` MATLAB syntax.

Values are converted with :func:`tifffile.matlabstr2py`, so MATLAB arrays and
cell arrays become Python lists.
"""

from __future__ import annotations

import json
import logging
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from tifffile import TiffFile, matlabstr2py

from bandaid.io.utils import parse_filename

__all__ = [
    "FORMAT_VERSION",
    "SI_OVERRIDE_ALLOWLIST",
    "parse_si_text",
    "n_channels_from_si",
    "read_scanimage_metadata",
    "save_session_metadata",
    "load_session_metadata",
    "sanitize_json",
]

log = logging.getLogger("bandaid.io")

FORMAT_VERSION = 2

#: SI parameters that are expected to differ between files of one session
#: (for example laser-power stepping); differences in these do not warn.
SI_OVERRIDE_ALLOWLIST = frozenset(
    {
        "SI.hBeams.powers",
        "SI.hBeams.powerFractions",
        "SI.hChannels.channelOffset",
        "SI.hScan2D.channelOffsets",
        "SI.hChannels.channelLUT",
        "SI.acqState",
        "SI.hScan2D.logFileCounter",
        "SI.hScan2D.logFileStem",
        "SI.loopAcqInterval",
    }
)

_DIFF_PLACEHOLDER = object()


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_si_text(text: str) -> dict[str, Any]:
    """Parse ``key = value`` lines (Software tag or per-frame description) into a dict."""
    out: dict[str, Any] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        try:
            out[key] = matlabstr2py(val)
        except Exception:
            out[key] = val
    return out


def _flatten_mask(si: dict[str, Any]) -> None:
    """``SI.hScan2D.mask`` arrives as a column vector ``[[158], [153], ...]``; flatten it."""
    mask = si.get("SI.hScan2D.mask")
    if isinstance(mask, str):
        try:
            mask = matlabstr2py(mask)
        except Exception:
            return
    if isinstance(mask, list) and mask and isinstance(mask[0], list):
        si["SI.hScan2D.mask"] = [int(np.ravel(v)[0]) for v in mask]
    elif isinstance(mask, list):
        si["SI.hScan2D.mask"] = [int(v) for v in mask]


def n_channels_from_si(si: dict[str, Any]) -> int:
    """Number of saved channels (pages per frame) from ``SI.hChannels.channelSave``."""
    return max(1, len(np.atleast_1d(si.get("SI.hChannels.channelSave", 1))))


def read_scanimage_metadata(filepath: str | Path, frames: bool = True) -> dict[str, Any]:
    """
    Read all ScanImage metadata from a TIFF file.

    Parameters
    ----------
    filepath : str or Path
    frames : bool
        Also parse the per-page ImageDescription (timestamps etc.). Set to
        ``False`` for speed when only acquisition parameters are needed.

    Returns
    -------
    dict
        ``'si_params'`` : dict — parsed ``SI.*`` parameters (arrays as lists).
        ``'roi'`` : dict or None — ROI geometry from the Artist tag.
        ``'si_version'`` : int or None — ScanImage metadata version.
        ``'frame_meta'`` : list of dict — one per page (empty if ``frames=False``).
        ``'n_pages'``, ``'n_channels'``, ``'n_frames'`` : int.
        ``'shape'`` : tuple — ``(H, W)``; ``'dtype'`` : str; ``'is_bigtiff'`` : bool.

    Examples
    --------
    >>> meta = read_scanimage_metadata("trial.tif")
    >>> meta['si_params']['SI.hRoiManager.scanFrameRate']
    7.518796992481203
    """
    filepath = Path(filepath)
    with TiffFile(str(filepath)) as tif:
        if not tif.is_scanimage:
            raise ValueError(f"{filepath.name} is not a ScanImage TIFF")
        sm = tif.scanimage_metadata or {}
        si_params = dict(sm.get("FrameData") or {})
        if not si_params:  # older files: fall back to the Software tag text
            sw = tif.pages[0].tags.get("Software")
            if sw is not None:
                si_params = parse_si_text(str(sw.value))
        _flatten_mask(si_params)
        page0 = tif.pages[0]
        n_pages = len(tif.pages)
        frame_meta = [parse_si_text(p.description or "") for p in tif.pages] if frames else []
        shape, dtype, is_bigtiff = page0.shape, page0.dtype, tif.is_bigtiff
        roi, version = sm.get("RoiGroups"), sm.get("version")

    n_channels = n_channels_from_si(si_params)
    return {
        "si_params": si_params,
        "roi": roi,
        "si_version": version,
        "frame_meta": frame_meta,
        "n_pages": n_pages,
        "n_channels": n_channels,
        "n_frames": n_pages // n_channels,
        "shape": tuple(int(s) for s in shape),
        "dtype": str(dtype),
        "is_bigtiff": bool(is_bigtiff),
    }


# ---------------------------------------------------------------------------
# Field extraction
# ---------------------------------------------------------------------------

def _extract_acquisition(si: dict[str, Any]) -> dict[str, Any]:
    """Pick the commonly used acquisition fields from the raw SI parameters."""
    g = si.get
    version = ".".join(
        str(g(k, "")) for k in ("SI.VERSION_MAJOR", "SI.VERSION_MINOR", "SI.VERSION_UPDATE")
    )
    line_period = g("SI.hRoiManager.linePeriod")
    frame_period = g("SI.hRoiManager.scanFramePeriod")
    lines = g("SI.hRoiManager.linesPerFrame")
    n_flyback = None
    if line_period and frame_period and lines:
        n_flyback = int(round(frame_period / line_period)) - int(lines)
    return {
        "scanimage_version": version,
        "imaging_system": g("SI.imagingSystem"),
        "acq_state": g("SI.acqState", ""),
        "frame_rate_hz": g("SI.hRoiManager.scanFrameRate"),
        "frame_period_s": frame_period,
        "pixels_per_line": g("SI.hRoiManager.pixelsPerLine"),
        "lines_per_frame": lines,
        "line_period_s": line_period,
        "n_flyback_lines": n_flyback,
        "flyback_time_s": g("SI.hScan2D.flybackTimePerFrame"),
        "zoom": g("SI.hRoiManager.scanZoomFactor"),
        "fov_deg": g("SI.hRoiManager.imagingFovDeg"),
        "fov_um": g("SI.hRoiManager.imagingFovUm"),
        "scan_mode": g("SI.hScan2D.scanMode"),
        "bidirectional": g("SI.hScan2D.bidirectional"),
        "uniform_sampling": g("SI.hScan2D.uniformSampling"),
        "fill_fraction_spatial": g("SI.hScan2D.fillFractionSpatial"),
        "fill_fraction_temporal": g("SI.hScan2D.fillFractionTemporal"),
        "scanner_type": g("SI.hScan2D.scannerType"),
        "scanner_frequency_hz": g("SI.hScan2D.scannerFrequency"),
        "sample_rate_hz": g("SI.hScan2D.sampleRate"),
        "pixel_bin_factor": g("SI.hScan2D.pixelBinFactor"),
        "line_phase_s": g("SI.hScan2D.linePhase"),
        "mask_samples_per_pixel": g("SI.hScan2D.mask"),
        "n_channels_available": g("SI.hChannels.channelsAvailable"),
        "channels_saved": g("SI.hChannels.channelSave"),
        "channels_active": g("SI.hChannels.channelsActive"),
        "channel_offsets": g("SI.hChannels.channelOffset"),
        "laser_power_percent": g("SI.hBeams.powers"),
        "laser_power_fraction": g("SI.hBeams.powerFractions"),
        "motor_position": g("SI.hMotors.axesPosition"),
        "fastz_enabled": g("SI.hFastZ.enable"),
        "z_step_um": g("SI.hStackManager.actualStackZStepSize"),
    }


def _extract_frames(frame_meta: list[dict[str, Any]], n_channels: int = 1) -> dict[str, Any]:
    """Reorganize per-page metadata into per-frame arrays (one entry per frame, not per page)."""
    if not frame_meta:
        return {}
    per_frame = frame_meta[::n_channels] if n_channels > 1 else frame_meta
    keys = {
        "frame_numbers": "frameNumbers",
        "frame_numbers_acquisition": "frameNumberAcquisition",
        "timestamps_sec": "frameTimestamps_sec",
        "acq_trigger_timestamps_sec": "acqTriggerTimestamps_sec",
    }
    out: dict[str, Any] = {k: [fm.get(src) for fm in per_frame] for k, src in keys.items()}
    out["epoch"] = per_frame[0].get("epoch")
    return out


def _count_dropped_frames(timestamps: list[float | None], frame_period: float | None) -> int:
    ts = np.asarray([t for t in timestamps if t is not None], dtype=float)
    if ts.size < 2:
        return 0
    dt = np.diff(ts)
    period = frame_period or float(np.median(dt))
    if not period or period <= 0:
        return 0
    gaps = dt[dt > 1.5 * period]
    return int(np.sum(np.round(gaps / period) - 1))


def _si_diff(base: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    diff = {}
    for key in set(base) | set(other):
        if base.get(key, _DIFF_PLACEHOLDER) != other.get(key, _DIFF_PLACEHOLDER):
            diff[key] = other.get(key)
    return diff


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def sanitize_json(obj: Any) -> Any:
    """Convert numpy types, NaN (-> null) and +-inf (-> "inf"/"-inf") for strict JSON."""
    if isinstance(obj, dict):
        return {str(k): sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return sanitize_json(obj.tolist())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        if math.isnan(f):
            return None
        if math.isinf(f):
            return "inf" if f > 0 else "-inf"
        return f
    if isinstance(obj, (np.dtype, Path)):
        return str(obj)
    if isinstance(obj, bytes):
        return obj.decode("latin1")
    return obj


def write_json(obj: Any, path: str | Path) -> Path:
    """Write *obj* as strict JSON (no NaN/Infinity tokens)."""
    path = Path(path)
    with open(path, "w") as f:
        json.dump(sanitize_json(obj), f, indent=2, allow_nan=False)
    return path


# ---------------------------------------------------------------------------
# Session metadata
# ---------------------------------------------------------------------------

def save_session_metadata(
    session_dir: str | Path,
    output_path: str | Path | None = None,
    mouse: str | None = None,
    date: str | None = None,
    session_id: str | None = None,
    frames: bool = True,
) -> Path:
    """
    Collect metadata from all TIFFs in a session folder into one JSON file.

    Shared acquisition parameters are written once (from the first file), the
    complete ``SI.*`` parameter set once, and per file: frame data (timestamps,
    frame numbers) plus ``si_overrides`` with any parameters that differ from
    the first file (for example stepped laser power). A warning is raised only
    for differences outside :data:`SI_OVERRIDE_ALLOWLIST`.

    See ``session_metadata_format.md`` for the JSON schema.

    Parameters
    ----------
    session_dir : str or Path
        Folder with one or more TIFF files from a single session.
    output_path : str, Path, or None
        Output JSON path. Defaults to ``session_dir/session_metadata.json``.
    mouse, date, session_id : str, optional
        Session identifiers. Each is inferred from the folder name
        ``<mouse>_<YYYYMMDD>[_<session_id>]`` when not given.
    frames : bool
        Parse per-frame metadata (default). ``False`` is faster but leaves the
        ``frames`` arrays empty.

    Returns
    -------
    pathlib.Path
        Path to the written JSON file.

    Examples
    --------
    >>> save_session_metadata("data/HK13/HK13_20260814")
    PosixPath('data/HK13/HK13_20260814/session_metadata.json')
    """
    session_dir = Path(session_dir)
    tiff_paths = sorted(session_dir.glob("*.tif")) + sorted(session_dir.glob("*.tiff"))
    if not tiff_paths:
        raise FileNotFoundError(f"No TIFF files found in {session_dir}")

    parsed = parse_filename(session_dir.name)
    parts = session_dir.name.split("_")
    if mouse is None:
        mouse = parsed["mouse"]
    if date is None:
        date = parsed["date"]
    if session_id is None and parsed["mouse"] and parsed["date"] and len(parts) >= 3:
        session_id = "_".join(parts[2:])

    metas = [read_scanimage_metadata(tp, frames=frames) for tp in tiff_paths]
    base_si = metas[0]["si_params"]
    acquisition = _extract_acquisition(base_si)
    frame_period = acquisition.get("frame_period_s")

    file_entries = []
    total_frames = 0
    for i, (tp, meta) in enumerate(zip(tiff_paths, metas, strict=True)):
        overrides = _si_diff(base_si, meta["si_params"]) if i > 0 else {}
        unexpected = sorted(set(overrides) - SI_OVERRIDE_ALLOWLIST)
        if unexpected:
            warnings.warn(
                f"SI params differ between {tiff_paths[0].name} and {tp.name}: "
                f"{', '.join(unexpected)}",
                stacklevel=2,
            )

        n_ch = meta["n_channels"]
        n_frames = meta["n_frames"]
        total_frames += n_frames
        frames_d = _extract_frames(meta["frame_meta"], n_ch)

        last_fm = meta["frame_meta"][-1] if meta["frame_meta"] else {}
        acq_complete = (last_fm.get("endOfAcquisition") == 1) if meta["frame_meta"] else None

        fna = [v for v in frames_d.get("frame_numbers_acquisition", []) if v is not None]
        if fna and (int(fna[-1]) - int(fna[0]) + 1) != n_frames:
            warnings.warn(
                f"{tp.name}: frame count mismatch: {n_frames} frames but "
                f"frameNumberAcquisition runs {fna[0]}..{fna[-1]}",
                stacklevel=2,
            )
        n_dropped = _count_dropped_frames(frames_d.get("timestamps_sec", []), frame_period)
        if n_dropped:
            warnings.warn(f"{tp.name}: {n_dropped} dropped frame(s) inferred from timestamps",
                          stacklevel=2)

        file_entries.append(
            {
                "filename": tp.name,
                "n_frames": n_frames,
                "n_channels": n_ch,
                "acquisition_complete": acq_complete,
                "n_dropped_frames": n_dropped,
                "shape": list(meta["shape"]),
                "dtype": meta["dtype"],
                "si_overrides": overrides,
                "frames": frames_d,
            }
        )

    session_json = {
        "format_version": FORMAT_VERSION,
        "session": {
            "mouse": mouse,
            "date": date,
            "session_id": session_id,
            "n_files": len(tiff_paths),
            "total_frames": total_frames,
        },
        "acquisition": acquisition,
        "roi": metas[0]["roi"],
        "si_params": base_si,
        "files": file_entries,
    }

    if output_path is None:
        output_path = session_dir / "session_metadata.json"
    output_path = write_json(session_json, output_path)

    incomplete = [f["filename"] for f in file_entries if f["acquisition_complete"] is False]
    log.info(
        "Session metadata saved to %s (%.1f KB): %d file(s), %d frames",
        output_path, output_path.stat().st_size / 1024, len(tiff_paths), total_frames,
    )
    if incomplete:
        log.warning("Incomplete acquisition in: %s", ", ".join(incomplete))
    return output_path


def load_session_metadata(path: str | Path) -> dict[str, Any]:
    """
    Load a ``session_metadata.json`` file.

    *path* may be the JSON file or a session folder containing one.

    Examples
    --------
    >>> meta = load_session_metadata("data/HK13/HK13_20260814")
    >>> meta['session']['total_frames']
    9024
    """
    path = Path(path)
    if path.is_dir():
        path = path / "session_metadata.json"
    with open(path) as f:
        return json.load(f)
