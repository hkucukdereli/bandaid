"""
TIFF I/O and ScanImage metadata utilities for calcium imaging data.
"""

import os
import re
import json
import warnings
from pathlib import Path
from typing import Union

import numpy as np
from tifffile import imread, TiffFile, TiffWriter

try:
    from ScanImageTiffReader import ScanImageTiffReader
    HAS_SCANIMAGE_READER = True
except ImportError:
    HAS_SCANIMAGE_READER = False


# ---------------------------------------------------------------------------
# TIFF reading
# ---------------------------------------------------------------------------

def get_tiff_reader(
    filepath: Union[str, Path],
    imaging_type: str = '2p',
    tiff_reader: str = 'auto',
):
    """
    Open a TIFF file for frame-by-frame access without loading everything.

    Parameters
    ----------
    filepath : str or Path
        Path to the TIFF file.
    imaging_type : str, optional
        ``'1p'`` or ``'2p'``. Controls the default reader when
        *tiff_reader* is ``'auto'``. Default ``'2p'``.
    tiff_reader : str, optional
        ``'auto'``, ``'scanimage'``, or ``'tifffile'``. Default ``'auto'``.

    Returns
    -------
    ScanImageTiffReader or tifffile.TiffFile

    Examples
    --------
    >>> reader = get_tiff_reader("stack.tif", tiff_reader='tifffile')
    >>> frame_100 = reader.pages[100].asarray()
    >>> reader.close()
    """
    filepath = str(filepath)
    if tiff_reader == 'auto':
        use_scanimage = imaging_type == '2p'
    elif tiff_reader == 'scanimage':
        use_scanimage = True
    elif tiff_reader == 'tifffile':
        use_scanimage = False
    else:
        raise ValueError(f"Invalid tiff_reader '{tiff_reader}'")

    if use_scanimage:
        if not HAS_SCANIMAGE_READER:
            raise ImportError("Install: pip install scanimage-tiff-reader")
        return ScanImageTiffReader(filepath)
    return TiffFile(filepath)


def load_tiff_stack(
    filepath: Union[str, Path],
    imaging_type: str = '1p',
    tiff_reader: str = 'auto',
    verbose: bool = False,
) -> np.ndarray:
    """
    Load an entire TIFF stack into memory.

    For large files, use :func:`get_tiff_reader` instead to avoid
    loading everything at once.

    Parameters
    ----------
    filepath : str or Path
        Path to the TIFF file.
    imaging_type : str, optional
        ``'1p'`` or ``'2p'``. Default ``'1p'``.
    tiff_reader : str, optional
        ``'auto'``, ``'scanimage'``, or ``'tifffile'``. Default ``'auto'``.
    verbose : bool, optional
        Print loading info. Default ``False``.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_frames, H, W)`` or ``(H, W)`` for single frames.

    Examples
    --------
    >>> data = load_tiff_stack("trial.tif", imaging_type='2p')
    >>> data.shape
    (10000, 512, 512)
    """
    filepath = str(filepath)
    if tiff_reader == 'auto':
        use_scanimage = imaging_type == '2p'
    elif tiff_reader == 'scanimage':
        use_scanimage = True
    elif tiff_reader == 'tifffile':
        use_scanimage = False
    else:
        raise ValueError(f"Invalid tiff_reader '{tiff_reader}'")

    if use_scanimage:
        if not HAS_SCANIMAGE_READER:
            raise ImportError("Install: pip install scanimage-tiff-reader")
        if verbose:
            print(f"Loading with ScanImageTiffReader: {os.path.basename(filepath)}")
        return ScanImageTiffReader(filepath).data()
    if verbose:
        print(f"Loading with tifffile: {os.path.basename(filepath)}")
    return imread(filepath)


# ---------------------------------------------------------------------------
# TIFF writing
# ---------------------------------------------------------------------------

def save_tiff(mov: np.ndarray, fname: Union[str, Path]) -> None:
    """
    Save an image stack to a TIFF file (int16).

    Parameters
    ----------
    mov : numpy.ndarray
        Stack of shape ``(n_frames, H, W)``.
    fname : str or Path
        Output path.
    """
    with TiffWriter(str(fname)) as tif:
        for frame in np.floor(mov).astype(np.int16):
            tif.write(frame, contiguous=True)


# ---------------------------------------------------------------------------
# ScanImage metadata parsing
# ---------------------------------------------------------------------------

def _parse_si_value(value: str):
    """Convert a ScanImage value string to a Python type (bool, int, float, or str)."""
    if value in ('true', 'false'):
        return value == 'true'
    if value in ('Inf', '-Inf'):
        return float(value.replace('Inf', 'inf'))
    if value == 'NaN':
        return float('nan')
    if re.match(r'^-?\d+$', value):
        return int(value)
    if re.match(r'^-?\d+\.?\d*(?:e[+-]?\d+)?$', value):
        return float(value)
    if value.startswith("'") and value.endswith("'"):
        return value.strip("'")
    return value


def _parse_si_keyvalue(text: str) -> dict:
    """Parse ``key = value`` lines into a dict."""
    params = {}
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or '=' not in line:
            continue
        key, _, val = line.partition('=')
        params[key.strip()] = _parse_si_value(val.strip())
    return params


def read_scanimage_metadata(filepath: Union[str, Path]) -> dict:
    """
    Read all ScanImage metadata from a TIFF file.

    Reads three TIFF tags that ScanImage writes:

    - **Software** (tag 305) — acquisition parameters (``SI.*`` key-values).
    - **Artist** (tag 315) — ROI / scanfield geometry (JSON).
    - **ImageDescription** (tag 270) — per-frame info (timestamps, triggers).

    Parameters
    ----------
    filepath : str or Path
        Path to a ScanImage TIFF.

    Returns
    -------
    dict
        ``'si_params'`` : dict — parsed ``SI.*`` parameters.
        ``'roi'`` : dict or None — ROI geometry.
        ``'frame_meta'`` : list of dict — one per frame.
        ``'n_frames'`` : int — number of frames.
        ``'shape'`` : tuple — ``(H, W)`` in pixels.
        ``'dtype'`` : str — pixel type (e.g. ``'int16'``).

    Examples
    --------
    >>> meta = read_scanimage_metadata("trial.tif")
    >>> meta['si_params']['SI.hRoiManager.scanFrameRate']
    14.49
    """
    filepath = Path(filepath)
    with TiffFile(str(filepath)) as tif:
        page0 = tif.pages[0]

        si_params = {}
        sw_tag = page0.tags.get('Software')
        if sw_tag:
            si_params = _parse_si_keyvalue(sw_tag.value)

        roi = None
        artist_tag = page0.tags.get('Artist')
        if artist_tag:
            try:
                roi = json.loads(artist_tag.value)
            except json.JSONDecodeError:
                roi = None

        frame_meta = []
        for page in tif.pages:
            frame_meta.append(_parse_si_keyvalue(page.description))

        shape = page0.shape
        dtype = page0.dtype

    return {
        'si_params': si_params,
        'roi': roi,
        'frame_meta': frame_meta,
        'n_frames': len(frame_meta),
        'shape': shape,
        'dtype': str(dtype),
    }


def _extract_acquisition(si_params: dict) -> dict:
    """Pick the commonly used acquisition fields from the raw SI parameters."""
    g = si_params.get

    version_parts = [
        g('SI.VERSION_MAJOR', ''),
        g('SI.VERSION_MINOR', ''),
        g('SI.VERSION_UPDATE', ''),
    ]
    version = '.'.join(str(v) for v in version_parts)

    return {
        'scanimage_version': version,
        'acq_state': g('SI.acqState', ''),
        'frame_rate_hz': g('SI.hRoiManager.scanFrameRate', None),
        'pixels_per_line': g('SI.hRoiManager.pixelsPerLine', None),
        'lines_per_frame': g('SI.hRoiManager.linesPerFrame', None),
        'line_period_s': g('SI.hRoiManager.linePeriod', None),
        'zoom': g('SI.hRoiManager.scanZoomFactor', None),
        'fov_deg': g('SI.hRoiManager.imagingFovDeg', None),
        'fov_um': g('SI.hRoiManager.imagingFovUm', None),
        'bidirectional': g('SI.hScan2D.bidirectional', None),
        'fill_fraction': g('SI.hScan2D.fillFractionSpatial', None),
        'scanner_type': g('SI.hScan2D.scannerType', None),
        'scanner_frequency_hz': g('SI.hScan2D.scannerFrequency', None),
        'n_channels_available': g('SI.hChannels.channelsAvailable', None),
        'channels_saved': g('SI.hChannels.channelSave', None),
        'channel_offsets': g('SI.hChannels.channelOffset', None),
        'laser_power_percent': g('SI.hBeams.powers', None),
        'laser_power_fraction': g('SI.hBeams.powerFractions', None),
        'motor_position': g('SI.hMotors.axesPosition', None),
        'fastz_enabled': g('SI.hFastZ.enable', None),
        'z_step_um': g('SI.hStackManager.actualStackZStepSize', None),
    }


def _extract_frames(frame_meta: list[dict]) -> dict:
    """Reorganize per-frame metadata from a list-of-dicts into a dict-of-lists."""
    if not frame_meta:
        return {}

    frame_numbers = []
    frame_numbers_acquisition = []
    timestamps_sec = []
    acq_trigger_sec = []

    for fm in frame_meta:
        frame_numbers.append(fm.get('frameNumbers', None))
        frame_numbers_acquisition.append(fm.get('frameNumberAcquisition', None))
        timestamps_sec.append(fm.get('frameTimestamps_sec', None))
        acq_trigger_sec.append(fm.get('acqTriggerTimestamps_sec', None))

    epoch = frame_meta[0].get('epoch', None)

    return {
        'frame_numbers': frame_numbers,
        'frame_numbers_acquisition': frame_numbers_acquisition,
        'timestamps_sec': timestamps_sec,
        'acq_trigger_timestamps_sec': acq_trigger_sec,
        'epoch': epoch,
    }


# ---------------------------------------------------------------------------
# Session metadata
# ---------------------------------------------------------------------------

def save_session_metadata(
    session_dir: Union[str, Path],
    output_path: Union[str, Path, None] = None,
    mouse: str | None = None,
    date: str | None = None,
    session_id: str | None = None,
) -> Path:
    """
    Collect metadata from all TIFFs in a session folder into one JSON file.

    Writes shared acquisition parameters once, then per-file frame data
    (timestamps, frame numbers). Warns if parameters differ between files
    or if a recording was interrupted.

    See ``session_metadata_format.md`` for the full JSON schema.

    Parameters
    ----------
    session_dir : str or Path
        Folder with one or more TIFF files from a single session.
    output_path : str, Path, or None, optional
        Output JSON path. Defaults to ``session_dir/session_metadata.json``.
    mouse : str or None, optional
        Mouse ID. Inferred from folder name if not given.
    date : str or None, optional
        Date (YYYYMMDD). Inferred from folder name if not given.
    session_id : str or None, optional
        Session identifier. Inferred from folder name if not given.

    Returns
    -------
    pathlib.Path
        Path to the written JSON file.

    Examples
    --------
    >>> save_session_metadata("/data/AMD8/AMD8_20260618/AMD8_20260618_002")
    Session metadata saved to: .../session_metadata.json (918.4 KB)
      1 file(s), 13042 total frames
    """
    session_dir = Path(session_dir)
    tiff_paths = sorted(session_dir.glob('*.tif')) + sorted(session_dir.glob('*.tiff'))
    if not tiff_paths:
        raise FileNotFoundError(f"No TIFF files found in {session_dir}")

    # Infer session identifiers from folder name
    if mouse is None or date is None:
        parts = session_dir.name.split('_')
        if mouse is None and len(parts) >= 1:
            mouse = parts[0]
        if date is None and len(parts) >= 2:
            date = parts[1]
        if session_id is None and len(parts) >= 3:
            session_id = parts[2]

    # Read metadata from all files
    all_file_meta = []
    all_si_params = []
    shared_roi = None

    for tp in tiff_paths:
        meta = read_scanimage_metadata(tp)
        all_file_meta.append((tp, meta))
        all_si_params.append(meta['si_params'])
        if shared_roi is None:
            shared_roi = meta['roi']

    # Validate that acquisition params are consistent across files
    if len(all_si_params) > 1:
        baseline = all_si_params[0]
        for i, params in enumerate(all_si_params[1:], 1):
            diffs = {}
            for key in set(baseline) | set(params):
                v0 = baseline.get(key)
                vi = params.get(key)
                if v0 != vi:
                    diffs[key] = {'file_0': v0, f'file_{i}': vi}
            if diffs:
                diff_keys = ', '.join(diffs.keys())
                warnings.warn(
                    f"SI params differ between {tiff_paths[0].name} and "
                    f"{tiff_paths[i].name}: {diff_keys}"
                )

    # Build session JSON
    first_si = all_si_params[0]
    acquisition = _extract_acquisition(first_si)

    total_frames = 0
    file_entries = []

    for tp, meta in all_file_meta:
        frames = _extract_frames(meta['frame_meta'])
        n_frames_actual = meta['n_frames']
        total_frames += n_frames_actual

        last_fm = meta['frame_meta'][-1] if meta['frame_meta'] else {}
        acq_complete = last_fm.get('endOfAcquisition', 0) == 1

        last_fna = last_fm.get('frameNumberAcquisition', None)
        if last_fna is not None and int(last_fna) != n_frames_actual:
            warnings.warn(
                f"{tp.name}: frame count mismatch — file has {n_frames_actual} "
                f"pages but last frameNumberAcquisition is {last_fna}"
            )

        file_entry = {
            'filename': tp.name,
            'n_frames': n_frames_actual,
            'acquisition_complete': acq_complete,
            'shape': list(meta['shape']),
            'dtype': meta['dtype'],
            'frames': frames,
        }
        file_entries.append(file_entry)

    session_json = {
        'session': {
            'mouse': mouse,
            'date': date,
            'session_id': session_id,
            'n_files': len(tiff_paths),
            'total_frames': total_frames,
        },
        'acquisition': acquisition,
        'roi': shared_roi,
        'files': file_entries,
    }

    # Write
    if output_path is None:
        output_path = session_dir / 'session_metadata.json'
    output_path = Path(output_path)

    with open(output_path, 'w') as f:
        json.dump(session_json, f, indent=2, default=_json_default)

    incomplete = [f['filename'] for f in file_entries if not f['acquisition_complete']]
    print(f"Session metadata saved to: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")
    print(f"  {len(tiff_paths)} file(s), {total_frames} total frames")
    if incomplete:
        print(f"  WARNING: incomplete acquisition in: {', '.join(incomplete)}")

    return output_path


def load_session_metadata(path: Union[str, Path]) -> dict:
    """
    Load a ``session_metadata.json`` file.

    Parameters
    ----------
    path : str or Path
        Path to the JSON file, or a session folder containing one.

    Returns
    -------
    dict
        Session metadata (see ``session_metadata_format.md``).

    Examples
    --------
    >>> meta = load_session_metadata("/data/AMD8_20260618_002")
    >>> meta['session']['total_frames']
    13042
    """
    path = Path(path)
    if path.is_dir():
        path = path / 'session_metadata.json'
    with open(path) as f:
        return json.load(f)


def _json_default(obj):
    """Handle numpy types during JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj):
            return None
        if np.isinf(obj):
            return str(obj)
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.dtype):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
