"""File I/O: TIFF stacks, ScanImage metadata, and lab naming conventions."""

from bandaid.io.scanimage import (
    load_session_metadata,
    n_channels_from_si,
    parse_si_text,
    read_scanimage_metadata,
    sanitize_json,
    save_session_metadata,
    write_json,
)
from bandaid.io.tiff import (
    TiffChunkWriter,
    get_tiff_reader,
    iter_frames,
    load_tiff_stack,
    n_channels_in_tiff,
    open_tiff,
    save_tiff,
)
from bandaid.io.utils import get_dirs, get_files, get_subdir, parse_filename

__all__ = [
    "read_scanimage_metadata",
    "save_session_metadata",
    "load_session_metadata",
    "parse_si_text",
    "n_channels_from_si",
    "sanitize_json",
    "write_json",
    "open_tiff",
    "load_tiff_stack",
    "iter_frames",
    "save_tiff",
    "TiffChunkWriter",
    "n_channels_in_tiff",
    "get_tiff_reader",
    "parse_filename",
    "get_files",
    "get_dirs",
    "get_subdir",
]
