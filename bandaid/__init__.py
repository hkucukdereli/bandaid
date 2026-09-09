"""
bandaid: detect and remove moving periodic banding (PMT ripple) from
ScanImage two-photon calcium imaging movies.
"""

from bandaid._util import package_version
from bandaid.destripe import (
    destripe_edge,
    destripe_edge_highpass,
    destripe_per_column,
    destripe_per_row,
)
from bandaid.pca_clean import remove_ripple_pca
from bandaid.ripple import (
    RippleDetection,
    RippleFit,
    RippleReport,
    detect_ripple,
    remove_ripple,
)
from bandaid.scan_geometry import ScanGeometry

__version__ = package_version()

__all__ = [
    "__version__",
    "ScanGeometry",
    "RippleDetection",
    "RippleFit",
    "RippleReport",
    "detect_ripple",
    "remove_ripple",
    "remove_ripple_pca",
    "destripe_edge",
    "destripe_edge_highpass",
    "destripe_per_column",
    "destripe_per_row",
]
