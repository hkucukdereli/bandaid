"""Acceptance tests on the HK13 example session (skipped when the data is absent)."""

import numpy as np
import pytest
import tifffile

from bandaid.io import load_tiff_stack, read_scanimage_metadata
from bandaid.ripple import (
    RowGroups,
    column_profiles,
    detect_ripple,
    line_power_ratios,
    remove_ripple,
)
from bandaid.scan_geometry import ScanGeometry

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def hk13_geometry(hk13_tiffs):
    return ScanGeometry.from_si_params(read_scanimage_metadata(hk13_tiffs[0], frames=False)["si_params"])


def test_detection_on_both_files(hk13_tiffs):
    amps = []
    for path in hk13_tiffs:
        det = detect_ripple(path)
        assert det.detected
        assert det.harmonic_index == 50
        assert det.snr_db > 25
        assert det.n_harmonics_detected >= 7
        assert det.line_power_ratio[1] > 100
        assert 15 < det.amplitude_counts < 19
        assert det.rho_row_block == pytest.approx(det.rho_cycles_per_row, rel=0.15)
        amps.append(det.amplitude_counts)
    # additive artifact: same amplitude at 50% and 90% laser power
    assert abs(amps[0] - amps[1]) < 1.5


def test_removal_block_reaches_floor(hk13_tiffs, hk13_geometry):
    mov = load_tiff_stack(hk13_tiffs[0], frames=range(2000, 2600))
    cleaned, report = remove_ripple(mov, hk13_geometry)
    assert cleaned.dtype == mov.dtype and cleaned.shape == mov.shape
    m, f = report.metrics[0], report.fit[0]
    assert all(v < 2 for v in m["line_power_ratio_after"].values())
    assert m["dynamic_variance_removed"] > 0.8
    assert m["mean_image_change_max"] < 0.1
    assert f["phase_advance_mad"] < 0.01
    assert f["n_outlier_frames"] < 5
    # confirm on the cleaned pixels themselves, not just the model
    groups = RowGroups.from_geometry(hk13_geometry, 1)
    prof = column_profiles(cleaned, groups)[:, 0]
    ratios = line_power_ratios(prof - prof.mean(axis=0), hk13_geometry, f["f0_hz"], 8)
    assert all(v < 2 for v in ratios.values())
    removed = mov.astype(np.float32) - cleaned.astype(np.float32)
    assert 12 < removed.std() < 20
    assert np.abs(removed.mean(axis=0)).max() < 0.1


def test_full_file_streaming(hk13_tiffs, tmp_path):
    out = tmp_path / "clean.tif"
    _, report = remove_ripple(hk13_tiffs[1], out=out, report_path=tmp_path / "report.json")
    assert report.removed and report.runtime_s["total"] < 60
    assert all(v < 2 for v in report.metrics[0]["line_power_ratio_after"].values())
    assert report.metrics[0]["mean_image_change_max"] < 0.1
    with tifffile.TiffFile(out) as tif:
        assert tif.series[0].shape == (4512, 256, 256)
        assert tif.series[0].dtype == np.int16 and tif.is_bigtiff
    assert (tmp_path / "report.json").stat().st_size > 1000
