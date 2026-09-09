import numpy as np
import pytest

from bandaid.scan_geometry import ScanGeometry

W, H, F_RES, FS, FFS = 256, 256, 2000.0, 125e6, 0.9
T_LINE, T_FRAME = 1 / F_RES, 0.133


def arcsin_geometry(**kw) -> ScanGeometry:
    base = dict(
        n_pixels=W, n_lines=H, line_period_s=T_LINE, frame_period_s=T_FRAME,
        sample_rate_hz=FS, scanner_frequency_hz=F_RES, fill_fraction_spatial=FFS,
    )
    base.update(kw)
    return ScanGeometry(**base)


def test_fill_fraction_roundtrip():
    g = arcsin_geometry()
    assert g.fill_fraction_temporal == pytest.approx(2 / np.pi * np.arcsin(FFS))
    g2 = arcsin_geometry(fill_fraction_spatial=None, fill_fraction_temporal=g.fill_fraction_temporal)
    assert g2.fill_fraction_spatial == pytest.approx(FFS)


def test_flyback_lines():
    assert arcsin_geometry().n_flyback_lines == 10
    assert arcsin_geometry().lines_per_frame_period == 266
    assert arcsin_geometry(frame_period_s=None).n_flyback_lines == 0


def test_arcsin_dwell_matches_known_scanimage_mask_values():
    d = arcsin_geometry().dwell_samples()
    assert d[W // 2] == pytest.approx(70, abs=0.5)
    assert d[0] == pytest.approx(158, abs=0.5)
    assert d[0] == pytest.approx(d[-1])
    assert np.all(np.diff(d[: W // 2]) < 0)  # dwell shrinks toward the center


def test_mask_and_arcsin_pixel_times_agree():
    g0 = arcsin_geometry()
    mask = np.rint(g0.dwell_samples()).astype(int)
    g1 = arcsin_geometry(mask=mask)
    t0, t1 = g0.pixel_times(), g1.pixel_times()
    assert np.all(np.diff(t1) > 0)
    assert t1[0] > 0 and t1[-1] < g1.line_acq_time_s
    rel = np.max(np.abs((t0 - t0[W // 2]) - (t1 - t1[W // 2]))) / g1.line_acq_time_s
    assert rel < 1e-3
    assert g1.line_acq_time_s == pytest.approx(mask.sum() / FS)
    assert g0.line_acq_time_s == pytest.approx(g0.fill_fraction_temporal / (2 * F_RES))


def test_mask_coercion_from_nested_list_and_matlab_string():
    mask = np.rint(arcsin_geometry().dwell_samples()).astype(int)
    nested = [[int(v)] for v in mask]
    g = arcsin_geometry(mask=nested)
    assert g.mask.shape == (W,) and g.mask.dtype.kind == "i"
    g2 = arcsin_geometry(mask="[" + ";".join(str(int(v)) for v in mask) + "]")
    assert np.array_equal(g2.mask, mask)
    with pytest.raises(ValueError):
        arcsin_geometry(mask=mask[:-1])


def test_from_si_params_minimal_dict():
    mask = np.rint(arcsin_geometry().dwell_samples()).astype(int)
    si = {
        "SI.hRoiManager.pixelsPerLine": W,
        "SI.hRoiManager.linesPerFrame": H,
        "SI.hRoiManager.linePeriod": T_LINE,
        "SI.hRoiManager.scanFramePeriod": T_FRAME,
        "SI.hScan2D.sampleRate": FS,
        "SI.hScan2D.mask": [[int(v)] for v in mask],
        "SI.hScan2D.scannerFrequency": F_RES,
        "SI.hScan2D.fillFractionSpatial": FFS,
        "SI.hScan2D.fillFractionTemporal": 0.7128674137425877,
        "SI.hScan2D.bidirectional": False,
        "SI.hScan2D.uniformSampling": False,
        "SI.hScan2D.scanMode": "resonant",
        "SI.hChannels.channelSave": 1,
        "SI.imagingSystem": "MINI2P_001",
    }
    g = ScanGeometry.from_si_params(si)
    assert g.n_channels == 1 and g.imaging_system == "MINI2P_001"
    assert g.mask is not None and g.n_flyback_lines == 10
    g2 = ScanGeometry.from_si_params({**si, "SI.hChannels.channelSave": [1, 2]})
    assert g2.n_channels == 2
    meta = {"si_params": si, "files": [{"si_overrides": {}}, {"si_overrides": {"SI.hRoiManager.linesPerFrame": 128}}]}
    assert ScanGeometry.from_session_metadata(meta, 1).n_lines == 128


def test_uniform_geometry_is_linear():
    g = ScanGeometry.uniform(n_pixels=100, n_lines=10, line_period_s=1e-3, acq_fraction=0.8)
    t = g.pixel_times()
    assert np.allclose(np.diff(t), np.diff(t)[0])
    assert t[-1] < 0.8e-3 and g.line_acq_time_s == pytest.approx(0.8e-3)
    assert np.allclose(g.dwell_samples(), g.dwell_samples()[0])


def test_pixel_times_2d_bidirectional_reverses_odd_rows():
    g = arcsin_geometry(bidirectional=True, n_lines=4)
    t = g.pixel_times()
    t2 = g.pixel_times_2d()
    assert t2.shape == (4, W)
    assert np.allclose(t2[0], t)
    assert np.allclose(t2[1], T_LINE + (g.line_acq_time_s - t))
    assert np.allclose(t2[2], 2 * T_LINE + t)
    uni = arcsin_geometry(n_lines=4).pixel_times_2d()
    assert np.allclose(uni[1], T_LINE + t)


def test_integration_gain_limits():
    g = arcsin_geometry()
    assert np.allclose(g.integration_gain(0.0), 1.0)
    gain = g.integration_gain(100e3)
    assert np.all(gain <= 1.0) and np.all(gain > 0.9)
    # a period equal to the edge dwell time is averaged away
    f_edge = FS / g.dwell_samples()[0]
    assert g.integration_gain(f_edge)[0] < 0.05


def test_to_dict_is_json_friendly():
    import json

    json.dumps(arcsin_geometry().to_dict())


@pytest.mark.data
def test_hk13_mask_consistency(hk13_tiffs):
    import tifffile

    with tifffile.TiffFile(str(hk13_tiffs[0])) as tif:
        si = tif.scanimage_metadata["FrameData"]
    g = ScanGeometry.from_si_params(si)
    assert g.mask is not None and g.mask.size == 256
    expected_sum = g.sample_rate_hz * g.fill_fraction_temporal / (2 * g.scanner_frequency_hz)
    assert abs(g.mask.sum() - expected_sum) < 2
    assert g.n_flyback_lines == 10
    t_mask = g.pixel_times()
    t_arcsin = g.replace(mask=None).pixel_times()
    rel = np.max(np.abs((t_mask - t_mask[128]) - (t_arcsin - t_arcsin[128]))) / g.line_acq_time_s
    assert rel < 1e-3
