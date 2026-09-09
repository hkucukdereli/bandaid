import numpy as np
import pytest
import tifffile

from bandaid.ripple import (
    RippleDetection,
    RowGroups,
    column_profiles,
    detect_ripple,
    fit_ripple,
    remove_ripple,
    ripple_frames,
)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def test_detection_recovers_ripple_parameters(synthetic_movie):
    s = synthetic_movie
    det = detect_ripple(s.movie, s.geometry)
    assert det.detected
    assert det.harmonic_index == s.harmonic_index
    assert det.snr_db > 20
    assert det.line_power_ratio[1] > 100 and det.line_power_ratio[2] > 20
    assert det.n_harmonics_detected >= 2
    assert det.amplitude_counts == pytest.approx(s.amplitudes[0], rel=0.1)
    rel = np.asarray(det.harmonic_rel_amplitudes[:5])
    assert np.allclose(rel, s.amplitudes[:5] / s.amplitudes[0], atol=0.06)
    adv_true = s.rho_cycles_per_row * s.geometry.lines_per_frame_period
    assert det.phase_advance_cycles_per_frame == pytest.approx(adv_true, rel=0.05)
    assert det.rho_cycles_per_row == pytest.approx(s.rho_cycles_per_row, rel=0.1)
    assert det.rho_row_block == pytest.approx(s.rho_cycles_per_row, rel=0.15)
    assert det.explained_variance_k1 > 0.5 and det.explained_variance > 0.9
    assert det.to_dict()["line_power_ratio"]["1"] == det.line_power_ratio[1]


def test_no_ripple_is_rejected_and_left_untouched(no_ripple_movie):
    n = no_ripple_movie
    det = detect_ripple(n.movie, n.geometry)
    assert not det.detected
    assert det.line_power_ratio[1] < 10 and det.explained_variance_k1 < 0.1
    out, report = remove_ripple(n.movie, n.geometry)
    assert out is n.movie and not report.removed
    assert report.fit == [None] and report.metrics == [None]
    # force still runs and changes almost nothing
    forced, report = remove_ripple(n.movie, n.geometry, force=True)
    assert report.removed
    assert rms(forced.astype(np.float32) - n.movie) < 1.5


def test_removal_reaches_noise_floor(synthetic_movie):
    s = synthetic_movie
    cleaned, report = remove_ripple(s.movie, s.geometry)
    assert cleaned.shape == s.movie.shape and cleaned.dtype == s.movie.dtype
    err = cleaned.astype(np.float32) - s.truth
    assert rms(err) < 0.1 * rms(s.ripple)
    # no frame is left with a substantial residual
    per_frame = np.sqrt(np.mean(np.square(err), axis=(1, 2)))
    assert per_frame.max() < 0.2 * rms(s.ripple)
    m = report.metrics[0]
    finite = {k: v for k, v in m["line_power_ratio_after"].items() if np.isfinite(v)}
    assert finite and all(v < 2 for v in finite.values())
    assert m["dynamic_variance_removed"] > 0.8
    assert m["mean_image_change_max"] < 0.2
    f = report.fit[0]
    assert f["harmonic_index"] == 50 and f["n_outlier_frames"] == 0
    assert f["rho_cycles_per_row_median"] == pytest.approx(s.rho_cycles_per_row, rel=0.1)
    assert report.removed and report.to_dict()["removed"]


def test_tiff_path_matches_array_and_writes_report(synthetic_movie, tmp_path):
    s = synthetic_movie
    src = tmp_path / "movie.tif"
    tifffile.imwrite(src, s.movie)
    with pytest.raises(ValueError, match="scan geometry"):
        remove_ripple(src)
    ref, _ = remove_ripple(s.movie, s.geometry)
    out = tmp_path / "clean" / "movie.tif"
    res, report = remove_ripple(src, s.geometry, out=out, chunk_frames=50, report_path=tmp_path / "r.json")
    assert res is None and out.exists() and (tmp_path / "r.json").exists()
    with tifffile.TiffFile(out) as tif:
        written = tif.asarray()
        assert written.dtype == s.movie.dtype
    assert np.array_equal(written, ref)
    assert report.output == str(out) and report.source == str(src)
    det = detect_ripple(src, s.geometry)
    assert det.detected and det.harmonic_index == 50
    # detection can be reused
    res2, _ = remove_ripple(s.movie, s.geometry, detection=det)
    assert np.array_equal(res2, ref)


def test_uint16_input_does_not_wrap(synthetic_movie):
    s = synthetic_movie
    u = (s.movie.astype(np.int32) + 200).astype(np.uint16)
    cleaned, report = remove_ripple(u, s.geometry)
    assert cleaned.dtype == np.uint16 and report.removed
    assert cleaned.max() < 5000  # nothing near 65535
    ref, _ = remove_ripple(s.movie, s.geometry)
    diff = cleaned.astype(np.int32) - 200 - ref.astype(np.int32)
    assert np.abs(diff).max() <= 1


def test_float_output_and_no_mask(synthetic_movie):
    s = synthetic_movie
    cleaned, _ = remove_ripple(s.movie, s.geometry, dtype="float32", fit_mask=None)
    assert cleaned.dtype == np.float32
    assert rms(cleaned - s.truth) < 0.12 * rms(s.ripple)


def test_multichannel_array(synthetic_movie):
    from synthetic import make_synthetic_movie

    a = synthetic_movie
    b = make_synthetic_movie(seed=3, fundamental_amplitude=10.0)
    stack = np.stack([a.movie, b.movie], axis=1)  # (T, C, H, W)
    cleaned, report = remove_ripple(stack, a.geometry)
    assert cleaned.shape == stack.shape and report.n_channels == 2
    assert rms(cleaned[:, 0].astype(np.float32) - a.truth) < 0.1 * rms(a.ripple)
    assert rms(cleaned[:, 1].astype(np.float32) - b.truth) < 0.12 * rms(b.ripple)
    assert report.fit[1]["amplitude_median_counts"][0] == pytest.approx(10.0, rel=0.1)


def test_bidirectional_path(recwarn):
    from synthetic import make_synthetic_movie

    s = make_synthetic_movie(bidirectional=True, seed=5, T=160)
    det = detect_ripple(s.movie, s.geometry)
    assert det.detected and det.harmonic_index == 50
    cleaned, report = remove_ripple(s.movie, s.geometry)
    assert any("bidirectional" in str(w.message) for w in recwarn.list)
    assert rms(cleaned.astype(np.float32) - s.truth) < 0.15 * rms(s.ripple)


def test_fit_and_frames_consistency(synthetic_movie):
    s = synthetic_movie
    groups = RowGroups.from_geometry(s.geometry, 8)
    assert groups.n == 8 and groups.n_parities == 1
    prof = column_profiles(s.movie, groups)
    assert prof.shape == (s.movie.shape[0], 8, s.movie.shape[2])
    fit = fit_ripple(prof, groups, s.geometry, s.harmonic_index / s.geometry.line_period_s, n_harmonics=5)
    frames = ripple_frames(fit, range(10, 20))
    assert frames.shape == (10, *s.movie.shape[1:])
    assert rms(frames - s.ripple[10:20]) < 0.1 * rms(s.ripple)
    assert fit.amplitude.shape == (s.movie.shape[0], 1, 5)
    assert fit.phase_advance().shape == (s.movie.shape[0] - 1,)


def test_geometry_shape_mismatch_raises(synthetic_movie):
    s = synthetic_movie
    with pytest.raises(ValueError, match="geometry"):
        detect_ripple(s.movie[:, :64], s.geometry)


def test_detection_dataclass_roundtrip(synthetic_movie):
    s = synthetic_movie
    det = detect_ripple(s.movie, s.geometry)
    d = det.to_dict()
    assert isinstance(d["frames_used"], list) and d["thresholds"]["snr_db"] == 10.0
    assert isinstance(det, RippleDetection)
