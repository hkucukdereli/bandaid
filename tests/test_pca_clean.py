import numpy as np
import pytest

pytest.importorskip("sklearn")

from bandaid.pca_clean import detect_ripple_pcs, inspect_pca, remove_ripple_pca  # noqa: E402


@pytest.fixture(scope="module")
def rank1_stack():
    """Smooth image plus a high-frequency spatial pattern with a time-varying weight."""
    rng = np.random.default_rng(0)
    T, H, W = 60, 32, 32
    yy, xx = np.mgrid[0:H, 0:W]
    image = 200 + 100 * np.exp(-((yy - 16) ** 2 + (xx - 12) ** 2) / 30.0)
    pattern = np.cos(2 * np.pi * 0.3 * xx + 0.5 * yy)      # high spatial frequency
    weight = 30 * np.sin(np.arange(T) * 0.4)
    noise = rng.normal(0, 2, size=(T, H, W))
    clean = image[None] + noise
    stack = (clean + weight[:, None, None] * pattern[None]).astype(np.int16)
    return stack, clean, pattern


def test_removes_high_frequency_component(rank1_stack):
    stack, clean, pattern = rank1_stack
    out, info = remove_ripple_pca(stack, n_noise_pcs=1, return_info=True)
    assert out.dtype == stack.dtype and out.shape == stack.shape
    err_before = np.sqrt(np.mean((stack - clean) ** 2))
    err_after = np.sqrt(np.mean((out - clean) ** 2))
    assert err_after < 0.2 * err_before
    assert info["noise_pcs"] == [0] and info["scores"].shape == (60, 20)


def test_auto_detect_flags_pattern_and_respects_no_detection(rank1_stack):
    stack, clean, pattern = rank1_stack
    out, info = remove_ripple_pca(stack, return_info=True)
    assert info["noise_pcs"] == [0]
    # a stack without any high-frequency component: nothing is removed, with a warning
    rng = np.random.default_rng(1)
    T, H, W = 40, 32, 32
    yy, xx = np.mgrid[0:H, 0:W]
    smooth = 100 + 50 * np.exp(-((yy - 10) ** 2 + (xx - 20) ** 2) / 40.0)
    slow = 20 * np.sin(np.arange(T) * 0.2)[:, None, None] * np.exp(-((yy - 20) ** 2 + (xx - 8) ** 2) / 60.0)
    plain = (smooth[None] + slow + rng.normal(0, 1, size=(T, H, W))).astype(np.int16)
    # (small 32x32 frames: the DC mask of the detector removes most of a smooth
    #  blob's spectrum, so raise the band edge to make "high frequency" meaningful)
    with pytest.warns(UserWarning, match="no ripple-like PCs"):
        out2, info2 = remove_ripple_pca(plain, freq_threshold=0.45, return_info=True)
    assert info2["noise_pcs"] == [] and np.array_equal(out2, plain)


def test_fit_frames_and_chunking_are_consistent(rank1_stack):
    stack, _, _ = rank1_stack
    a = remove_ripple_pca(stack, n_noise_pcs=1, chunk_frames=7)
    b = remove_ripple_pca(stack, n_noise_pcs=1, chunk_frames=1000)
    assert np.array_equal(a, b)
    c, info = remove_ripple_pca(stack, n_noise_pcs=1, fit_frames=20, return_info=True)
    assert len(info["fit_frames"]) == 20 and c.shape == stack.shape
    assert np.abs(c.astype(float) - a.astype(float)).mean() < 2.0


def test_detect_ripple_pcs_requires_shape_for_non_square():
    comps = np.random.default_rng(0).normal(size=(3, 6 * 8))
    with pytest.raises(ValueError):
        detect_ripple_pcs(comps, np.array([0.5, 0.3, 0.2]))
    assert detect_ripple_pcs(comps, np.array([0.5, 0.3, 0.2]), shape=(6, 8)) is not None


def test_inspect_pca_uses_reported_pcs(rank1_stack, tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    stack, _, _ = rank1_stack
    _, info = remove_ripple_pca(stack, n_noise_pcs=2, return_info=True)
    info["noise_pcs"] = [0, 2]  # non-contiguous selection must be honoured
    inspect_pca(stack, info, save_path=tmp_path / "x.png")
    assert (tmp_path / "x_components.png").exists() and (tmp_path / "x_comparison.png").exists()


def test_rejects_wrong_ndim():
    with pytest.raises(ValueError):
        remove_ripple_pca(np.zeros((4, 4)))
