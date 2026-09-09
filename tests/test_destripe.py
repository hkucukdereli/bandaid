import numpy as np
import pytest

from bandaid.destripe import (
    destripe_edge,
    destripe_edge_highpass,
    destripe_per_column,
    destripe_per_row,
    destripe_profile,
)


@pytest.fixture
def striped():
    rng = np.random.default_rng(0)
    base = rng.integers(100, 200, size=(3, 40, 48)).astype(np.int16)
    stripes = (30 * np.sin(np.arange(48) / 3.0)).astype(np.int16)
    return base, base + stripes[None, None, :], stripes


def test_column_stripes_removed_and_dtype_kept(striped):
    base, data, stripes = striped
    def stripe_amplitude(x):
        return np.abs((x.astype(float) - x.astype(float).mean()).mean(axis=(0, 1))).max()

    assert stripe_amplitude(data) > 25
    out = destripe_profile(data, "columns", "median")
    assert out.dtype == np.int16 and out.shape == data.shape
    assert stripe_amplitude(out) < 10  # median of 40 noisy rows leaves a little residual
    assert stripe_amplitude(destripe_profile(data, "columns", "mean")) < 1.0
    assert abs(out.mean() - data.mean()) < 0.6  # frame mean preserved


def test_rows_edge_highpass_and_percentile(striped):
    base, data, _ = striped
    rows = destripe_profile(data.transpose(0, 2, 1), "rows", "mean")
    assert rows.shape == (3, 48, 40)
    edge = destripe_profile(data, "columns", "mean", n_edge=5)
    assert edge.shape == data.shape
    hp = destripe_profile(data, "columns", "median", highpass_size=9)
    assert hp.dtype == np.int16
    pct = destripe_profile(data, "columns", 25)
    assert pct.dtype == np.int16
    f32 = destripe_profile(data, "columns", "median", dtype="float32")
    assert f32.dtype == np.float32


def test_single_frame_and_validation(striped):
    _, data, _ = striped
    single = destripe_profile(data[0], "columns")
    assert single.shape == data[0].shape and single.ndim == 2
    with pytest.raises(ValueError):
        destripe_profile(data, "diagonal")
    with pytest.raises(ValueError):
        destripe_profile(data, "columns", "medain")
    with pytest.raises(ValueError):
        destripe_profile(data, "columns", 150)
    with pytest.raises(ValueError):
        destripe_profile(np.zeros((2, 2, 2, 2)))


def test_uint16_does_not_wrap():
    u = np.zeros((1, 32, 32), dtype=np.uint16)
    u[0, :, 10] = 500
    u[0, 5, 10] = 0
    out = destripe_profile(u, "columns", "median")
    assert out.dtype == np.uint16 and out.max() < 1000 and out.min() == 0


def test_deprecated_wrappers_match_new_api(striped):
    _, data, _ = striped
    with pytest.warns(DeprecationWarning):
        a = destripe_edge(data, n_edge=10, axis=0, method="mean")
    assert np.array_equal(a, destripe_profile(data, "columns", "mean", n_edge=10))
    with pytest.warns(DeprecationWarning):
        b = destripe_edge_highpass(data, n_edge=10, axis=1, method="median", highpass_size=7)
    assert np.array_equal(b, destripe_profile(data, "rows", "median", n_edge=10, highpass_size=7))
    with pytest.warns(DeprecationWarning):
        c = destripe_per_column(data, percentile=30)
    assert np.array_equal(c, destripe_profile(data, "columns", 30))
    with pytest.warns(DeprecationWarning):
        d = destripe_per_row(data, method="mean")
    assert np.array_equal(d, destripe_profile(data, "rows", "mean"))
    assert a.dtype == np.int16  # destripe_edge used to return float32
