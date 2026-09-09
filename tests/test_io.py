import json
import warnings

import numpy as np
import pytest
import tifffile

from bandaid._util import cast_like
from bandaid.io import (
    TiffChunkWriter,
    get_files,
    iter_frames,
    load_session_metadata,
    load_tiff_stack,
    parse_filename,
    parse_si_text,
    read_scanimage_metadata,
    sanitize_json,
    save_session_metadata,
    save_tiff,
    write_json,
)


@pytest.fixture
def small_stack(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.integers(-500, 2000, size=(23, 12, 16)).astype(np.int16)
    path = tmp_path / "stack.tif"
    tifffile.imwrite(path, data)
    return path, data


# --------------------------------------------------------------------- dtype

def test_cast_like_rounds_and_clips():
    x = np.array([-1.6, -0.4, 0.4, 1.6, 40000.0, -40000.0], dtype=np.float32)
    out = cast_like(x, np.int16)
    assert out.dtype == np.int16
    assert out.tolist() == [-2, 0, 0, 2, 32767, -32768]
    assert cast_like(np.array([-3.0, 70000.0]), np.uint16).tolist() == [0, 65535]
    f = cast_like(x, np.float32)
    assert f is x


# ------------------------------------------------------------------- reading

def test_load_tiff_stack_full_and_ranges(small_stack):
    path, data = small_stack
    assert np.array_equal(load_tiff_stack(path), data)
    assert np.array_equal(load_tiff_stack(path, frames=range(3, 9)), data[3:9])
    assert np.array_equal(load_tiff_stack(path, frames=slice(0, 10, 3)), data[0:10:3])
    one = load_tiff_stack(path, frames=range(5, 6))
    assert one.shape == (1, 12, 16) and np.array_equal(one[0], data[5])
    assert load_tiff_stack(path, frames=range(30, 40)).shape == (0, 12, 16)


def test_load_tiff_stack_single_frame_file(tmp_path):
    path = tmp_path / "single.tif"
    tifffile.imwrite(path, np.ones((4, 5), dtype=np.uint16))
    assert load_tiff_stack(path).shape == (1, 4, 5)


def test_load_tiff_stack_legacy_kwargs_warn(small_stack):
    path, data = small_stack
    with pytest.warns(DeprecationWarning):
        out = load_tiff_stack(path, imaging_type="2p", tiff_reader="auto")
    assert np.array_equal(out, data)


def test_iter_frames_chunks_concatenate(small_stack):
    path, data = small_stack
    starts, chunks = zip(*iter_frames(path, chunk_frames=7), strict=True)
    assert list(starts) == [0, 7, 14, 21]
    assert [c.shape[0] for c in chunks] == [7, 7, 7, 2]
    assert np.array_equal(np.concatenate(chunks), data)
    starts, chunks = zip(*iter_frames(path, chunk_frames=4, frames=range(2, 20, 3)), strict=True)
    assert list(starts) == [2, 14]
    assert np.array_equal(np.concatenate(chunks), data[2:20:3])
    # auto chunk size bounded by bytes
    sizes = [c.shape[0] for _, c in iter_frames(path, max_chunk_bytes=12 * 16 * 2 * 5)]
    assert max(sizes) == 5


# ------------------------------------------------------------------- writing

def test_save_tiff_keeps_dtype_and_rounds(tmp_path, small_stack):
    _, data = small_stack
    x = data.astype(np.float32) + 0.4
    out = save_tiff(x, tmp_path / "f32.tif")
    with tifffile.TiffFile(out) as tif:
        assert tif.series[0].dtype == np.float32
        assert np.array_equal(tif.asarray(), x)
        assert tif.pages[0].tags["Software"].value.startswith("bandaid")
    out = save_tiff(x, tmp_path / "i16.tif", dtype="int16", chunk_frames=5)
    with tifffile.TiffFile(out) as tif:
        y = tif.asarray()
        assert y.dtype == np.int16 and y.shape == data.shape
        assert np.array_equal(y, np.clip(np.rint(x), -32768, 32767))
        assert len(tif.series) == 1 and tif.series[0].shape == data.shape


def test_save_tiff_uint16_does_not_wrap(tmp_path):
    x = np.array([[[-5.0, 3.2, 70000.0]]])
    out = save_tiff(x, tmp_path / "u16.tif", dtype="uint16")
    assert load_tiff_stack(out).tolist() == [[[0, 3, 65535]]]


def test_save_tiff_bigtiff_flag(tmp_path):
    data = np.zeros((2, 4, 4), dtype=np.int16)
    with tifffile.TiffFile(save_tiff(data, tmp_path / "a.tif")) as tif:
        assert not tif.is_bigtiff  # small, known size -> classic TIFF
    with tifffile.TiffFile(save_tiff(data, tmp_path / "b.tif", bigtiff=True)) as tif:
        assert tif.is_bigtiff


def test_chunk_writer_variable_chunks(tmp_path, small_stack):
    path, data = small_stack
    out = tmp_path / "chunked.tif"
    with TiffChunkWriter(out, dtype=np.int16) as w:
        for _, chunk in iter_frames(path, chunk_frames=9):
            w.write_chunk(chunk.astype(np.float32) - 0.3)
        assert w.n_frames == 23
    with tifffile.TiffFile(out) as tif:
        assert tif.is_bigtiff  # unknown size -> BigTIFF by default
        assert tif.series[0].shape == data.shape
        assert np.array_equal(tif.asarray(), data)
    with pytest.raises(RuntimeError):
        TiffChunkWriter(tmp_path / "x.tif", dtype=np.int16).write_chunk(data)


# ----------------------------------------------------------- SI parsing/JSON

def test_parse_si_text_types():
    text = (
        "SI.hBeams.powers = 20\nSI.arr = [1 2 3]\nSI.s = 'RG'\n"
        "epoch = [2026  8 14 14  5 20.892]\nauxTrigger0 = []\nI2CData = {}\n"
        "SI.flag = true\nSI.inf = Inf\nSI.nan = NaN\nSI.cell = {'a' 'b'}\n"
        "SI.nested = {[-1 1] [-1 1]}\nbad line without equals\n"
    )
    d = parse_si_text(text)
    assert d["SI.hBeams.powers"] == 20 and d["SI.arr"] == [1, 2, 3] and d["SI.s"] == "RG"
    assert d["epoch"] == [2026, 8, 14, 14, 5, 20.892]
    assert d["auxTrigger0"] == [] and d["SI.flag"] is True
    assert d["SI.inf"] == float("inf") and np.isnan(d["SI.nan"])
    assert d["SI.cell"] == ["a", "b"] and d["SI.nested"] == [[-1, 1], [-1, 1]]
    assert "bad line without equals" not in d


def test_sanitize_json_and_write_json(tmp_path):
    obj = {
        "a": np.int64(3), "b": np.float32(1.5), "c": float("nan"), "d": float("inf"),
        "e": np.array([[1, 2]]), "f": np.bool_(True), "g": (1, 2), "h": np.dtype("int16"),
    }
    s = sanitize_json(obj)
    assert s == {"a": 3, "b": 1.5, "c": None, "d": "inf", "e": [[1, 2]], "f": True,
                 "g": [1, 2], "h": "int16"}
    p = write_json(obj, tmp_path / "x.json")
    assert json.loads(p.read_text()) == s
    with pytest.raises(ValueError):
        json.dumps(obj["c"], allow_nan=False)


# ----------------------------------------------------------------- filenames

def test_parse_filename_cases():
    assert parse_filename("HK13_20260814_50pw_00002.tif") == {
        "mouse": "HK13", "date": "20260814", "index": 2, "rest": "50pw"}
    assert parse_filename("AMD4_20260211_00002.tif") == {
        "mouse": "AMD4", "date": "20260211", "index": 2, "rest": ""}
    assert parse_filename("HK123_20260301_4364_tag_0001.tif") == {
        "mouse": "HK123", "date": "20260301", "index": 1, "rest": "4364_tag"}
    assert parse_filename("HK13_20260814")["index"] is None
    # a pattern without a capture group is wrapped instead of raising
    assert parse_filename("HK13_20260814_00007.tif", index_pattern=r"\d{3,}")["index"] == 7


def test_get_files_sorted_and_uniform(tmp_path):
    for name in ["M1_20260101_00003.tif", "M1_20260101_00001.tiff", "notes.txt", "M1_20260101_x.tif"]:
        (tmp_path / name).write_bytes(b"")
    files = get_files(tmp_path)
    assert [(p.split("/")[-1], i) for p, i in files] == [
        ("M1_20260101_00001.tiff", 1), ("M1_20260101_00003.tif", 3)]
    files = get_files(tmp_path, index_pattern=None)
    assert [i for _, i in files] == [None, None, None]
    assert [p.split("/")[-1] for p, _ in files] == sorted(p.split("/")[-1] for p, _ in files)
    with pytest.raises(FileNotFoundError):
        get_files(tmp_path, extension="npy")


# ---------------------------------------------------------------------- data

@pytest.mark.data
def test_read_scanimage_metadata_hk13(hk13_tiffs):
    meta = read_scanimage_metadata(hk13_tiffs[0])
    si = meta["si_params"]
    assert len(si) == 334
    assert meta["n_frames"] == 4512 and meta["n_channels"] == 1 and meta["n_pages"] == 4512
    assert meta["shape"] == (256, 256) and meta["dtype"] == "int16" and meta["is_bigtiff"]
    mask = si["SI.hScan2D.mask"]
    assert isinstance(mask, list) and len(mask) == 256 and all(isinstance(v, int) for v in mask)
    assert mask[0] == 158 and mask[128] == 70
    assert si["SI.hChannels.channelOffset"] == [-1811, -2748, 1283, 754]
    assert meta["frame_meta"][0]["epoch"] == [2026, 8, 14, 14, 5, 20.892]
    assert meta["frame_meta"][-1]["endOfAcquisition"] == 1
    fast = read_scanimage_metadata(hk13_tiffs[0], frames=False)
    assert fast["frame_meta"] == [] and fast["n_frames"] == 4512


@pytest.mark.data
def test_session_metadata_roundtrip_hk13(hk13_session, tmp_path):
    out = tmp_path / "session_metadata.json"
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # power stepping must not warn
        path = save_session_metadata(hk13_session, output_path=out)
    text = path.read_text()
    assert "NaN" not in text and "Infinity" not in text
    meta = load_session_metadata(path)
    assert meta["format_version"] == 2
    assert meta["session"] == {
        "mouse": "HK13", "date": "20260814", "session_id": None, "n_files": 2,
        "total_frames": 9024}
    acq = meta["acquisition"]
    assert acq["n_flyback_lines"] == 10 and acq["sample_rate_hz"] == 125000000
    assert acq["fill_fraction_spatial"] == pytest.approx(0.9)
    assert len(acq["mask_samples_per_pixel"]) == 256 and acq["imaging_system"] == "MINI2P_001"
    assert meta["si_params"]["SI.hBeams.powers"] == 50
    f0, f1 = meta["files"]
    assert f0["si_overrides"] == {}
    assert f1["si_overrides"]["SI.hBeams.powers"] == 90
    assert set(f1["si_overrides"]) <= {
        "SI.hBeams.powers", "SI.hBeams.powerFractions", "SI.hChannels.channelOffset",
        "SI.hScan2D.channelOffsets", "SI.hChannels.channelLUT"}
    for f in (f0, f1):
        assert f["n_frames"] == 4512 and f["acquisition_complete"] is True
        assert f["n_dropped_frames"] == 0
        ts = f["frames"]["timestamps_sec"]
        assert len(ts) == 4512 and ts[0] == 0.0 and 599 < ts[-1] < 601
        assert f["frames"]["frame_numbers_acquisition"][-1] == 4512
    # directory shortcut for loading
    (tmp_path / "sess").mkdir()
    (tmp_path / "sess" / "session_metadata.json").write_text(text)
    assert load_session_metadata(tmp_path / "sess") == meta
    # geometry can be rebuilt from the JSON
    from bandaid.scan_geometry import ScanGeometry

    g = ScanGeometry.from_session_metadata(meta, file_index=1)
    assert g.mask.sum() == 22278 and g.n_flyback_lines == 10
