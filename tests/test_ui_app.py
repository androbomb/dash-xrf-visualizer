"""M4: UI build + selection helpers (no browser needed)."""
import json
from pathlib import Path

import numpy as np
import pytest

from xrf_visualizer.ui.app import (
    _ch_from_relayout,
    _clamp_roi,
    _parse_points,
    _roi_from_relayout,
    build_app,
    map_figure,
    spectrum_figure,
)

FIX = Path(__file__).parent / "fixtures"
H, W, C = 48, 64, 512


@pytest.fixture(scope="module")
def cube():
    return np.load(FIX / "synthetic_cube.npz", allow_pickle=False)["img"]


@pytest.fixture(scope="module")
def golden():
    return json.loads((FIX / "golden.json").read_text())


def _ids(node, out):
    """Collect component ids from a live dash layout (dicts or components)."""
    if node is None or isinstance(node, (str, int, float, bool)):
        return out
    if isinstance(node, dict):
        if isinstance(node.get("id"), str):
            out.append(node["id"])
        for v in node.values():
            _ids(v, out)
        return out
    if isinstance(node, (list, tuple)):
        for v in node:
            _ids(v, out)
        return out
    if isinstance(getattr(node, "id", None), str):
        out.append(node.id)
    _ids(getattr(node, "children", None), out)
    return out


def test_build_app(cube):
    app = build_app(cube)
    ids = _ids(app.layout, [])
    for expected in ("map", "spectrum", "roi-store", "ch-store", "points-store",
                     "apply-roi", "apply-points", "apply-ch", "clear", "m", "q"):
        assert expected in ids


def test_map_figure_matches_golden(cube, golden):
    fig = map_figure(cube, 100, 120)
    z = np.asarray(fig.data[0]["z"])
    np.testing.assert_array_equal(z, np.asarray(golden["map_interval_100_120"]))
    assert z.dtype == np.float64
    fig2 = map_figure(cube, 0, C, zmin=1, zmax=1e6, colorscale="Hot", log=True)
    assert np.asarray(fig2.data[0]["z"]).shape == (H, W)


def test_spectrum_figure_matches_golden(cube, golden):
    spec = np.asarray(spectrum_figure(cube, roi=[10, 20, 30, 40]).data[0]["y"])
    np.testing.assert_array_equal(spec, np.asarray(golden["spectrum_roi_10_20_30_40"]))
    pts = np.asarray(spectrum_figure(cube, points=[[10, 30], [19, 39]]).data[0]["y"])
    a64 = cube.astype(np.float64)
    np.testing.assert_array_equal(pts, a64[10, 30, :] + a64[19, 39, :])
    cal = np.asarray(spectrum_figure(cube, roi=[0, H, 0, W], m=0.0746, q=0.5).data[0]["x"])
    np.testing.assert_allclose(cal[0], 0.5, rtol=1e-12)
    np.testing.assert_allclose(cal[-1], 0.5 + 0.0746 * (C - 1), rtol=1e-12)


def test_roi_from_relayout():
    assert _roi_from_relayout(None, H, W) is None
    assert _roi_from_relayout({}, H, W) is None
    r = _roi_from_relayout({"xrange": [10.2, 39.8], "yrange": [19.5, 9.4]}, H, W)
    assert r == [9, 20, 10, 40]
    r = _roi_from_relayout({"xrange": [-5, 999], "yrange": [0.0, 47.9]}, H, W)
    assert r == [0, 48, 0, 64]
    r = _roi_from_relayout({"yrange": [5.0, 14.9]}, H, W)  # partial: x stays full
    assert r == [5, 15, 0, 64]
    assert _roi_from_relayout({"xautorange": True, "yautorange": True}, H, W) == [0, H, 0, W]


def test_ch_from_relayout():
    assert _ch_from_relayout(None, C) is None
    assert _ch_from_relayout({}, C) is None
    assert _ch_from_relayout({"xrange": [100.3, 119.7]}, C) == [100, 120]
    assert _ch_from_relayout({"xrange": [-5, 9999]}, C) == [0, C]
    assert _ch_from_relayout({"xautorange": True}, C) == [0, C]


def test_clamp_roi():
    assert _clamp_roi(H, W, -1, 20, 30, 999) == [0, 20, 30, 64]
    assert _clamp_roi(H, W, 20, 10, 40, 30) == [10, 20, 30, 40]  # reversed sorted
    assert _clamp_roi(H, W, None, None, None, None) == [0, H, 0, W]


def test_parse_points():
    assert _parse_points("10, 19", "30, 39") == [(10, 30), (19, 39)]
    assert _parse_points("10;19", "30 39") == [(10, 30), (19, 39)]
    assert _parse_points("", "") == []
    with pytest.raises(ValueError):
        _parse_points("10, 19", "30")
    with pytest.raises(ValueError):
        _parse_points("a", "1")