"""M4: UI build + selection helpers (no browser needed). M6: empty start + file loading."""
import base64
import http.server
import json
import socketserver
import threading
from functools import partial
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
NPZ = FIX / "synthetic_cube.npz"
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
    assert "100–120 ADC channels" in fig.layout.title.text
    assert "7.96–9.452 keV" in map_figure(cube, 100, 120, m=0.0746, q=0.5).layout.title.text


def test_spectrum_figure_matches_golden(cube, golden):
    spec = np.asarray(spectrum_figure(cube, roi=[10, 20, 30, 40]).data[0]["y"])
    np.testing.assert_array_equal(spec, np.asarray(golden["spectrum_roi_10_20_30_40"]))
    pts = np.asarray(spectrum_figure(cube, points=[[10, 30], [19, 39]]).data[0]["y"])
    a64 = cube.astype(np.float64)
    np.testing.assert_array_equal(pts, a64[10, 30, :] + a64[19, 39, :])
    cal = np.asarray(spectrum_figure(cube, roi=[0, H, 0, W], m=0.0746, q=0.5).data[0]["x"])
    np.testing.assert_allclose(cal[0], 0.5, rtol=1e-12)
    np.testing.assert_allclose(cal[-1], 0.5 + 0.0746 * (C - 1), rtol=1e-12)
    assert spectrum_figure(cube, roi=[10, 20, 30, 40]).layout.title.text == "Box: y 10–20, x 30–40"
    assert "Points" in spectrum_figure(cube, points=[[10, 30]]).layout.title.text


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


# ---------------------------------------------------------------- M6: loading

def _cb(app, marker):
    """Find the callback whose output set contains <marker> and return its
    raw callable (dash 4.5 wraps callbacks; __wrapped__ is the original
    function taking args in signature order and returning raw values)."""
    for key, spec in app.callback_map.items():
        if marker in key:
            return getattr(spec["callback"], "__wrapped__", spec["callback"])
    raise KeyError(marker)


def _http_callback(app, key, inputs, states, changed):
    spec = app.callback_map[key]
    output = spec["output"]
    body = {
        "output": key,
        "outputs": {"id": output.component_id, "property": output.component_property},
        "inputs": [{**item, "value": inputs[f"{item['id']}.{item['property']}"]}
                   for item in spec["inputs"]],
        "state": [{**item, "value": states[f"{item['id']}.{item['property']}"]}
                  for item in spec["state"]],
        "changedPropIds": [changed],
    }
    response = app.server.test_client().post("/_dash-update-component", json=body)
    assert response.status_code == 200, response.get_data(as_text=True)
    return response.json["response"][output.component_id][output.component_property]


def test_http_selection_apply_and_figure_refresh(cube):
    app = build_app(cube)
    roi = _http_callback(app, "roi-store.data",
        {"map.relayoutData": {"xrange": [2, 5], "yrange": [1, 3]}, "apply-roi.n_clicks": 0},
        {"roi-store.data": [0, H, 0, W], "roi-y0.value": 10, "roi-y1.value": 20,
         "roi-x0.value": 30, "roi-x1.value": 40}, "map.relayoutData")
    assert roi == [1, 3, 2, 5]
    spectrum = _http_callback(app, "spectrum.figure",
        {"roi-store.data": roi, "points-store.data": [], "apply-calibration.n_clicks": 0,
         "gen.data": 0}, {"m.value": None, "q.value": None}, "roi-store.data")
    assert spectrum["layout"]["title"]["text"] == "Box: y 1–3, x 2–5"

    # A stale zoom must not override the explicit numeric ROI button.
    roi = _http_callback(app, "roi-store.data",
        {"map.relayoutData": {"xrange": [2, 5], "yrange": [1, 3]}, "apply-roi.n_clicks": 1},
        {"roi-store.data": [1, 3, 2, 5], "roi-y0.value": 7, "roi-y1.value": 9,
         "roi-x0.value": 11, "roi-x1.value": 13}, "apply-roi.n_clicks")
    assert roi == [7, 9, 11, 13]


def test_http_map_controls_and_calibration(cube):
    app = build_app(cube)
    base = {"ch-store.data": [0, C], "zmin.value": None, "zmax.value": None,
            "colorscale.value": "Viridis", "log.value": [],
            "apply-calibration.n_clicks": 0, "gen.data": 0}
    states = {"m.value": None, "q.value": None}
    fig = _http_callback(app, "map.figure", {**base, "zmin.value": 0, "zmax.value": 10},
                         states, "zmin.value")
    assert fig["data"][0]["zmin"] == 0 and fig["data"][0]["zmax"] == 10
    calibrated = _http_callback(app, "map.figure",
        {**base, "apply-calibration.n_clicks": 1}, {"m.value": 0.5, "q.value": 1},
        "apply-calibration.n_clicks")
    assert calibrated["layout"]["title"]["text"] == "Integrated over 1–257 keV"
    spectrum = _http_callback(app, "spectrum.figure",
        {"roi-store.data": [0, H, 0, W], "points-store.data": [],
         "apply-calibration.n_clicks": 1, "gen.data": 0},
        {"m.value": 0.5, "q.value": 1}, "apply-calibration.n_clicks")
    assert spectrum["layout"]["xaxis"]["title"]["text"] == "E (keV)"


def test_http_spectrum_zoom_updates_map_and_interval_button(cube):
    app = build_app(cube)
    channels = _http_callback(app, "ch-store.data",
        {"spectrum.relayoutData": {"xrange": [100, 120]}, "apply-ch.n_clicks": 0},
        {"ch-store.data": [0, C], "ch-s.value": 0, "ch-e.value": C},
        "spectrum.relayoutData")
    assert channels == [100, 120]
    mapped = _http_callback(app, "map.figure",
        {"ch-store.data": channels, "zmin.value": None, "zmax.value": None,
         "colorscale.value": "Viridis", "log.value": [],
         "apply-calibration.n_clicks": 0, "gen.data": 0},
        {"m.value": None, "q.value": None}, "ch-store.data")
    assert mapped["layout"]["title"]["text"] == "Integrated over 100–120 ADC channels"

    channels = _http_callback(app, "ch-store.data",
        {"spectrum.relayoutData": {"xrange": [100, 120]}, "apply-ch.n_clicks": 1},
        {"ch-store.data": [100, 120], "ch-s.value": 10, "ch-e.value": 20},
        "apply-ch.n_clicks")
    assert channels == [10, 20]


@pytest.fixture
def empty_app():
    return build_app(None)


def test_build_app_empty():
    app = build_app(None)
    ids = _ids(app.layout, [])
    for expected in ("map", "spectrum", "roi-store", "ch-store", "points-store",
                     "apply-roi", "apply-points", "apply-ch", "clear", "m", "q",
                     "upload", "url", "load-url", "change-file", "load-msg",
                     "apply-calibration",
                     "cube-name", "gen"):
        assert expected in ids
    # figures render the empty state without a cube (raw figure, dash 4.5 single-Output)
    f = _cb(app, "map.figure")([0, 0], None, None, "Viridis", [], 0, 0, None, None)
    assert not f.data
    f = _cb(app, "spectrum.figure")([0, 0, 0, 0], [], 0, 0, None, None)
    assert not f.data


def test_on_load_upload(empty_app, cube):
    cb = _cb(empty_app, "load-msg")
    payload = "data:application/octet-stream;base64," + base64.b64encode(NPZ.read_bytes()).decode()
    out = cb(payload, "synthetic_cube.npz", 0, "", 0, 0)
    assert "Loaded synthetic_cube.npz" in out[0]
    assert out[4] == 1  # gen bumped
    # map figure now matches the processing layer for the full channel range
    m = _cb(empty_app, "map.figure")([0, C], None, None, "Viridis", [], 0, 1, None, None)
    z = np.asarray(m.data[0]["z"])
    from xrf_visualizer.processing import map_from_channels
    np.testing.assert_array_equal(z, map_from_channels(cube, 0, C))


def test_on_load_upload_bad_bytes(empty_app, cube):
    from dash import no_update
    cb = _cb(empty_app, "load-msg")
    payload = "data:application/octet-stream;base64," + base64.b64encode(b"not a cube").decode()
    out = cb(payload, "junk.npz", 0, "", 0, 0)
    assert out[0].startswith("Upload failed")
    assert out[4] is no_update
    # state untouched: figures still empty
    m = _cb(empty_app, "map.figure")([0, C], None, None, "Viridis", [], 0, 0, None, None)
    assert not m.data


def test_on_load_change(empty_app):
    cb = _cb(empty_app, "load-msg")
    out = cb(None, None, 0, "", 1, 3)
    assert "Cube cleared" in out[0]
    assert out[4] == 4


@pytest.fixture
def http_server():
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(FIX))
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def test_on_load_link(empty_app, cube, http_server):
    cb = _cb(empty_app, "load-msg")
    out = cb(None, None, 1, f"{http_server}/synthetic_cube.npz", 0, 0)
    assert "Loaded synthetic_cube.npz" in out[0]
    out2 = cb(None, None, 1, "ftp://nope", 0, 0)
    assert out2[0].startswith("Download failed")
