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
from dash import Dash

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


def test_legacy_boolean_jupyter_mode_uses_inline(monkeypatch):
    called = {}
    monkeypatch.setattr(Dash, "run", lambda self, *args, **kwargs: called.update(kwargs))
    build_app(None).run(jupyter_mode=True)
    assert called["jupyter_mode"] == "inline"


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
    logged = spectrum_figure(cube, roi=[10, 20, 30, 40], log=True)
    np.testing.assert_allclose(np.asarray(logged.data[0].y), np.log1p(spec))
    assert logged.layout.yaxis.title.text == "log1p(sum counts)"


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
    assert _roi_from_relayout({"xaxis.range[0]": 10.2, "xaxis.range[1]": 39.8,
                               "yaxis.range[0]": 19.5, "yaxis.range[1]": 9.4}, H, W) == [9, 20, 10, 40]


def test_ch_from_relayout():
    assert _ch_from_relayout(None, C) is None
    assert _ch_from_relayout({}, C) is None
    assert _ch_from_relayout({"xrange": [100.3, 119.7]}, C) == [100, 120]
    assert _ch_from_relayout({"xrange": [-5, 9999]}, C) == [0, C]
    assert _ch_from_relayout({"xautorange": True}, C) == [0, C]
    assert _ch_from_relayout({"xaxis.range[0]": 100.3, "xaxis.range[1]": 119.7}, C) == [100, 120]


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
    outputs = ([{"id": item.component_id, "property": item.component_property}
                for item in output] if isinstance(output, (list, tuple)) else
               {"id": output.component_id, "property": output.component_property})
    body = {
        "output": key,
        "outputs": outputs,
        "inputs": [{**item, "value": inputs.get(f"{item['id']}.{item['property']}",
                           0 if item["property"] == "n_clicks" else [] if item["id"] == "log" else None)}
                   for item in spec["inputs"]],
        "state": [{**item, "value": states.get(f"{item['id']}.{item['property']}",
                          [] if item["id"] == "log" else None)}
                  for item in spec["state"]],
        "changedPropIds": changed if isinstance(changed, list) else [changed],
    }
    response = app.server.test_client().post("/_dash-update-component", json=body)
    assert response.status_code == 200, response.get_data(as_text=True)
    result = response.json["response"]
    if isinstance(output, (list, tuple)):
        return tuple(result.get(item.component_id, {}).get(item.component_property) for item in output)
    return result[output.component_id][output.component_property]


def _plotly_array(value):
    if isinstance(value, dict) and "bdata" in value:
        array = np.frombuffer(base64.b64decode(value["bdata"]), dtype=value["dtype"])
        shape = value.get("shape")
        if shape is None:
            return array
        return array.reshape(tuple(map(int, shape.split(",")))) if isinstance(shape, str) else array.reshape(shape)
    return np.asarray(value)


def test_http_selection_apply_and_figure_refresh(cube):
    app = build_app(cube)
    roi = _http_callback(app, "roi-store.data",
        {"map.relayoutData": {"xaxis.range[0]": 2, "xaxis.range[1]": 5,
                              "yaxis.range[0]": 1, "yaxis.range[1]": 3},
         "apply-roi.n_clicks": 0},
        {"roi-store.data": [0, H, 0, W], "roi-y0.value": 10, "roi-y1.value": 20,
         "roi-x0.value": 30, "roi-x1.value": 40}, "map.relayoutData")
    assert roi == [1, 3, 2, 5]
    spectrum = _http_callback(app, "spectrum.figure",
        {"roi-store.data": roi, "points-store.data": [], "apply-calibration.n_clicks": 0,
         "ch-store.data": [0, C], "log.value": [], "gen.data": 0},
        {"m.value": None, "q.value": None}, "roi-store.data")
    assert spectrum["layout"]["title"]["text"] == "Box: y 1–3, x 2–5"
    np.testing.assert_array_equal(_plotly_array(spectrum["data"][0]["y"]),
                                  np.asarray(spectrum_figure(cube, roi=roi).data[0]["y"]))
    mapfig = _http_callback(app, "map.figure",
        {"ch-store.data": [0, C], "roi-store.data": roi, "points-store.data": [[2, 3]],
         "apply-scale.n_clicks": 0, "apply-calibration.n_clicks": 0, "gen.data": 0},
        {"m.value": None, "q.value": None, "zmin.value": None, "zmax.value": None,
         "colorscale.value": "Viridis", "log.value": []}, "points-store.data")
    assert mapfig["layout"]["xaxis"]["range"] == [1.5, 4.5]
    assert mapfig["layout"]["yaxis"]["range"] == [2.5, 0.5]
    assert mapfig["data"][1]["name"] == "Selected points"
    assert mapfig["data"][1]["x"] == [3] and mapfig["data"][1]["y"] == [2]

    pointfig = _http_callback(app, "spectrum.figure",
        {"roi-store.data": roi, "points-store.data": [[2, 3]], "ch-store.data": [0, C],
         "log.value": [], "apply-calibration.n_clicks": 0, "gen.data": 0},
        {"m.value": None, "q.value": None}, "points-store.data")
    np.testing.assert_array_equal(_plotly_array(pointfig["data"][0]["y"]), cube[2, 3, :].astype(np.float64))

    # A stale zoom must not override the explicit numeric ROI button.
    roi = _http_callback(app, "roi-store.data",
        {"map.relayoutData": {"xaxis.range[0]": 60, "xaxis.range[1]": 80,
                              "yaxis.range[0]": 0, "yaxis.range[1]": 140},
         "apply-roi.n_clicks": 1},
        {"roi-store.data": [1, 3, 2, 5], "roi-y0.value": 7, "roi-y1.value": 9,
         "roi-x0.value": 11, "roi-x1.value": 13},
        ["map.relayoutData", "apply-roi.n_clicks"])
    assert roi == [7, 9, 11, 13]


def test_apply_roi_zoom_on_unequal_image_axes():
    h, w, c = 145, 212, 8
    app = build_app(np.zeros((h, w, c), dtype=np.float32))
    roi = _http_callback(app, "roi-store.data",
        {"map.relayoutData": {"xaxis.range[0]": 60, "xaxis.range[1]": 80,
                              "yaxis.range[0]": 0, "yaxis.range[1]": 140},
         "apply-roi.n_clicks": 1},
        {"roi-store.data": [0, h, 0, w], "roi-y0.value": 70, "roi-y1.value": 90,
         "roi-x0.value": 60, "roi-x1.value": 80},
        ["map.relayoutData", "apply-roi.n_clicks"])
    assert roi == [70, 90, 60, 80]
    fig = _http_callback(app, "map.figure",
        {"ch-store.data": [0, c], "roi-store.data": roi, "points-store.data": [],
         "colorscale.value": "Viridis", "log.value": [], "apply-scale.n_clicks": 0,
         "apply-calibration.n_clicks": 0, "gen.data": 0, "m.value": None, "q.value": None},
        {"zmin.value": None, "zmax.value": None}, "roi-store.data")
    assert fig["layout"]["xaxis"]["range"] == [59.5, 79.5]
    assert fig["layout"]["yaxis"]["range"] == [89.5, 69.5]
    assert fig["layout"]["yaxis"]["autorange"] is False


def test_http_map_controls_and_calibration(cube):
    app = build_app(cube)
    scale_key = next(k for k, spec in app.callback_map.items() if "scale-store.data" in k)
    scale, zmin, zmax = _http_callback(app, scale_key,
        {"apply-scale.n_clicks": 1, "reset-scale.n_clicks": 0},
        {"zmin.value": 0, "zmax.value": 10}, "apply-scale.n_clicks")
    assert scale == {"zmin": 0, "zmax": 10} and (zmin, zmax) == (0, 10)
    base = {"ch-store.data": [0, C], "roi-store.data": [0, H, 0, W],
            "points-store.data": [], "colorscale.value": "Viridis", "log.value": [],
            "scale-store.data": scale, "cal-store.data": None, "gen.data": 0}
    fig = _http_callback(app, "map.figure", base, {}, "scale-store.data")
    assert fig["layout"]["coloraxis"]["cmin"] == 0
    assert fig["layout"]["coloraxis"]["cmax"] == 10

    reset, zmin, zmax = _http_callback(app, scale_key,
        {"apply-scale.n_clicks": 1, "reset-scale.n_clicks": 1},
        {"zmin.value": 0, "zmax.value": 10}, "reset-scale.n_clicks")
    assert reset == {"zmin": None, "zmax": None} and (zmin, zmax) == (None, None)
    reset_fig = _http_callback(app, "map.figure", {**base, "scale-store.data": reset},
                               {}, "scale-store.data")
    assert "cmin" not in reset_fig["layout"]["coloraxis"]
    assert "cmax" not in reset_fig["layout"]["coloraxis"]

    cal_key = next(k for k, spec in app.callback_map.items() if "cal-store.data" in k)
    calibration, message = _http_callback(app, cal_key,
        {"apply-calibration.n_clicks": 1}, {"m.value": 0.5, "q.value": 1},
        "apply-calibration.n_clicks")
    assert message == "Applied: E = 0.5 × channel + 1 keV"
    calibrated = _http_callback(app, "map.figure", {**base, "cal-store.data": calibration},
                                 {}, "cal-store.data")
    assert calibrated["layout"]["title"]["text"] == "Integrated over 1–257 keV"
    spectrum = _http_callback(app, "spectrum.figure",
        {"roi-store.data": [0, H, 0, W], "points-store.data": [], "ch-store.data": [0, C],
         "log.value": ["log"], "cal-store.data": calibration, "gen.data": 0},
        {}, "cal-store.data")
    assert spectrum["layout"]["xaxis"]["title"]["text"] == "E (keV; m=0.5, q=1)"
    assert spectrum["layout"]["xaxis"]["range"] == [0.75, 256.75]
    assert spectrum["layout"]["yaxis"]["title"]["text"] == "log1p(sum counts)"


def test_xrf_line_select_applies_calibrated_energy_width(cube):
    app = build_app(cube)
    line_key = next(k for k, spec in app.callback_map.items() if "line-msg.children" in k)
    out = _http_callback(app, line_key, {"apply-line.n_clicks": 1},
        {"line-select.value": 1.486, "line-width.value": 0.2,
         "cal-store.data": {"m": 0.01, "q": 0}, "ch-store.data": [0, C]},
        "apply-line.n_clicks")
    assert out == ([139, 159], "Integrating 1.486 ± 0.1 keV")


def test_clear_selection_resets_both_plots(cube):
    app = build_app(cube)
    key = next(k for k in app.callback_map if "points-store.data" in k)
    selections = _http_callback(app, key,
        {"apply-points.n_clicks": 1, "clear.n_clicks": 0},
        {"pt-rows.value": "2", "pt-cols.value": "3"}, "apply-points.n_clicks")
    assert selections[0] == [[2, 3]]

    cleared = _http_callback(app, key,
        {"apply-points.n_clicks": 1, "clear.n_clicks": 1},
        {"pt-rows.value": "2", "pt-cols.value": "3"}, "clear.n_clicks")
    assert cleared == ([], "", [0, H, 0, W], [0, C])
    mapfig = _http_callback(app, "map.figure",
        {"ch-store.data": cleared[3], "roi-store.data": cleared[2],
         "points-store.data": cleared[0], "colorscale.value": "Viridis", "log.value": [],
         "apply-scale.n_clicks": 0, "apply-calibration.n_clicks": 0, "gen.data": 0},
        {"m.value": None, "q.value": None, "zmin.value": None, "zmax.value": None}, "points-store.data")
    assert len(mapfig["data"]) == 1
    assert mapfig["layout"]["xaxis"]["range"] == [-0.5, W - 0.5]
    spectrum = _http_callback(app, "spectrum.figure",
        {"roi-store.data": cleared[2], "points-store.data": cleared[0],
         "ch-store.data": cleared[3], "log.value": [],
         "apply-calibration.n_clicks": 0, "gen.data": 0},
        {"m.value": None, "q.value": None}, "roi-store.data")
    assert spectrum["layout"]["title"]["text"] == f"Box: y 0–{H}, x 0–{W}"
    assert spectrum["layout"]["xaxis"]["range"] == [-0.5, C - 0.5]


def test_http_spectrum_zoom_updates_map_and_interval_button(cube):
    app = build_app(cube)
    channels = _http_callback(app, "ch-store.data",
        {"spectrum.relayoutData": {"xaxis.range[0]": 100, "xaxis.range[1]": 120},
         "apply-ch.n_clicks": 0},
        {"ch-store.data": [0, C], "ch-s.value": 0, "ch-e.value": C},
        "spectrum.relayoutData")
    assert channels == [100, 120]
    mapped = _http_callback(app, "map.figure",
        {"ch-store.data": channels, "zmin.value": None, "zmax.value": None,
         "roi-store.data": [0, H, 0, W],
         "colorscale.value": "Viridis", "log.value": [],
         "apply-scale.n_clicks": 0, "apply-calibration.n_clicks": 0, "gen.data": 0},
        {"m.value": None, "q.value": None}, "ch-store.data")
    assert mapped["layout"]["title"]["text"] == "Integrated over 100–120 ADC channels"
    np.testing.assert_array_equal(_plotly_array(mapped["data"][0]["z"]),
                                  np.asarray(map_figure(cube, 100, 120).data[0]["z"]))

    channels = _http_callback(app, "ch-store.data",
        {"spectrum.relayoutData": {"xaxis.range[0]": 100, "xaxis.range[1]": 120},
         "apply-ch.n_clicks": 1},
        {"ch-store.data": [100, 120], "ch-s.value": 10, "ch-e.value": 20},
        "apply-ch.n_clicks")
    assert channels == [10, 20]
    selected_spectrum = _http_callback(app, "spectrum.figure",
        {"roi-store.data": [0, H, 0, W], "points-store.data": [], "ch-store.data": channels,
         "log.value": [], "apply-calibration.n_clicks": 0, "gen.data": 0},
        {"m.value": None, "q.value": None}, "ch-store.data")
    assert selected_spectrum["layout"]["xaxis"]["range"] == [9.5, 19.5]


@pytest.fixture
def empty_app():
    return build_app(None)


def test_build_app_empty():
    app = build_app(None)
    ids = _ids(app.layout, [])
    for expected in ("map", "spectrum", "roi-store", "ch-store", "points-store",
                     "apply-roi", "apply-points", "apply-ch", "clear", "m", "q",
                     "upload", "url", "load-url", "change-file", "load-msg",
                     "apply-calibration", "apply-scale",
                     "reset-scale", "cal-msg", "line-select", "line-width",
                     "apply-line", "scale-store", "cal-store", "cube-name", "gen"):
        assert expected in ids
    # figures render the empty state without a cube (raw figure, dash 4.5 single-Output)
    f = _cb(app, "map.figure")([0, 0], [0, 0, 0, 0], [], "Viridis", [],
                                {"zmin": None, "zmax": None}, None, 0)
    assert not f.data
    f = _cb(app, "spectrum.figure")([0, 0, 0, 0], [], [0, 0], [], None, 0)
    assert not f.data


def test_on_load_upload(empty_app, cube):
    cb = _cb(empty_app, "load-msg")
    payload = "data:application/octet-stream;base64," + base64.b64encode(NPZ.read_bytes()).decode()
    out = cb(payload, "synthetic_cube.npz", 0, "", 0, 0)
    assert "Loaded synthetic_cube.npz" in out[0]
    assert out[4] == 1  # gen bumped
    # map figure now matches the processing layer for the full channel range
    m = _cb(empty_app, "map.figure")([0, C], [0, H, 0, W], [], "Viridis", [],
                                      {"zmin": None, "zmax": None}, None, 1)
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
    m = _cb(empty_app, "map.figure")([0, C], [0, H, 0, W], [], "Viridis", [],
                                      {"zmin": None, "zmax": None}, None, 0)
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
