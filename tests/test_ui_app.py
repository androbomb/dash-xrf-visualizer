"""M4: UI build + selection helpers (no browser needed). M6: empty start + file loading."""
import base64
import http.server
import json
import socketserver
import struct
import threading
import zlib
from functools import partial
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
from dash import Dash
from plotly.colors import sample_colorscale

from xrf_visualizer.ui.app import (
    _add_elemental_line,
    _ch_from_relayout,
    _clamp_roi,
    _calibration_from_points,
    _parse_calibration_point,
    _line_channel_range,
    _map_png_bytes,
    _parse_points,
    _roi_from_relayout,
    _roi_from_selection,
    build_app,
    map_figure,
    spectrum_figure,
)
from xrf_visualizer.processing import map_from_channels

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


def _component(node, component_id):
    if getattr(node, "id", None) == component_id:
        return node
    children = getattr(node, "children", None)
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        found = _component(child, component_id) if child is not None else None
        if found is not None:
            return found
    return None


def test_build_app(cube):
    app = build_app(cube)
    ids = _ids(app.layout, [])
    for expected in ("map", "spectrum", "roi-store", "ch-store", "points-store",
                     "apply-roi", "apply-points", "apply-ch", "clear", "m", "q",
                     "show-energy", "xmode-store"):
        assert expected in ids
    clientside = "\n".join(app._inline_scripts)
    assert _component(app.layout, "map").config["doubleClick"] == "reset"
    assert 'addEventListener("dblclick"' in clientside
    assert 'set_props("map-reset-store"' in clientside
    assert 'getElementById("q").value' in clientside


def test_dark_theme_control_labels_and_reversed_gray_default(cube):
    app = build_app(cube)
    colorscale = _component(app.layout, "colorscale")
    assert colorscale.value == "Greys_r"
    assert sample_colorscale("Greys_r", [0, 1]) == ["rgb(0, 0, 0)", "rgb(255, 255, 255)"]
    for component_id in ("log", "elemental-shared-scale"):
        assert _component(app.layout, component_id).labelStyle["color"] == "var(--bs-body-color)"
    assert "Pick line" in str(app.layout)


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
    np.testing.assert_allclose(np.asarray(fig2.data[0]["z"]),
                               np.log1p(map_from_channels(cube, 0, C)))
    assert fig2.layout.coloraxis.cmin == pytest.approx(np.log1p(1))
    assert fig2.layout.coloraxis.cmax == pytest.approx(np.log1p(1e6))
    assert "100–120 ADC channels" in fig.layout.title.text
    assert "7.96–9.452 keV" in map_figure(cube, 100, 120, m=0.0746, q=0.5).layout.title.text
    assert fig.layout.xaxis.scaleanchor == "y"


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


def test_roi_from_rectangular_selection():
    selection = {"range": {"x": [10.2, 20.8], "y": [4.1, 9.7]}}
    assert _roi_from_selection(selection, H, W) == [4, 10, 10, 21]
    assert _roi_from_selection(None, H, W) is None


def test_rectangular_graph_selection_updates_roi_and_aspect(cube):
    app = build_app(cube)
    roi = _http_callback(app, "roi-store.data",
        {"map.relayoutData": None,
         "map.selectedData": {"range": {"x": [10.2, 20.8], "y": [4.1, 9.7]}},
         "map-reset-store.data": 0,
         "apply-roi.n_clicks": 0},
        {"roi-store.data": [0, H, 0, W], "roi-y0.value": 0, "roi-y1.value": H,
         "roi-x0.value": 0, "roi-x1.value": W}, "map.selectedData")
    assert roi == [4, 10, 10, 21]
    frame = _http_callback(app, "map-frame.style", {"roi-store.data": roi}, {},
                           "roi-store.data")
    assert frame == {"width": "100%", "aspectRatio": 11 / 6}
    reset = _http_callback(app, "roi-store.data",
        {"map.relayoutData": {"xaxis.autorange": True, "yaxis.autorange": True},
         "map.selectedData": {"range": {"x": [10.2, 20.8], "y": [4.1, 9.7]}},
         "apply-roi.n_clicks": 0},
        {"roi-store.data": roi, "roi-y0.value": 4, "roi-y1.value": 10,
         "roi-x0.value": 10, "roi-x1.value": 21}, "map.relayoutData")
    assert reset == [0, H, 0, W]

    reset = _http_callback(app, "roi-store.data",
        {"map.relayoutData": None, "map.selectedData": None,
         "map-reset-store.data": 1, "apply-roi.n_clicks": 0},
        {"roi-store.data": roi, "roi-y0.value": 4, "roi-y1.value": 10,
         "roi-x0.value": 10, "roi-x1.value": 21}, "map-reset-store.data")
    assert reset == [0, H, 0, W]


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


def test_figure_callbacks_reuse_last_reduction(cube, monkeypatch):
    # Audit item 15: palette/log/channel-axis changes are presentation
    # only; the channel sums run once per (revision, selection). Count
    # the kernel calls instead of timing anything.
    import xrf_visualizer.ui.app as app_mod
    real_map, real_spec = app_mod.map_from_channels, app_mod.spectrum_from_roi
    calls = []
    def counting_map(*a, **k):
        calls.append("map")
        return real_map(*a, **k)
    def counting_spec(*a, **k):
        calls.append("spec")
        return real_spec(*a, **k)
    monkeypatch.setattr(app_mod, "map_from_channels", counting_map)
    monkeypatch.setattr(app_mod, "spectrum_from_roi", counting_spec)
    app = build_app(cube)
    map_in = {"ch-store.data": [0, C], "roi-store.data": [0, H, 0, W],
              "points-store.data": [], "colorscale.value": "Greys_r",
              "log.value": [], "scale-store.data": None, "cal-store.data": None,
              "xmode-store.data": None, "gen.data": 0}
    spec_in = {"roi-store.data": [0, H, 0, W], "points-store.data": [],
               "ch-store.data": [0, C], "log.value": [], "cal-store.data": None,
               "xmode-store.data": None, "gen.data": 0}
    _http_callback(app, "map.figure", map_in, {}, "gen.data")
    _http_callback(app, "spectrum.figure", spec_in, {}, "gen.data")
    assert calls == ["map", "spec"]
    # Same selection, new palette / log axis: no new reduction.
    _http_callback(app, "map.figure", {**map_in, "colorscale.value": "Viridis"},
                   {}, "colorscale.value")
    _http_callback(app, "spectrum.figure", {**spec_in, "log.value": ["log"]},
                   {}, "log.value")
    assert calls == ["map", "spec"]
    # A new channel range is a new selection: reduced again.
    _http_callback(app, "map.figure", {**map_in, "ch-store.data": [10, 20]},
                   {}, "ch-store.data")
    assert calls == ["map", "spec", "map"]


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
    log_fig = _http_callback(app, "map.figure",
        {**base, "log.value": ["log"], "scale-store.data": {"zmin": 1, "zmax": 2000}},
        {}, "log.value")
    assert log_fig["layout"]["coloraxis"]["cmin"] == pytest.approx(np.log1p(1))
    assert log_fig["layout"]["coloraxis"]["cmax"] == pytest.approx(np.log1p(2000))

    reset, zmin, zmax = _http_callback(app, scale_key,
        {"apply-scale.n_clicks": 1, "reset-scale.n_clicks": 1},
        {"zmin.value": 0, "zmax.value": 10}, "reset-scale.n_clicks")
    assert reset == {"zmin": None, "zmax": None} and (zmin, zmax) == (None, None)
    reset_fig = _http_callback(app, "map.figure", {**base, "scale-store.data": reset},
                               {}, "scale-store.data")
    assert "cmin" not in reset_fig["layout"]["coloraxis"]
    assert "cmax" not in reset_fig["layout"]["coloraxis"]

    cal_key = next(k for k, spec in app.callback_map.items()
                   if any(item["id"] == "cal-input-store" for item in spec["inputs"]))
    calibration, message, apply_line_disabled = _http_callback(app, cal_key,
        {"cal-input-store.data": {"m": "0.5", "q": "1"}}, {},
        "cal-input-store.data")
    assert message == "Applied: E = 0.5 × channel + 1.0 keV"
    assert apply_line_disabled is False
    calibrated = _http_callback(app, "map.figure", {**base, "cal-store.data": calibration},
                                 {}, "cal-store.data")
    assert calibrated["layout"]["title"]["text"] == "Integrated over 1–257 keV"
    spectrum = _http_callback(app, "spectrum.figure",
        {"roi-store.data": [0, H, 0, W], "points-store.data": [], "ch-store.data": [0, C],
         "log.value": ["log"], "cal-store.data": calibration, "gen.data": 0},
        {}, "cal-store.data")
    assert spectrum["layout"]["xaxis"]["title"]["text"] == "E (keV; m=0.5, q=1.0)"
    assert spectrum["layout"]["xaxis"]["range"] == [0.75, 256.75]
    assert spectrum["layout"]["yaxis"]["title"]["text"] == "log1p(sum counts)"


def test_xrf_line_select_applies_calibrated_energy_width(cube):
    app = build_app(cube)
    line_key = next(k for k, spec in app.callback_map.items() if "line-msg.children" in k)
    out = _http_callback(app, line_key, {"apply-line.n_clicks": 1},
        {"line-select.value": 1.486, "line-width.value": 0.2,
         "cal-store.data": {"m": 0.01, "q": 0}},
        "apply-line.n_clicks")
    assert out == ([139, 159], "Integrating 1.486 ± 0.1 keV")
    selected_key = next(k for k, spec in app.callback_map.items()
                        if "line-selected.children" in k)
    assert _http_callback(app, selected_key, {"line-select.value": 1.486}, {},
                          "line-select.value") == "Selected: Al (Ka) — 1.486 keV"


def test_elemental_gallery_full_field_scale_and_zip(cube):
    app = build_app(cube)
    # Use two catalog entries whose calibrated windows fit the synthetic cube.
    from xrf_visualizer.ui.app import _XRF_LINES
    rows = [{"line_id": i, "width": 0.2} for i, line in enumerate(_XRF_LINES)
            if line["element"].startswith(("Al (Ka)", "Si (Ka)"))][:2]
    assert len(rows) == 2
    widths = [row["width"] for row in rows]
    calibration = {"m": 0.01, "q": 0.0}
    compute = next(spec["callback"].__wrapped__ for spec in app.callback_map.values()
                   if any(item["id"] == "elemental-compute" for item in spec["inputs"]))
    metadata, message, index = compute(1, rows, widths, calibration)
    assert len(metadata) == 2 and index == 0 and "2 full-field" in message

    figure = next(spec["callback"].__wrapped__ for spec in app.callback_map.values()
                  if any(item.component_id == "elemental-map"
                         for item in (spec["output"] if isinstance(spec["output"], list)
                                      else [spec["output"]])))
    shared = figure(metadata, 0, ["shared"], "Hot", [])
    assert shared.layout.title.text.startswith("Al (Ka)")
    assert "range" in shared.layout.title.text and "scale" in shared.layout.title.text
    assert shared.layout.coloraxis.cmin == pytest.approx(min(m["minimum"] for m in metadata))
    assert shared.layout.coloraxis.cmax == pytest.approx(max(m["maximum"] for m in metadata))
    np.testing.assert_array_equal(_plotly_array(shared.data[0].z),
                                  map_from_channels(cube, 139, 159))
    assert shared.layout.xaxis.scaleanchor == "y"
    auto = figure(metadata, 0, [], "Hot", ["log"])
    assert auto.layout.coloraxis.cmin == pytest.approx(np.log1p(metadata[0]["minimum"]))
    assert auto.layout.coloraxis.cmax == pytest.approx(np.log1p(metadata[0]["maximum"]))

    spectrum_figure_callback = next(spec["callback"].__wrapped__
                                    for spec in app.callback_map.values()
                                    if any(item.component_id == "elemental-spectrum"
                                           for item in (spec["output"] if isinstance(spec["output"], list)
                                                        else [spec["output"]])))
    spectrum_fig = spectrum_figure_callback(metadata, 0, [])
    assert spectrum_fig.layout.title.text.startswith("Full-field spectrum — Al (Ka)")
    assert spectrum_fig.layout.xaxis.title.text == "Energy (keV)"
    assert spectrum_fig.layout.shapes[0].type == "rect"

    download = next(spec["callback"].__wrapped__ for spec in app.callback_map.values()
                    if any(item["id"] == "elemental-download" for item in spec["inputs"]))
    data, status = download(1, metadata, ["shared"], "Hot", [])
    assert status == "Downloaded 2 maps and spectra"
    with ZipFile(BytesIO(base64.b64decode(data["content"]))) as archive:
        assert len(archive.namelist()) == 5
        map_files = [name for name in archive.namelist() if name.endswith(".png")]
        spectrum_files = [name for name in archive.namelist() if name.endswith("_spectra.svg")]
        assert len(map_files) == len(spectrum_files) == 2
        for filename in map_files:
            png = archive.read(filename)
            width, height = struct.unpack(">II", png[16:24])
            assert (width, height) == (W, H)
        expected_png = _map_png_bytes(map_from_channels(cube, 139, 159), "Hot",
                                      min(m["minimum"] for m in metadata),
                                      max(m["maximum"] for m in metadata))
        assert expected_png in [archive.read(name) for name in map_files]
        assert all(b"Energy (keV)" in archive.read(name) for name in spectrum_files)
        manifest = json.loads(archive.read("metadata.json"))
        assert len(manifest) == 2
        assert manifest[0]["file"] in map_files
        assert manifest[0]["spectrum_file"] in spectrum_files
        assert manifest[0]["min_count"] == pytest.approx(min(m["minimum"] for m in metadata))
        assert manifest[0]["line_centre_keV"] == pytest.approx(1.486)
        assert manifest[0]["integration_zone_keV"] == pytest.approx([1.386, 1.586])


def test_change_file_clears_elemental_results(cube):
    # F11: results belong to a dataset revision; Change file discards them,
    # so no stale map/spectrum survives until a new Compute.
    app = build_app(cube)
    from xrf_visualizer.ui.app import _XRF_LINES as lines
    rows = [{"line_id": i, "width": 0.2} for i, line in enumerate(lines)
            if line["element"].startswith("Al (Ka)")][:1]
    calibration = {"m": 0.01, "q": 0.0}
    compute = next(spec["callback"].__wrapped__ for spec in app.callback_map.values()
                   if any(item["id"] == "elemental-compute" for item in spec["inputs"]))
    metadata, _message, _index = compute(1, rows, [0.2], calibration)
    assert len(metadata) == 1
    figure = _cb(app, "elemental-map.figure")
    fig = figure(metadata, 0, [], "Hot", [])
    assert fig.data[0].z is not None
    download = next(spec["callback"].__wrapped__ for spec in app.callback_map.values()
                    if any(item["id"] == "elemental-download" for item in spec["inputs"]))
    _data, status = download(1, metadata, [], "Hot", [])
    assert status.startswith("Downloaded")
    out = _load_via_http(app, "change-file.n_clicks", {"change-file.n_clicks": 1}, {"gen.data": 1})
    assert "cleared" in out[0]
    empty_fig = figure(metadata, 0, [], "Hot", [])
    assert empty_fig.layout.annotations[0]["text"] == ("Apply calibration, add lines, then Compute")
    _data, status = download(1, metadata, [], "Hot", [])
    assert status == "Compute maps first"


def test_elemental_map_requires_calibration_and_width_is_float():
    row = {"line_id": 3, "width": 0.15}
    assert _add_elemental_line([row], 3) == [row]
    assert _line_channel_range(1.5, "0,075", {"m": 0.01, "q": 0}, 512) == (147, 154)
    with pytest.raises(ValueError, match="positive"):
        _line_channel_range(1.5, 0, {"m": 0.01, "q": 0}, 512)
    app = build_app(np.zeros((2, 3, 4), dtype=np.float32))
    buttons = _cb(app, "elemental-add.disabled")
    assert buttons(None, 0, [], [])[:2] == (True, True)
    assert buttons({"m": 1, "q": 0}, 0, [{"line_id": 1}], [])[:2] == (False, False)


def test_calibration_inputs_accept_full_float_precision(cube):
    app = build_app(cube)
    assert _component(app.layout, "m").to_plotly_json()["namespace"] == "dash_core_components"
    assert _component(app.layout, "q").to_plotly_json()["namespace"] == "dash_core_components"
    assert _component(app.layout, "m").type == "text"
    assert _component(app.layout, "q").type == "text"
    cal_key = next(k for k, spec in app.callback_map.items()
                   if any(item["id"] == "cal-input-store" for item in spec["inputs"]))
    calibration, message, _ = _http_callback(app, cal_key,
        {"cal-input-store.data": {"m": "0,075", "q": "0,1199"}}, {},
        "cal-input-store.data")
    assert calibration == {"m": 0.075, "q": 0.1199}
    assert message == "Applied: E = 0.075 × channel + 0.1199 keV"
    fig = spectrum_figure(cube, m=calibration["m"], q=calibration["q"])
    assert fig.layout.xaxis.title.text == "E (keV; m=0.075, q=0.1199)"


def test_calibration_point_pair_and_ols_fit_are_applied(cube):
    assert _calibration_from_points([
        {"channel": 0, "energy": 1.1}, {"channel": 10, "energy": 3.1},
    ]) == pytest.approx((0.2, 1.1))
    m, q = _calibration_from_points([
        {"channel": 0, "energy": 1}, {"channel": 10, "energy": 3},
        {"channel": 20, "energy": 6},
    ])
    assert m == pytest.approx(0.25)
    assert q == pytest.approx(5 / 6)
    assert _parse_calibration_point("7", "0,123456789012345") == {
        "channel": 7, "energy": 0.123456789012345}
    with pytest.raises(ValueError, match="integer"):
        _parse_calibration_point("7.5", "1.2")
    with pytest.raises(ValueError, match="distinct"):
        _calibration_from_points([
            {"channel": 3, "energy": 1}, {"channel": 3, "energy": 2},
        ])

    app = build_app(cube)
    fit = next(spec["callback"].__wrapped__ for spec in app.callback_map.values()
               if any(item["id"] == "cal-point-fit" for item in spec["inputs"]))
    calibration, status, point_status, line_disabled, m_text, q_text = fit(1, [
        {"channel": 0, "energy": 1}, {"channel": 10, "energy": 3},
        {"channel": 20, "energy": 6},
    ])
    assert calibration == pytest.approx({"m": 0.25, "q": 5 / 6})
    assert "Applied fit from 3 points" in status and point_status == status
    assert line_disabled is False
    assert float(m_text) == calibration["m"] and float(q_text) == calibration["q"]


def test_download_current_map_is_cropped_pixel_exact_png(cube):
    app = build_app(cube)
    data = _http_callback(app, "map-download.data", {"download-map.n_clicks": 1},
        {"ch-store.data": [100, 110], "roi-store.data": [10, 12, 20, 23],
         "colorscale.value": "Hot", "log.value": [],
         "scale-store.data": {"zmin": None, "zmax": None}}, "download-map.n_clicks")
    png = base64.b64decode(data["content"])
    assert data["filename"] == "xrf_map.png"
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    width, height, depth, color_type = struct.unpack(">IIBB", png[16:26])
    assert (width, height, depth, color_type) == (3, 2, 8, 6)
    pos, compressed = 8, b""
    while pos < len(png):
        size = struct.unpack(">I", png[pos:pos + 4])[0]
        kind = png[pos + 4:pos + 8]
        chunk = png[pos + 8:pos + 8 + size]
        if kind == b"IDAT":
            compressed += chunk
        pos += 12 + size
    raw = zlib.decompress(compressed)
    assert len(raw) == height * (1 + width * 4)
    assert all(raw[row * (1 + width * 4)] == 0 for row in range(height))
    # F7: the limits come from the full map, before the ROI crop
    full = map_from_channels(cube, 100, 110)
    current = full[10:12, 20:23]
    value = (current[0, 0] - full.min()) / (full.max() - full.min())
    expected = np.rint(np.asarray(sample_colorscale(
        "Hot", [value], colortype="tuple")[0]) * 255).astype(np.uint8)
    assert np.array_equal(np.frombuffer(raw[1:5], dtype=np.uint8), [*expected, 255])

    log_data = _http_callback(app, "map-download.data", {"download-map.n_clicks": 1},
        {"ch-store.data": [100, 110], "roi-store.data": [10, 12, 20, 23],
         "colorscale.value": "Hot", "log.value": ["log"],
         "scale-store.data": {"zmin": 1, "zmax": 2000}}, "download-map.n_clicks")
    png = base64.b64decode(log_data["content"])
    pos, compressed = 8, b""
    while pos < len(png):
        size = struct.unpack(">I", png[pos:pos + 4])[0]
        kind = png[pos + 4:pos + 8]
        if kind == b"IDAT":
            compressed += png[pos + 8:pos + 8 + size]
        pos += 12 + size
    raw = zlib.decompress(compressed)
    logged = np.log1p(current[0, 0])
    value = np.clip((logged - np.log1p(1)) / (np.log1p(2000) - np.log1p(1)), 0, 1)
    expected = np.rint(np.asarray(sample_colorscale(
        "Hot", [value], colortype="tuple")[0]) * 255).astype(np.uint8)
    assert np.array_equal(np.frombuffer(raw[1:5], dtype=np.uint8), [*expected, 255])


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


def test_calibrated_spectrum_zoom_maps_energy_back_to_channels(cube):
    app = build_app(cube)
    m, q = 0.0023, 0.0036
    channels = _http_callback(app, "ch-store.data",
        {"spectrum.relayoutData": {"xaxis.range[0]": m * 100 + q,
                                    "xaxis.range[1]": m * 120 + q},
         "apply-ch.n_clicks": 0},
        {"ch-store.data": [0, C], "ch-s.value": 0, "ch-e.value": C,
         "cal-store.data": {"m": m, "q": q}}, "spectrum.relayoutData")
    assert channels == [100, 120]


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
                     "apply-line", "line-selected", "scale-store", "cal-store",
                     "download-map", "map-frame", "map-reset-store", "map-bind-store",
                     "cal-input-store",
                     "cube-name", "gen"):
        assert expected in ids
    # figures render the empty state without a cube (raw figure, dash 4.5 single-Output)
    f = _cb(app, "map.figure")([0, 0], [0, 0, 0, 0], [], "Viridis", [],
                                {"zmin": None, "zmax": None}, None, None, 0)
    assert not f.data
    f = _cb(app, "spectrum.figure")([0, 0, 0, 0], [], [0, 0], [], None, None, 0)
    assert not f.data


def test_build_app_rejects_empty_axes():
    # F13: validation at the UI entry point, not only in the readers
    with pytest.raises(ValueError, match="positive"):
        build_app(np.zeros((0, 5, 6), dtype=np.float32))


def _load_via_http(app, trigger, inputs=None, states=None):
    # ctx.triggered_id only exists inside a request, so _on_load is driven
    # through the dash endpoint, not the raw callable.
    key = next(k for k in app.callback_map if "load-msg" in k)
    return _http_callback(app, key, inputs or {}, states or {}, trigger)


def test_on_load_upload(empty_app, cube):
    payload = "data:application/octet-stream;base64," + base64.b64encode(NPZ.read_bytes()).decode()
    out = _load_via_http(empty_app, "upload.contents",
                         inputs={"upload.contents": payload},
                         states={"upload.filename": "synthetic_cube.npz"})
    assert "Loaded synthetic_cube.npz" in out[0]
    assert out[2] == 1  # gen bumped
    # F2: a load commits full-range ROI/channels, empty points and auto scale
    assert out[8] == [0, H, 0, W] and out[9] == [0, C]
    assert out[10] == [] and out[11] == {"zmin": None, "zmax": None}
    # map figure now matches the processing layer for the full channel range
    m = _http_callback(empty_app, "map.figure",
        {"ch-store.data": [0, C], "roi-store.data": [0, H, 0, W], "points-store.data": [],
         "apply-scale.n_clicks": 0, "apply-calibration.n_clicks": 0, "gen.data": 1},
        {"m.value": None, "q.value": None, "zmin.value": None, "zmax.value": None,
         "colorscale.value": "Viridis", "log.value": []}, "gen.data")
    from xrf_visualizer.processing import map_from_channels
    np.testing.assert_array_equal(_plotly_array(m["data"][0]["z"]),
                                  map_from_channels(cube, 0, C))


def test_on_load_upload_bad_bytes(empty_app, cube):
    payload = "data:application/octet-stream;base64," + base64.b64encode(b"not a cube").decode()
    out = _load_via_http(empty_app, "upload.contents",
                         inputs={"upload.contents": payload},
                         states={"upload.filename": "junk.npz"})
    assert out[0].startswith("Upload failed")
    assert out[2] is None  # gen untouched
    # state untouched: figures still empty
    m = _http_callback(empty_app, "map.figure",
        {"ch-store.data": [0, C], "roi-store.data": [0, H, 0, W], "points-store.data": [],
         "apply-scale.n_clicks": 0, "apply-calibration.n_clicks": 0, "gen.data": 0},
        {"m.value": None, "q.value": None, "zmin.value": None, "zmax.value": None,
         "colorscale.value": "Viridis", "log.value": []}, "gen.data")
    assert not m["data"]


def test_on_load_change(empty_app):
    cleared = _load_via_http(empty_app, "change-file.n_clicks",
                             inputs={"change-file.n_clicks": 1},
                             states={"gen.data": 3})
    assert "Cube cleared" in cleared[0]
    assert cleared[2] == 4
    # F2: clear commits empty stores and controls
    assert cleared[8] == [0, 0, 0, 0] and cleared[9] == [0, 0]
    assert cleared[10] == [] and cleared[11] == {"zmin": None, "zmax": None}
    assert cleared[18] == [] and cleared[22] == "" and cleared[23] == ""


def test_on_load_dispatches_on_triggered_input_not_counters(empty_app, http_server):
    # F1: the n_clicks counters are cumulative; the old code branched on them,
    # so after one Change-file or URL click every later upload was discarded
    # or overwritten. Dispatch must follow ctx.triggered_id.
    payload = "data:application/octet-stream;base64," + base64.b64encode(NPZ.read_bytes()).decode()
    out = _load_via_http(empty_app, "upload.contents",
                         inputs={"upload.contents": payload},
                         states={"upload.filename": "synthetic_cube.npz"})
    assert "Loaded synthetic_cube.npz" in out[0]

    cleared = _load_via_http(empty_app, "change-file.n_clicks",
                             inputs={"change-file.n_clicks": 1},
                             states={"gen.data": 1})
    assert "Cube cleared" in cleared[0]
    assert cleared[2] == 2

    # the next upload still loads even though change-file.n_clicks is 1
    reupload = _load_via_http(empty_app, "upload.contents",
                              inputs={"upload.contents": payload, "change-file.n_clicks": 1},
                              states={"upload.filename": "synthetic_cube.npz", "gen.data": 2})
    assert "Loaded synthetic_cube.npz" in reupload[0]
    assert reupload[2] == 3

    small = np.zeros((8, 16, 100), dtype=np.float32)
    buf = BytesIO()
    np.savez(buf, img=small)
    small_payload = "data:application/octet-stream;base64," + base64.b64encode(buf.getvalue()).decode()
    # a stale URL click + URL must not override a fresh upload
    fresh = _load_via_http(empty_app, "upload.contents",
                           inputs={"upload.contents": small_payload, "load-url.n_clicks": 1,
                                   "change-file.n_clicks": 1},
                           states={"upload.filename": "small.npz", "gen.data": 3,
                                   "url.value": f"{http_server}/synthetic_cube.npz"})
    assert "Loaded small.npz" in fresh[0]


def test_on_load_replaces_stale_selections(cube):
    # F2: on load the new dataset owns all selections; a stale ROI, channels,
    # points, scale and calibration points must not survive.
    app = build_app(cube)
    _http_callback(app, "roi-store.data",
        {"map.relayoutData": {"xaxis.range[0]": 10.2, "xaxis.range[1]": 20.8,
                              "yaxis.range[0]": 4.1, "yaxis.range[1]": 9.7},
         "apply-roi.n_clicks": 0},
        {"roi-store.data": [0, H, 0, W], "roi-y0.value": 0, "roi-y1.value": H,
         "roi-x0.value": 0, "roi-x1.value": W}, "map.relayoutData")

    small = np.zeros((8, 16, 100), dtype=np.float32)
    buf = BytesIO()
    np.savez(buf, img=small)
    payload = "data:application/octet-stream;base64," + base64.b64encode(buf.getvalue()).decode()
    out = _load_via_http(app, "upload.contents",
                         inputs={"upload.contents": payload},
                         states={"upload.filename": "small.npz", "gen.data": 0})
    assert out[0] == "Loaded small.npz — 8 x 16 x 100"
    assert out[8] == [0, 8, 0, 16]  # full new range, not the stale [4, 10, 10, 21]
    assert out[9] == [0, 100]
    assert out[10] == [] and out[11] == {"zmin": None, "zmax": None}
    assert out[18] == [] and out[20] is None and out[21] is None
    assert out[22] == "" and out[23] == ""
    # state is committed: the map figure now renders the new cube
    fig = _http_callback(app, "map.figure",
        {"ch-store.data": [0, 100], "roi-store.data": [0, 8, 0, 16],
         "points-store.data": [], "apply-scale.n_clicks": 0,
         "apply-calibration.n_clicks": 0, "gen.data": 1},
        {"m.value": None, "q.value": None, "zmin.value": None, "zmax.value": None,
         "colorscale.value": "Viridis", "log.value": []}, "gen.data")
    assert _plotly_array(fig["data"][0]["z"]).shape == (8, 16)


@pytest.fixture
def http_server(monkeypatch):
    # F10 gate off for loopback test servers (gate covered in test_readers).
    monkeypatch.setattr("xrf_visualizer.data._assert_public_url",
                        lambda url: None)
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(FIX))
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def test_on_load_link(empty_app, cube, http_server):
    out = _load_via_http(empty_app, "load-url.n_clicks",
                         inputs={"load-url.n_clicks": 1},
                         states={"url.value": f"{http_server}/synthetic_cube.npz"})
    assert "Loaded synthetic_cube.npz" in out[0]
    out2 = _load_via_http(empty_app, "load-url.n_clicks",
                          inputs={"load-url.n_clicks": 1},
                          states={"url.value": "ftp://nope"})
    assert out2[0].startswith("Download failed")


# ------------------------------------------------------- step 2: audit F3-F7

def test_clamp_roi_preserves_explicit_zero():
    # F3: an explicit 0 is a value, not "unset"
    assert _clamp_roi(H, W, 0, 0, 0, 0) == [0, 0, 0, 0]
    assert _clamp_roi(H, W, 0, 10, 0, 0) == [0, 10, 0, 0]
    assert _clamp_roi(H, W, None, 0, 0, W) == [0, 0, 0, W]


def test_empty_selections_render_zero_maps(cube):
    # F3: empty channel interval / empty ROI produce zeros, not the full range
    fig = map_figure(cube, 5, 5)
    np.testing.assert_array_equal(np.asarray(fig.data[0]["z"]),
                                  np.zeros((H, W), dtype=np.float64))
    spec = spectrum_figure(cube, roi=[10, 10, 20, 20])
    np.testing.assert_array_equal(np.asarray(spec.data[0]["y"]),
                                  np.zeros(C, dtype=np.float64))


def test_line_channel_range_edge_tolerance():
    # F5: P (Ka) 2.01 +/- 0.05 keV with m=0.01, q=0 -> (196, 207)
    cal = {"m": 0.01, "q": 0.0}
    assert _line_channel_range(2.01, 0.1, cal, 1000) == (196, 207)
    # a grid point exactly on the window edge is included (F4 grid branch)
    grid = np.arange(1000, dtype=np.float64) * 0.01
    assert _line_channel_range(2.01, 0.1, cal, 1000, energy=grid) == (196, 207)


def test_ch_from_relayout_half_pixel_round_trip():
    # F6: the drawn edge of channel i is i-0.5; the inverse must recover i
    assert _ch_from_relayout({"xrange": [9.5, 19.5]}, C) == [10, 20]
    assert _ch_from_relayout({"xrange": [0.095, 0.195]}, C, 0.01, 0.0) == [10, 20]
    # non-uniform file grid (F4): invert with searchsorted
    grid = np.linspace(0.0, 10.0, 11)  # 1.0 keV per channel
    assert _ch_from_relayout({"xrange": [2.5, 6.5]}, 11, energy=grid) == [3, 7]


def test_roi_round_trip_half_pixel():
    # F6: ROI [10,12,20,23] is drawn on x=[19.5,22.5], y=[11.5,9.5]
    assert _roi_from_relayout({"xrange": [19.5, 22.5], "yrange": [11.5, 9.5]},
                              H, W) == [10, 12, 20, 23]


def _png_payload(png):
    pos, compressed = 8, b""
    while pos < len(png):
        size = struct.unpack(">I", png[pos:pos + 4])[0]
        kind = png[pos + 4:pos + 8]
        if kind == b"IDAT":
            compressed += png[pos + 8:pos + 8 + size]
        pos += 12 + size
    return zlib.decompress(compressed)


def test_map_png_constant_matches_figure_range():
    # F7: a constant map uses [v, v+1] like the figure -> the t=0 color
    img = np.full((2, 2), 5.0)
    png = _map_png_bytes(img)
    first = np.frombuffer(_png_payload(png)[1:5], dtype=np.uint8)
    expected = np.rint(np.asarray(sample_colorscale("Greys_r", [0.0],
                                                   colortype="tuple")[0])
                       * 255).astype(np.uint8)
    assert np.array_equal(first, [*expected, 255])


def test_spectrum_figure_file_energy_axis(cube):
    # F4: the file energy grid is the axis when no manual calibration is set
    grid = np.linspace(0.1, 10.1, C)
    fig = spectrum_figure(cube, roi=[0, H, 0, W], grid=grid)
    np.testing.assert_allclose(np.asarray(fig.data[0]["x"]), grid, rtol=0, atol=0)
    assert fig.layout.xaxis.title.text == "E (keV; file axis)"
    adc = spectrum_figure(cube, roi=[0, H, 0, W], grid=grid, show_energy=False)
    np.testing.assert_array_equal(np.asarray(adc.data[0]["x"]), np.arange(C))
    assert adc.layout.xaxis.title.text == "channel (ADC)"


def test_map_figure_file_energy_axis_title(cube):
    grid = np.linspace(0.1, 10.1, C)
    fig = map_figure(cube, 10, 20, grid=grid)
    assert f"{grid[10]:.12g}–{grid[19]:.12g} keV" in fig.layout.title.text
    adc = map_figure(cube, 10, 20, grid=grid, show_energy=False)
    assert "10–20 ADC channels" in adc.layout.title.text


def test_xmode_toggle_follows_calibration(cube):
    app = build_app(cube)
    sw = _component(app.layout, "show-energy")
    assert sw.value == []
    toggle = _cb(app, "show-energy.value")
    assert toggle(None, 0) == []
    assert toggle({"m": 0.01, "q": 0.0}, 0) == ["energy"]
    assert _cb(app, "xmode-store.data")(["energy"]) is True
    assert _cb(app, "xmode-store.data")([]) is False


def test_load_shows_file_calibration_in_m_q_fields():
    # F4: a file energy axis is displayed in the m/q fields on load
    arr = np.zeros((2, 3, 4), dtype=np.float64)
    grid = np.array([0.5, 1.5, 2.5, 3.5])
    buf = BytesIO()
    np.savez(buf, img=arr, energy=grid)
    app = build_app(None)
    key = next(k for k, spec in app.callback_map.items()
               if "load-msg.children" in k)
    out = _http_callback(app, key,
                         {"upload.contents":
                          "data:;base64," + base64.b64encode(buf.getvalue()).decode(),
                          "load-url.n_clicks": 0, "change-file.n_clicks": 0},
                         {"upload.filename": "cube.npz", "url.value": "",
                          "gen.data": 0},
                         "upload.contents")
    assert "Loaded cube.npz — 2 x 3 x 4" in out[0]
    assert out[5] is None  # cal-store stays unset until the user applies it
    assert float(out[-2]) == pytest.approx(1.0, abs=1e-9)
    assert float(out[-1]) == pytest.approx(0.5, abs=1e-9)
    assert "File energy axis" in out[23]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
