"""Dash UI (M4): 2-D map on the left, spectrum on the right.

Zoom is selection (legacy semantics): zooming the map recomputes the
spectrum for the zoomed region; zooming the spectrum recomputes the map for
the zoomed channel interval. autorange/reset = full field. Explicit
selection: numeric ROI panel, multi-point list, numeric channel interval.
All selections are clamped before reaching the processing layer, which
raises ValueError on out-of-range input.

M6 file loading: build_app(cube=None) starts empty. The user loads a cube
via upload (.npz/.h5/.npy) or an http(s) link in the UI; "Change file"
clears it. The current cube lives in a Python-side closure (``state``),
never in a dash Store (too big for JSON round-trips). Selection stores may
be stale after a new load; the processing layer clamps, and a ``gen`` store
bump forces the figure callbacks to recompute.

Dash 4.5 return convention (source-verified in dash/_callback.py,
``_prepare_response``): a callback registered with a single Output receives
its return value AS-IS; only multi-output callbacks take a tuple of values.
Returning ``[value]`` from a single-output callback double-wraps it (the
store got ``[[...]]``, the graph got ``[fig]`` -> 500s after zoom).

M7 look: ported from the legacy dash-xrf-analyzer app (user request) —
dash-bootstrap-components DARKLY theme, #4196B4 navbar, fluid container,
5/7 columns, outline info buttons.
"""
import base64
import re
from pathlib import Path

import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update

from ..data import download_cube, load_cube_bytes
from ..processing import (
    calibrate,
    map_from_channels,
    spectrum_from_points,
    spectrum_from_roi,
)

_COLORSCALES = ("Viridis", "Plasma", "Hot", "Greys", "Jet")
_ACCENT = "#4196B4"  # legacy infn_lightblue


class _XrfDash(Dash):
    def run(self, *args, **kwargs):
        if kwargs.get("jupyter_mode") is True:
            kwargs["jupyter_mode"] = "inline"
        return super().run(*args, **kwargs)


# ---------------------------------------------------------------- pure logic

def _roi_from_relayout(rl, H, W):
    """Zoom -> ROI [y0, y1, x0, x1] (y=row, x=col) from plotly relayoutData.

    autorange/reset -> full field; relayout without ranges (pan) -> None.
    """
    if not rl:
        return None
    if any(k in ("xautorange", "yautorange", "xaxis.autorange", "yaxis.autorange")
           for k in rl):
        return [0, H, 0, W]
    if "xrange" not in rl and "yrange" not in rl:
        return None
    xa, xb = sorted(rl.get("xrange", (0, W)))
    ya, yb = sorted(rl.get("yrange", (0, H)))
    x0 = max(0, min(int(np.floor(xa)), W))
    x1 = max(0, min(int(np.ceil(xb)), W))
    y0 = max(0, min(int(np.floor(ya)), H))
    y1 = max(0, min(int(np.ceil(yb)), H))
    return [y0, y1, x0, x1]


def _ch_from_relayout(rl, C):
    """Spectrum zoom -> channel interval [s, e] from relayoutData; None = ignore."""
    if not rl:
        return None
    if any(k in ("xautorange", "xaxis.autorange") for k in rl):
        return [0, C]
    if "xrange" not in rl:
        return None
    a, b = sorted(rl["xrange"])
    s = max(0, min(int(np.floor(a)), C))
    e = max(0, min(int(np.ceil(b)), C))
    return [s, e]


def _clamp_roi(H, W, y0, y1, x0, x1):
    """Clamp/sort a numeric ROI to [0, H] x [0, W]; None values fall to edges."""
    y0 = max(0, min(int(y0 or 0), H))
    y1 = max(0, min(int(y1 or H), H))
    x0 = max(0, min(int(x0 or 0), W))
    x1 = max(0, min(int(x1 or W), W))
    return [min(y0, y1), max(y0, y1), min(x0, x1), max(x0, x1)]


def _parse_points(rows_s, cols_s):
    """'10, 19' + '30, 39' -> [(10, 30), (19, 39)]; ValueError on bad input."""
    def _ints(s):
        return [int(x) for x in re.split(r"[,; ]+", str(s).strip()) if x]
    rows, cols = _ints(rows_s), _ints(cols_s)
    if len(rows) != len(cols):
        raise ValueError(f"rows ({len(rows)}) and cols ({len(cols)}) must match")
    return list(zip(rows, cols))


# ------------------------------------------------------------ figure builders

def map_figure(cube, ch_start, ch_end, zmin=None, zmax=None,
               colorscale="Viridis", log=False, m=None, q=None, roi=None):
    """Current map: plain float64 channel sum, display options applied."""
    image = map_from_channels(cube, ch_start, ch_end)
    if log:  # shortcut: log via data transform; zmin/zmax then apply to log1p values
        image = np.log1p(image)
    fig = px.imshow(image, color_continuous_scale=colorscale or "Viridis",
                    aspect="auto", labels={"x": "x (col)", "y": "y (row)", "color": "sum"})
    if zmin is not None:  # px.imshow emits a Heatmap trace: scale limits are zmin/zmax, not cmin/cmax
        fig.update_traces(zmin=float(zmin))
    if zmax is not None:
        fig.update_traces(zmax=float(zmax))
    interval = (f"{m * ch_start + (q or 0):g}–{m * ch_end + (q or 0):g} keV"
                if m is not None else f"{ch_start}–{ch_end} ADC channels")
    fig.update_layout(title=f"Integrated over {interval}",
                      margin=dict(l=40, r=10, t=55, b=30))
    if roi:
        y0, y1, x0, x1 = roi
        fig.update_layout(xaxis_range=[x0 - 0.5, x1 - 0.5],
                          yaxis_range=[y1 - 0.5, y0 - 0.5])
    return fig


def spectrum_figure(cube, roi=None, points=None, m=None, q=None, channel_range=None):
    """Current spectrum: plain float64 pixel sum (points win over ROI).

    With m != 0 the x-axis is the calibrated energy axis E = m*i + q (keV).
    """
    H, W, C = cube.shape
    if points:
        spec = spectrum_from_points(cube, [p[0] for p in points], [p[1] for p in points])
        selection = "Points: " + "; ".join(f"(row {y}, col {x})" for y, x in points)
    else:
        y0, y1, x0, x1 = roi if roi else [0, H, 0, W]
        spec = spectrum_from_roi(cube, y0, y1, x0, x1)
        selection = f"Box: y {y0}–{y1}, x {x0}–{x1}"
    if m is not None:
        e = calibrate(C, m, q or 0.0)
        x, xlab = e, "E (keV)"
    else:
        x, xlab = np.arange(C), "channel (ADC)"
    # go.Scatter (not px.line): px.line wants a pandas frame for plain arrays
    fig = go.Figure(go.Scatter(x=x, y=spec, mode="lines"))
    fig.update_layout(title=selection, xaxis_title=xlab, yaxis_title="sum counts",
                      margin=dict(l=50, r=10, t=55, b=30))
    if channel_range:
        start, end = channel_range
        limits = [start - 0.5, end - 0.5]
        if m is not None:
            limits = [m * i + (q or 0.0) for i in limits]
        fig.update_xaxes(range=sorted(limits))
    return fig


# ----------------------------------------------------------------------- app

def _as_array(cube):
    """Accept None, an ndarray, or an XrfCube (uses its .array)."""
    if cube is None:
        return None
    if isinstance(cube, np.ndarray):
        return cube
    arr = getattr(cube, "array", None)
    if arr is None:
        raise TypeError(f"cube must be an XrfCube or ndarray, got {type(cube).__name__}")
    return arr


def _empty_figure(label):
    fig = go.Figure()
    fig.update_layout(
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        annotations=[dict(text=label, showarrow=False, font=dict(size=18))],
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


def build_app(cube=None):
    """Dash app for an XRF cube. Run with app.run(...) or
    app.run(jupyter_mode=True) inside a notebook.

    ``cube`` may be an XrfCube, a (H, W, C) ndarray, or None (M6: start
    empty, load via upload or link in the UI).
    """
    arr = _as_array(cube)
    state = {"cube": arr,
             "name": Path(getattr(cube, "path", "") or "cube").name}
    H, W, C = arr.shape if arr is not None else (0, 0, 0)
    app = _XrfDash(__name__, external_stylesheets=[dbc.themes.DARKLY])
    app.layout = dbc.Container(fluid=True, children=[
        dbc.Row([
            dbc.Col([
                html.H3("XRF visualizer", style={"display": "inline-block", "margin-right": 12}),
                html.Span(id="cube-name",
                          children=f"— {H} x {W} x {C}" if arr is not None else "— no cube loaded",
                          style={"verticalAlign": "middle"}),
            ]),
            dbc.Col([
                dbc.Button("Change file", id="change-file", n_clicks=0,
                           outline=True, color="secondary"),
            ], width="auto"),
        ]),
        dbc.Navbar(
            [dbc.NavbarBrand("single-point spectrum & map from a 3-D XRF cube")],
            color=_ACCENT,
        ),
        html.Br(),
        # --- M6 file loading (was the top inputs block) ---
        dbc.Row([
            dbc.Col(md=5, children=[
                dcc.Upload(
                    id="upload",
                    children=html.Div("Upload .npz / .h5 / .npy (drag & drop or click)",
                                      style={"display": "inline-block", "width": "100%",
                                             "lineHeight": "50px", "border": "1px dashed #999",
                                             "textAlign": "center"}),
                    multiple=False,
                ),
            ]),
            dbc.Col(md=4, children=[
                dbc.Input(id="url", type="text",
                          placeholder="https://… (share/download link: ownCloud, nextCloud, Zenodo, …)",
                          style={"width": "100%"}),
            ]),
            dbc.Col(md=3, children=[
                dbc.Button("Load from link", id="load-url", n_clicks=0,
                           outline=True, color="info"),
            ]),
        ]),
        html.Div(id="load-msg", style={"color": "orange"}),
        # --- body: 5/7 split like the legacy app ---
        dbc.Row([
            ### map column
            dbc.Col(md=5, children=[
                dbc.Col(html.H4("XRF Map", style={"color": _ACCENT}),
                        width={"size": 6, "offset": 3}),
                dcc.Graph(id="map", style={"height": "60vh"}),
                dbc.Row(className="g-2 align-items-center mb-2", children=[
                    dbc.Col([dbc.Label("zmin", html_for="zmin"), dbc.Input(id="zmin", type="number")], width=2),
                    dbc.Col([dbc.Label("zmax", html_for="zmax"), dbc.Input(id="zmax", type="number")], width=2),
                    dbc.Col([dbc.Label("colormap", html_for="colorscale"), dcc.Dropdown(id="colorscale",
                                 options=[{"label": n, "value": n} for n in _COLORSCALES],
                                 value="Viridis")], width=4),
                    dbc.Col(dcc.Checklist(id="log", options=[{"label": "log", "value": "log"}], value=[]), width=2, className="pt-4"),
                    dbc.Col(dbc.Button("Apply scale", id="apply-scale", n_clicks=0,
                                       outline=True, color="info"), width=2),
                ]),
                html.B("Interval → map"),
                dbc.Row(className="g-2 align-items-end mb-2", children=[
                    dbc.Col([dbc.Label("ch begin", html_for="ch-s"), dbc.Input(id="ch-s", type="number", value=0)], width=4),
                    dbc.Col([dbc.Label("ch end", html_for="ch-e"), dbc.Input(id="ch-e", type="number", value=C)], width=4),
                    dbc.Col(dbc.Button("Apply interval", id="apply-ch", n_clicks=0, outline=True, color="info"), width=4),
                ]),
            ]),
            ### spectrum column
            dbc.Col(md=7, children=[
                dbc.Col(html.H4("XRF Spectrum", style={"color": _ACCENT}),
                        width={"size": 6, "offset": 3}),
                dcc.Graph(id="spectrum", style={"height": "60vh"}),
                html.B("ROI (rows y, cols x) → spectrum"),
                dbc.Row(className="g-2 mb-2", children=[
                    dbc.Col([dbc.Label("x0", html_for="roi-x0"), dbc.Input(id="roi-x0", type="number", value=0)], width=6),
                    dbc.Col([dbc.Label("x1", html_for="roi-x1"), dbc.Input(id="roi-x1", type="number", value=W)], width=6),
                    dbc.Col([dbc.Label("y0", html_for="roi-y0"), dbc.Input(id="roi-y0", type="number", value=0)], width=6),
                    dbc.Col([dbc.Label("y1", html_for="roi-y1"), dbc.Input(id="roi-y1", type="number", value=H)], width=6),
                    dbc.Col(dbc.Button("Apply ROI", id="apply-roi", n_clicks=0, outline=True, color="info"), width=12),
                ]),
                html.B("Points → spectrum"),
                dbc.Row(className="g-2 align-items-end mb-2", children=[
                    dbc.Col([dbc.Label("rows", html_for="pt-rows"), dbc.Input(id="pt-rows", type="text", placeholder="10, 19")], width=4),
                    dbc.Col([dbc.Label("cols", html_for="pt-cols"), dbc.Input(id="pt-cols", type="text", placeholder="30, 39")], width=4),
                    dbc.Col(dbc.Button("Apply points", id="apply-points", n_clicks=0, outline=True, color="info"), width=4),
                    dbc.Col(dbc.Button("Clear selection", id="clear", n_clicks=0, outline=True, color="danger"), width=12),
                ]),
                html.B("Calibration (keV = m × channel + q)"),
                dbc.Row(className="g-2 align-items-end", children=[
                    dbc.Col([dbc.Label("m", html_for="m"), dbc.Input(id="m", type="number", step=0.001)], width=6),
                    dbc.Col([dbc.Label("q", html_for="q"), dbc.Input(id="q", type="number", step=0.001)], width=6),
                    dbc.Col(dbc.Button("Apply calibration", id="apply-calibration", n_clicks=0, outline=True, color="info"), width=12),
                ]),
            ]),
        ]),
        html.Div(id="msg", style={"color": "orange"}),
        dcc.Store(id="roi-store", data=[0, H, 0, W]),
        dcc.Store(id="ch-store", data=[0, C]),
        dcc.Store(id="points-store", data=[]),
        dcc.Store(id="gen", data=0),
    ])

    @app.callback(Output("map", "figure"),
                  Input("ch-store", "data"), Input("roi-store", "data"),
                  Input("zmin", "value"), Input("zmax", "value"),
                  Input("colorscale", "value"), Input("log", "value"),
                  Input("apply-scale", "n_clicks"), Input("apply-calibration", "n_clicks"),
                  Input("gen", "data"),
                  State("m", "value"), State("q", "value"))
    def _map_figure(ch, roi, zmin, zmax, cscale, log, _n_scale, _n_cal, _gen, m, q):
        c = state["cube"]
        if c is None:
            return _empty_figure("No cube loaded — upload a file or paste a link above")
        s, e = ch
        if e is None or e <= (s or 0):  # stale/empty interval -> full channels
            s, e = 0, c.shape[2]
        # dash 4.5: single Output -> return the raw value (no [..] wrap)
        return map_figure(c, s, e, zmin, zmax, cscale, bool(log), m, q, roi)

    @app.callback(Output("spectrum", "figure"),
                  Input("roi-store", "data"), Input("points-store", "data"),
                  Input("ch-store", "data"),
                  Input("apply-calibration", "n_clicks"), Input("gen", "data"),
                  State("m", "value"), State("q", "value"))
    def _spec_figure(roi, pts, channels, _n_cal, _gen, m, q):
        c = state["cube"]
        if c is None:
            return _empty_figure("No cube loaded")
        H, W, _ = c.shape
        if pts:  # drop points stale from a previous cube; all stale -> ROI
            keep = [p for p in pts if 0 <= p[0] < H and 0 <= p[1] < W]
            pts = keep or None
        if roi and roi[1] <= roi[0] and roi[3] <= roi[2]:  # degenerate ROI -> full frame
            roi = None
        return spectrum_figure(c, roi=roi, points=pts, m=m, q=q, channel_range=channels)

    @app.callback(Output("roi-store", "data"),
                  Input("map", "relayoutData"), Input("apply-roi", "n_clicks"),
                  State("roi-store", "data"),
                  State("roi-y0", "value"), State("roi-y1", "value"),
                  State("roi-x0", "value"), State("roi-x1", "value"),
                  prevent_initial_call=True)
    def _set_roi(rl, n, cur, y0, y1, x0, x1):
        c = state["cube"]
        if c is None:
            return cur
        H, W, _ = c.shape
        if ctx.triggered_id == "map":
            r = _roi_from_relayout(rl, H, W)
            if r is not None:
                return r
        elif ctx.triggered_id == "apply-roi":
            return _clamp_roi(H, W, y0, y1, x0, x1)
        return cur

    @app.callback(Output("ch-store", "data"),
                  Input("spectrum", "relayoutData"), Input("apply-ch", "n_clicks"),
                  State("ch-store", "data"),
                  State("ch-s", "value"), State("ch-e", "value"),
                  prevent_initial_call=True)
    def _set_ch(rl, n, cur, s, e):
        c = state["cube"]
        if c is None:
            return cur
        C = c.shape[2]
        if ctx.triggered_id == "spectrum":
            r = _ch_from_relayout(rl, C)
            if r is not None:
                return r
        elif ctx.triggered_id == "apply-ch":
            s = max(0, min(int(s or 0), C))
            e = max(0, min(int(e or C), C))
            return [min(s, e), max(s, e)]
        return cur

    @app.callback([Output("points-store", "data"), Output("msg", "children")],
                  Input("apply-points", "n_clicks"), Input("clear", "n_clicks"),
                  State("pt-rows", "value"), State("pt-cols", "value"),
                  prevent_initial_call=True)
    def _set_points(n_pt, n_cl, rows_s, cols_s):
        c = state["cube"]
        if n_cl:
            return [], ""
        if c is None:
            return no_update, "Load a cube first (upload or link)"
        H, W, _ = c.shape
        try:
            pts = [[r, c_] for r, c_ in _parse_points(rows_s, cols_s)]
            bad = [p for p in pts if not (0 <= p[0] < H and 0 <= p[1] < W)]
            if bad:
                raise ValueError(f"point {bad[0]} out of bounds ({H}x{W})")
            return pts, ""
        except ValueError as e:
            return no_update, str(e)

    # --- M6: file loading (upload / link / change) ---------------------------
    @app.callback([Output("load-msg", "children"), Output("cube-name", "children"),
                   Output("zmin", "value"), Output("zmax", "value"),
                   Output("gen", "data")],
                  Input("upload", "contents"), State("upload", "filename"),
                  Input("load-url", "n_clicks"), State("url", "value"),
                  Input("change-file", "n_clicks"),
                  State("gen", "data"),
                  prevent_initial_call=True)
    def _on_load(contents, filename, n_url, url, n_change, gen):
        """Single entry point for file loading (M6). Priority: change > link >
        upload. On error the cube state is untouched. The gen bump forces the
        figure callbacks to recompute (selection stores may be stale)."""
        if n_change:
            state["cube"], state["name"] = None, ""
            return ("Cube cleared — upload a file or paste a link.",
                    "— no cube loaded", None, None, (gen or 0) + 1)
        if n_url and url and url.strip():
            try:
                cube = download_cube(url.strip())
            except Exception as e:
                return f"Download failed: {e}", no_update, no_update, no_update, no_update
            state["cube"], state["name"] = cube.array, cube.path.name
            h, w, c_ = cube.array.shape
            return (f"Loaded {cube.path.name} — {h} x {w} x {c_}",
                    f"— {cube.path.name} ({h} x {w} x {c_})",
                    None, None, (gen or 0) + 1)
        if contents:
            try:
                _, b64 = contents.split(",", 1)
                data = base64.b64decode(b64)
                cube = load_cube_bytes(data, filename or "")
            except Exception as e:
                return f"Upload failed: {e}", no_update, no_update, no_update, no_update
            state["cube"], state["name"] = cube.array, Path(filename or "cube").name
            h, w, c_ = cube.array.shape
            return (f"Loaded {state['name']} — {h} x {w} x {c_}",
                    f"— {state['name']} ({h} x {w} x {c_})",
                    None, None, (gen or 0) + 1)
        return no_update, no_update, no_update, no_update, no_update

    return app
