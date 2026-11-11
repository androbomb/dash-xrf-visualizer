"""Dash UI (M4): 2-D map on the left, spectrum on the right.

Zoom is selection (legacy semantics): zooming the map recomputes the
spectrum for the zoomed region; zooming the spectrum recomputes the map for
the zoomed channel interval. autorange/reset = full field. Explicit
selection: numeric ROI panel, multi-point list, numeric channel interval.
All selections are clamped before reaching the processing layer, which
raises ValueError on out-of-range input.
"""
import re

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html, no_update

from ..processing import (
    calibrate,
    map_from_channels,
    spectrum_from_points,
    spectrum_from_roi,
)

_COLORSCALES = ("Viridis", "Plasma", "Hot", "Greys", "Jet")


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
               colorscale="Viridis", log=False):
    """Current map: plain float64 channel sum, display options applied."""
    m = map_from_channels(cube, ch_start, ch_end)
    if log:  # shortcut: log via data transform; zmin/zmax then apply to log1p values
        m = np.log1p(m)
    fig = px.imshow(m, color_continuous_scale=colorscale or "Viridis",
                    aspect="auto", labels={"x": "x (col)", "y": "y (row)", "color": "sum"})
    if zmin:  # px.imshow emits a Heatmap trace: scale limits are zmin/zmax, not cmin/cmax
        fig.update_traces(zmin=float(zmin))
    if zmax:
        fig.update_traces(zmax=float(zmax))
    fig.update_layout(margin=dict(l=40, r=10, t=30, b=30))
    return fig


def spectrum_figure(cube, roi=None, points=None, m=None, q=None):
    """Current spectrum: plain float64 pixel sum (points win over ROI).

    With m != 0 the x-axis is the calibrated energy axis E = m*i + q (keV).
    """
    H, W, C = cube.shape
    if points:
        spec = spectrum_from_points(cube, [p[0] for p in points], [p[1] for p in points])
    else:
        y0, y1, x0, x1 = roi if roi else [0, H, 0, W]
        spec = spectrum_from_roi(cube, y0, y1, x0, x1)
    if m:
        e = calibrate(C, m, q or 0.0)
        x, xlab = e, "E (keV)"
    else:
        x, xlab = np.arange(C), "channel (ADC)"
    # go.Scatter (not px.line): px.line wants a pandas frame for plain arrays
    fig = go.Figure(go.Scatter(x=x, y=spec, mode="lines"))
    fig.update_layout(xaxis_title=xlab, yaxis_title="sum counts",
                      margin=dict(l=50, r=10, t=30, b=30))
    return fig


# ----------------------------------------------------------------------- app

def build_app(cube):
    """Dash app around a loaded (H, W, C) cube. Run with app.run(...) or
    app.run(jupyter_mode=True) inside a notebook."""
    H, W, C = cube.shape
    app = Dash(__name__)
    app.layout = html.Div([
        html.H3(f"XRF visualizer — {H} x {W} x {C}"),
        html.Div([
            html.Div(dcc.Graph(id="map", style={"height": "60vh"}),
                     style={"width": "60%", "display": "inline-block"}),
            html.Div(dcc.Graph(id="spectrum", style={"height": "60vh"}),
                     style={"width": "38%", "display": "inline-block"}),
        ]),
        html.Div([
            html.B("ROI (rows y, cols x) -> spectrum: "),
            html.Label("y0"), dcc.Input(id="roi-y0", type="number", value=0, style={"width": 70}),
            html.Label("y1"), dcc.Input(id="roi-y1", type="number", value=H, style={"width": 70}),
            html.Label("x0"), dcc.Input(id="roi-x0", type="number", value=0, style={"width": 70}),
            html.Label("x1"), dcc.Input(id="roi-x1", type="number", value=W, style={"width": 70}),
            html.Button("Apply ROI", id="apply-roi", n_clicks=0),
            html.B("  Points -> spectrum: "),
            html.Label("rows"), dcc.Input(id="pt-rows", type="text", placeholder="10, 19", style={"width": 110}),
            html.Label("cols"), dcc.Input(id="pt-cols", type="text", placeholder="30, 39", style={"width": 110}),
            html.Button("Apply points", id="apply-points", n_clicks=0),
            html.Button("Clear selection", id="clear", n_clicks=0),
            html.B("  Interval -> map: "),
            html.Label("ch s"), dcc.Input(id="ch-s", type="number", value=0, style={"width": 70}),
            html.Label("ch e"), dcc.Input(id="ch-e", type="number", value=C, style={"width": 70}),
            html.Button("Apply interval", id="apply-ch", n_clicks=0),
            html.B("  Display: "),
            html.Label("zmin"), dcc.Input(id="zmin", type="number", style={"width": 70}),
            html.Label("zmax"), dcc.Input(id="zmax", type="number", style={"width": 70}),
            dcc.Dropdown(id="colorscale",
                         options=[{"label": n, "value": n} for n in _COLORSCALES],
                         value="Viridis", style={"width": 110}),
            dcc.Checklist(id="log", options=[{"label": "log", "value": "log"}], value=[]),
            html.B("  Calibration (keV): "),
            html.Label("m"), dcc.Input(id="m", type="number", step=0.001, style={"width": 70}),
            html.Label("q"), dcc.Input(id="q", type="number", step=0.001, style={"width": 70}),
        ]),
        html.Div(id="msg", style={"color": "red"}),
        dcc.Store(id="roi-store", data=[0, H, 0, W]),
        dcc.Store(id="ch-store", data=[0, C]),
        dcc.Store(id="points-store", data=[]),
    ])

    @app.callback(Output("map", "figure"),
                  Input("ch-store", "data"),
                  Input("zmin", "value"), Input("zmax", "value"),
                  Input("colorscale", "value"), Input("log", "value"))
    def _map_figure(ch, zmin, zmax, cscale, log):
        s, e = ch
        # dash 4: callbacks return a list of values, even for a single Output
        return [map_figure(cube, s, e, zmin, zmax, cscale, bool(log))]

    @app.callback(Output("spectrum", "figure"),
                  Input("roi-store", "data"), Input("points-store", "data"),
                  Input("m", "value"), Input("q", "value"))
    def _spec_figure(roi, pts, m, q):
        return [spectrum_figure(cube, roi=roi, points=pts, m=m, q=q)]

    @app.callback(Output("roi-store", "data"),
                  Input("map", "relayoutData"), Input("apply-roi", "n_clicks"),
                  State("roi-store", "data"),
                  State("roi-y0", "value"), State("roi-y1", "value"),
                  State("roi-x0", "value"), State("roi-x1", "value"),
                  prevent_initial_call=True)
    def _set_roi(rl, n, cur, y0, y1, x0, x1):
        if rl:
            r = _roi_from_relayout(rl, H, W)
            if r is not None:
                return [r]
        if n:
            return [_clamp_roi(H, W, y0, y1, x0, x1)]
        return [cur]

    @app.callback(Output("ch-store", "data"),
                  Input("spectrum", "relayoutData"), Input("apply-ch", "n_clicks"),
                  State("ch-store", "data"),
                  State("ch-s", "value"), State("ch-e", "value"),
                  prevent_initial_call=True)
    def _set_ch(rl, n, cur, s, e):
        if rl:
            r = _ch_from_relayout(rl, C)
            if r is not None:
                return [r]
        if n:
            s = max(0, min(int(s or 0), C))
            e = max(0, min(int(e or C), C))
            return [[min(s, e), max(s, e)]]
        return [cur]

    @app.callback([Output("points-store", "data"), Output("msg", "children")],
                  Input("apply-points", "n_clicks"), Input("clear", "n_clicks"),
                  State("pt-rows", "value"), State("pt-cols", "value"),
                  prevent_initial_call=True)
    def _set_points(n_pt, n_cl, rows_s, cols_s):
        if n_cl:
            return [], ""
        try:
            pts = [[r, c] for r, c in _parse_points(rows_s, cols_s)]
            bad = [p for p in pts if not (0 <= p[0] < H and 0 <= p[1] < W)]
            if bad:
                raise ValueError(f"point {bad[0]} out of bounds ({H}x{W})")
            return pts, ""
        except ValueError as e:
            return no_update, str(e)

    return app