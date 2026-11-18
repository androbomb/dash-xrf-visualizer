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
bump forces the figure callbacks to recompute. Elemental gallery
results are committed as a revision-stamped record: a stale result
(dataset changed during compute) is discarded instead of published (F11).

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
import json
import math
import re
import struct
import threading
import zipfile
import zlib
from io import BytesIO
from html import escape as html_escape
from pathlib import Path

import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, no_update

from ..data import download_cube, load_cube_bytes, load_xrf_lines, validate_cube_array
from ..processing import (
    calibrate,
    map_from_channels,
    spectrum_from_points,
    spectrum_from_roi,
)
from .results import ResultRecord, try_commit

_COLORSCALES = ("Greys_r", "Viridis", "Plasma", "Hot", "Jet")
_ACCENT = "#4196B4"  # legacy infn_lightblue
_XRF_LINES = load_xrf_lines()

# Dash 4.x dcc components (Dropdown, Checklist, ...) are styled with the
# built-in LIGHT --Dash-* token palette; Dash 4.5 has no dark-mode detection,
# so under darkly they render e.g. white dropdown text on a white trigger.
# CSS custom properties inherit, so defining the dark values on the layout
# root re-themes every dcc component to match darkly.
_DASH_DARK_VARS = {
    "--Dash-Text-Strong": "rgba(233, 236, 239, 0.95)",
    "--Dash-Text-Weak": "rgba(233, 236, 239, 0.65)",
    "--Dash-Text-Disabled": "rgba(233, 236, 239, 0.45)",
    "--Dash-Fill-Inverse-Strong": "#343a40",
    "--Dash-Fill-Interactive-Strong": _ACCENT,
    "--Dash-Fill-Interactive-Weak": "rgba(65, 150, 180, 0.15)",
    "--Dash-Fill-Disabled": "rgba(255, 255, 255, 0.08)",
    "--Dash-Stroke-Strong": "rgba(255, 255, 255, 0.25)",
    "--Dash-Shading-Strong": "rgba(0, 0, 0, 0.55)",
    "--Dash-Shading-Weak": "rgba(0, 0, 0, 0.35)",
}


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
           and rl[k] for k in rl):
        return [0, H, 0, W]
    if not any(k in rl for k in ("xrange", "yrange", "xaxis.range", "yaxis.range",
                                  "xaxis.range[0]", "xaxis.range[1]",
                                  "yaxis.range[0]", "yaxis.range[1]")):
        return None
    xa, xb = rl.get("xrange", rl.get("xaxis.range", (
        rl.get("xaxis.range[0]", 0), rl.get("xaxis.range[1]", W))))
    ya, yb = rl.get("yrange", rl.get("yaxis.range", (
        rl.get("yaxis.range[0]", 0), rl.get("yaxis.range[1]", H))))
    xa, xb = sorted((xa, xb))
    ya, yb = sorted((ya, yb))
    # F6: the drawn edge of pixel i is i-0.5, so floor(a+0.5)/ceil(b) is
    # the exact inverse of what the user selected on screen.
    x0 = max(0, min(int(np.floor(xa + 0.5)), W))
    x1 = max(0, min(int(np.ceil(xb)), W))
    y0 = max(0, min(int(np.floor(ya + 0.5)), H))
    y1 = max(0, min(int(np.ceil(yb)), H))
    return [y0, y1, x0, x1]


def _roi_from_selection(selection, H, W):
    if not selection or not selection.get("range"):
        return None
    ranges = selection["range"]
    if "x" not in ranges or "y" not in ranges:
        return None
    return _roi_from_relayout({"xrange": ranges["x"], "yrange": ranges["y"]}, H, W)


def _png_chunk(kind, data):
    return (struct.pack(">I", len(data)) + kind + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))


def _map_png_bytes(image, colorscale="Greys_r", zmin=None, zmax=None, log=False):
    """Encode one source map value per RGBA PNG pixel, with no plot furniture."""
    image = np.asarray(image, dtype=np.float64)
    if log:
        image = np.log1p(image)
        zmin = np.log1p(max(0.0, float(zmin))) if zmin is not None else None
        zmax = np.log1p(max(0.0, float(zmax))) if zmax is not None else None
    valid = np.isfinite(image)
    if valid.any():
        low = float(zmin) if zmin is not None else float(image[valid].min())
        high = float(zmax) if zmax is not None else float(image[valid].max())
    else:
        low, high = 0.0, 1.0
    if high == low:
        # F7: same constant-range rule as the figure, which uses [v, v+1]
        high = low + 1.0
    scaled = np.clip((image - low) / (high - low), 0, 1)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)
    rgb = np.rint(np.asarray(sample_colorscale(
        colorscale, scaled.ravel().tolist(), colortype="tuple")).reshape(*image.shape, 3) * 255
    ).astype(np.uint8)
    rgba = np.dstack((rgb, np.where(valid, 255, 0).astype(np.uint8)))
    height, width = image.shape
    rows = b"".join(b"\x00" + rgba[row].tobytes() for row in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", header)
            + _png_chunk(b"IDAT", zlib.compress(rows)) + _png_chunk(b"IEND", b""))


def _ch_from_relayout(rl, C, m=None, q=0, energy=None):
    """Spectrum zoom -> channel interval [s, e] from relayoutData; None = ignore.

    energy: file energy grid (keV); when the x-axis is that grid (no manual
    calibration) the zoom range is inverted with np.searchsorted (F4).
    """
    if not rl:
        return None
    if any(k in ("xautorange", "xaxis.autorange") and rl[k] for k in rl):
        return [0, C]
    if not any(k in rl for k in ("xrange", "xaxis.range", "xaxis.range[0]", "xaxis.range[1]")):
        return None
    a, b = rl.get("xrange", rl.get("xaxis.range", (
        rl.get("xaxis.range[0]", 0), rl.get("xaxis.range[1]", C))))
    a, b = sorted((a, b))
    if m is not None:
        a, b = (a - q) / m, (b - q) / m
        a, b = sorted((a, b))
        eps = np.finfo(float).eps * max(1, abs(a), abs(b)) * 4
        a, b = a + eps, b - eps
    elif energy is not None:
        grid = np.asarray(energy, dtype=np.float64)
        eps = 4.0 * np.finfo(float).eps * max(1.0, abs(a), abs(b))
        a = float(np.searchsorted(grid, a - eps, side="left"))
        b = float(np.searchsorted(grid, b + eps, side="right"))
    # F6: floor(a+0.5)/ceil(b) is the exact inverse of the drawn pixel edges
    s = max(0, min(int(np.floor(a + 0.5)), C))
    e = max(0, min(int(np.ceil(b)), C))
    return [s, e]


def _clamp_roi(H, W, y0, y1, x0, x1):
    """Clamp/sort a numeric ROI to [0, H] x [0, W]. None means "not set" and
    falls to the box edge (y0/x0 -> 0, y1/x1 -> H/W, so all-None = full frame);
    any explicit value (including 0) is preserved and clamped (F3)."""
    y0 = 0 if y0 is None else max(0, min(H, int(y0)))
    y1 = H if y1 is None else max(0, min(H, int(y1)))
    x0 = 0 if x0 is None else max(0, min(W, int(x0)))
    x1 = W if x1 is None else max(0, min(W, int(x1)))
    return [min(y0, y1), max(y0, y1), min(x0, x1), max(x0, x1)]


def _parse_points(rows_s, cols_s):
    """'10, 19' + '30, 39' -> [(10, 30), (19, 39)]; ValueError on bad input."""
    def _ints(s):
        return [int(x) for x in re.split(r"[,; ]+", str(s).strip()) if x]
    rows, cols = _ints(rows_s), _ints(cols_s)
    if len(rows) != len(cols):
        raise ValueError(f"rows ({len(rows)}) and cols ({len(cols)}) must match")
    return list(zip(rows, cols))


def _line_channel_range(line_energy, width, calibration, channels, energy=None):
    """Return the half-open ADC channel interval for a centered keV window.

    energy: file energy grid (keV); the window is then inverted with
    np.searchsorted (exact for non-uniform grids, F4). Otherwise E = m*i + q
    with a few-ulp edge tolerance so decimal borders land on the right side
    (F5): a channel center exactly on the edge is included, one just outside
    is not.
    """
    m, q = float(calibration["m"]), float(calibration["q"])
    line_energy = float(line_energy)
    width = float(str(width).strip().replace(",", "."))
    if not all(math.isfinite(v) for v in (m, q, line_energy, width)) or m <= 0 or width <= 0:
        raise ValueError("Calibration must be valid and width must be positive")
    lo, hi = line_energy - width / 2, line_energy + width / 2
    if energy is not None:
        grid = np.asarray(energy, dtype=np.float64)
        eps = 4.0 * np.finfo(float).eps * max(1.0, abs(lo), abs(hi))
        start = max(0, int(np.searchsorted(grid, lo - eps, side="left")))
        end = min(int(channels), int(np.searchsorted(grid, hi + eps, side="right")))
    else:
        lo_ch, hi_ch = (lo - q) / m, (hi - q) / m
        eps = 4.0 * np.finfo(float).eps * max(1.0, abs(lo_ch), abs(hi_ch))
        start = max(0, int(np.ceil(lo_ch - eps)))
        end = min(int(channels), int(np.floor(hi_ch + eps)) + 1)
    if start >= end:
        raise ValueError("Selected line and width do not overlap cube channels")
    return start, end


def _fit_energy_grid(grid):
    """Least-squares (m, q) of E = m*i + q against a file energy grid (F4)."""
    idx = np.arange(len(grid), dtype=np.float64)
    m, q = np.polyfit(idx, np.asarray(grid, dtype=np.float64), 1)
    return float(m), float(q)


def _cal_matches_grid(calibration, grid):
    """True when the stored m/q is the file axis itself (the linear fit of
    the grid), i.e. the user has not overridden it with a manual calibration."""
    if not calibration or grid is None:
        return False
    try:
        m = float(calibration.get("m"))
        q = float(calibration.get("q") or 0.0)
    except (TypeError, ValueError):
        return False
    fit_m, fit_q = _fit_energy_grid(grid)
    return (abs(m - fit_m) <= 1e-9 * max(1.0, abs(fit_m))
            and abs(q - fit_q) <= 1e-9 * max(1.0, abs(fit_q)))


def _x_on(xmode, m, grid):
    """X-axis mode. The stored toggle (user choice) wins; xmode=None means
    "store not wired" (legacy callers and tests) and keeps the old behavior:
    energy whenever a manual calibration or a file grid is active."""
    if xmode is None:
        return m is not None or grid is not None
    return bool(xmode) and (m is not None or grid is not None)


def _add_elemental_line(rows, line_id):
    if line_id in {row["line_id"] for row in rows}:
        return rows
    return [*rows, {"line_id": line_id, "width": 0.1}]


def _calibration_from_points(points):
    if len(points) < 2:
        raise ValueError("Add at least two channel/energy points")
    x = np.asarray([int(point["channel"]) for point in points], dtype=np.float64)
    y = np.asarray([float(str(point["energy"]).strip().replace(",", "."))
                     for point in points], dtype=np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.unique(x).size < 2:
        raise ValueError("Channels must include at least two distinct integers and energies must be finite")
    if len(points) == 2:
        m = (y[1] - y[0]) / (x[1] - x[0])
    else:
        m = float(np.polyfit(x, y, 1)[0])
    q = float(y.mean() - m * x.mean())
    if not math.isfinite(m) or m <= 0 or not math.isfinite(q):
        raise ValueError("Calibration slope must be positive and coefficients finite")
    return float(m), q


def _parse_calibration_point(channel, energy):
    x = float(channel)
    y = float(str(energy).strip().replace(",", "."))
    if not math.isfinite(x) or not x.is_integer() or not math.isfinite(y):
        raise ValueError("Channel must be an integer and energy must be finite")
    return {"channel": int(x), "energy": y}


def _elemental_map_figure(image, title, zmin, zmax, colorscale="Greys_r", log=False):
    shown = np.log1p(image) if log else image
    fig = px.imshow(shown, color_continuous_scale=colorscale or "Greys_r", aspect="equal",
                    labels={"x": "x (col)", "y": "y (row)", "color": "sum"})
    if log:
        zmin, zmax = np.log1p(max(0.0, zmin)), np.log1p(max(0.0, zmax))
    if zmin == zmax:
        zmax = zmin + 1.0
    fig.update_coloraxes(cmin=zmin, cmax=zmax)
    fig.update_layout(title=title, margin=dict(l=40, r=10, t=55, b=30))
    return fig


def _elemental_spectrum_figure(spectrum, calibration, item, log=False):
    energy = calibrate(len(spectrum), calibration["m"], calibration["q"])
    y = np.log1p(spectrum) if log else spectrum
    fig = go.Figure(go.Scatter(x=energy, y=y, mode="lines", name="Full-field sum"))
    low, high = item["integration_zone_keV"]
    fig.add_vrect(x0=low, x1=high, fillcolor="#f39c12", opacity=0.2, line_width=0,
                  annotation_text="Integration zone", annotation_position="top left")
    fig.add_vline(x=item["line_centre_keV"], line_dash="dash", line_color="#e67e22",
                  annotation_text=f"{item['line_name']} center")
    fig.update_layout(title=f"Full-field spectrum — {item['line_name']} ({item['integration_width_keV']:g} keV)",
                      xaxis_title="Energy (keV)",
                      yaxis_title="log1p(sum counts)" if log else "sum counts",
                      margin=dict(l=55, r=15, t=55, b=40))
    return fig


def _spectrum_svg_bytes(spectrum, calibration, item, log=False):
    """Write a labeled, vector spectrum plot without requiring a raster backend."""
    energies = calibrate(len(spectrum), calibration["m"], calibration["q"])
    values = np.log1p(spectrum) if log else np.asarray(spectrum, dtype=np.float64)
    width, height = 1200, 720
    left, right, top, bottom = 95, 30, 90, 80
    x_min, x_max = float(energies[0]), float(energies[-1])
    if x_max == x_min:
        x_max = x_min + 1
    finite = values[np.isfinite(values)]
    y_max = float(finite.max()) if finite.size else 1.0
    y_max = y_max if y_max > 0 else 1.0
    x = lambda value: left + (value - x_min) / (x_max - x_min) * (width - left - right)
    y = lambda value: height - bottom - value / y_max * (height - top - bottom)
    points = " ".join(f"{x(float(e)):.2f},{y(float(v)):.2f}"
                       for e, v in zip(energies, values)
                       if np.isfinite(e) and np.isfinite(v))
    title = html_escape(f"Full-field spectrum — {item['line_name']} ({item['integration_width_keV']:g} keV)")
    zone_low, zone_high = item["integration_zone_keV"]
    zone_left, zone_right = max(x_min, zone_low), min(x_max, zone_high)
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{left}" y="42" font-family="sans-serif" font-size="25">{title}</text>']
    for i in range(6):
        fraction = i / 5
        py, tick = y(y_max * fraction), y_max * fraction
        px, xtick = left + (width - left - right) * fraction, x_min + (x_max - x_min) * fraction
        lines.extend((f'<line x1="{left}" y1="{py:.2f}" x2="{width-right}" y2="{py:.2f}" stroke="#dddddd"/>',
                      f'<text x="{left-12}" y="{py+5:.2f}" text-anchor="end" font-family="sans-serif" font-size="15">{tick:.4g}</text>',
                      f'<line x1="{px:.2f}" y1="{top}" x2="{px:.2f}" y2="{height-bottom}" stroke="#eeeeee"/>',
                      f'<text x="{px:.2f}" y="{height-bottom+27}" text-anchor="middle" font-family="sans-serif" font-size="15">{xtick:.4g}</text>'))
    lines.extend((f'<rect x="{x(zone_left):.2f}" y="{top}" width="{max(0, x(zone_right)-x(zone_left)):.2f}" height="{height-top-bottom}" fill="#f39c12" opacity="0.2"/>',
                  f'<line x1="{x(item["line_centre_keV"]):.2f}" y1="{top}" x2="{x(item["line_centre_keV"]):.2f}" y2="{height-bottom}" stroke="#e67e22" stroke-dasharray="8 6" stroke-width="2"/>',
                  f'<polyline points="{points}" fill="none" stroke="#5968e8" stroke-width="2"/>',
                  f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#222"/>',
                  f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#222"/>',
                  f'<text x="{(left+width-right)/2}" y="{height-18}" text-anchor="middle" font-family="sans-serif" font-size="18">Energy (keV)</text>',
                  f'<text x="24" y="{height/2}" text-anchor="middle" transform="rotate(-90 24 {height/2})" font-family="sans-serif" font-size="18">{"log1p(sum counts)" if log else "sum counts"}</text>',
                  '</svg>'))
    return "\n".join(lines).encode("utf-8")


# ------------------------------------------------------------ figure builders

def map_figure(cube, ch_start, ch_end, zmin=None, zmax=None,
               colorscale="Greys_r", log=False, m=None, q=None, roi=None, points=None,
               show_energy=True, grid=None, image=None):
    """Current map: plain float64 channel sum, display options applied.

    ``image``: precomputed (H, W) sum from the UI cache; None -> reduce now.
    """
    if image is None:
        image = map_from_channels(cube, ch_start, ch_end)
    if log:
        image = np.log1p(image)
        zmin = np.log1p(max(0.0, float(zmin))) if zmin is not None else None
        zmax = np.log1p(max(0.0, float(zmax))) if zmax is not None else None
    fig = px.imshow(image, color_continuous_scale=colorscale or "Greys_r",
                    aspect="equal", labels={"x": "x (col)", "y": "y (row)", "color": "sum"})
    fig.update_layout(dragmode="select")
    if zmin is not None or zmax is not None:
        fig.update_coloraxes(cmin=float(zmin) if zmin is not None else None,
                             cmax=float(zmax) if zmax is not None else None)
    grid = np.asarray(grid, dtype=np.float64) if grid is not None else None
    if show_energy and m is not None:
        interval = f"{m * ch_start + (q or 0):.12g}–{m * ch_end + (q or 0):.12g} keV"
    elif show_energy and grid is not None and 0 <= ch_start < ch_end <= grid.shape[0]:
        # F4: the file energy axis (keV), even when non-uniform
        interval = f"{grid[ch_start]:.12g}–{grid[ch_end - 1]:.12g} keV"
    else:
        interval = f"{ch_start}–{ch_end} ADC channels"
    fig.update_layout(title=f"Integrated over {interval}",
                      margin=dict(l=40, r=10, t=55, b=30))
    if roi:
        y0, y1, x0, x1 = roi
        # Disable imshow's reversed autorange so the explicit row bounds win.
        fig.update_layout(xaxis_autorange=False, yaxis_autorange=False,
                          xaxis_range=[x0 - 0.5, x1 - 0.5],
                          yaxis_range=[y1 - 0.5, y0 - 0.5])
    if points:
        fig.add_scatter(x=[p[1] for p in points], y=[p[0] for p in points],
                        mode="markers", marker={"size": 10, "color": "white",
                                                  "line": {"color": "black", "width": 1}},
                        name="Selected points")
    return fig


def spectrum_figure(cube, roi=None, points=None, m=None, q=None, channel_range=None,
                    log=False, show_energy=True, grid=None, spectrum=None):
    """Current spectrum: plain float64 pixel sum (points win over ROI).

    With m != 0 the x-axis is the calibrated energy axis E = m*i + q (keV);
    with a file energy grid it is that grid (F4); with show_energy=False the
    axis is always raw ADC channels (the calibration-check toggle).
    ``spectrum``: precomputed (C,) sum from the UI cache; None -> reduce now.
    """
    H, W, C = cube.shape
    if points:
        selection = "Points: " + "; ".join(f"(row {y}, col {x})" for y, x in points)
        spec = (spectrum_from_points(cube, [p[0] for p in points],
                                     [p[1] for p in points])
                if spectrum is None else spectrum)
    else:
        y0, y1, x0, x1 = roi if roi else [0, H, 0, W]
        selection = f"Box: y {y0}–{y1}, x {x0}–{x1}"
        spec = (spectrum_from_roi(cube, y0, y1, x0, x1)
                if spectrum is None else spectrum)
    if log:
        spec = np.log1p(spec)
    grid = np.asarray(grid, dtype=np.float64) if grid is not None else None
    if show_energy and m is not None:
        e = calibrate(C, m, q or 0.0)
        x, xlab = e, f"E (keV; m={m}, q={q or 0})"
    elif show_energy and grid is not None and grid.shape[0] == C:
        x, xlab = grid, "E (keV; file axis)"
    else:
        x, xlab = np.arange(C), "channel (ADC)"
    # go.Scatter (not px.line): px.line wants a pandas frame for plain arrays
    fig = go.Figure(go.Scatter(x=x, y=spec, mode="lines"))
    fig.update_layout(title=selection, xaxis_title=xlab,
                      yaxis_title="log1p(sum counts)" if log else "sum counts",
                      margin=dict(l=50, r=10, t=55, b=30))
    if channel_range:
        start, end = channel_range
        if show_energy and m is not None:
            limits = [m * i + (q or 0.0) for i in (start - 0.5, end - 0.5)]
        elif (show_energy and grid is not None and grid.shape[0] == C
                and 0 <= start < end <= C):
            limits = [float(grid[start]), float(grid[end - 1])]
        else:
            limits = [start - 0.5, end - 0.5]
        fig.update_xaxes(range=sorted(limits))
    return fig


# ----------------------------------------------------------------------- app

def _as_array(cube):
    """Accept None, an ndarray, or an XrfCube (uses its .array)."""
    if cube is None:
        return None
    if isinstance(cube, np.ndarray):
        arr = cube
    else:
        arr = getattr(cube, "array", None)
        if arr is None:
            raise TypeError(f"cube must be an XrfCube or ndarray, got {type(cube).__name__}")
    arr = validate_cube_array(arr, "cube")
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
             "name": Path(getattr(cube, "path", "") or "cube").name,
             "energy": getattr(cube, "energy", None),
             "revision": 0,   # server-side dataset revision (F11)
             "result": None,  # committed ResultRecord, or None
             "map_cache": None,   # last (H, W) sum, keyed (revision, s, e)
             "spec_cache": None}  # last (C,) sum, keyed (revision, selection)
    state_lock = threading.Lock()  # guards revision bump, result commit, cache slots

    def _cached_map_image(c, revision, s, e):
        """Last (H, W) channel sum for (revision, s, e): a palette, ROI,
        title or scale change must not re-run the reduction (audit item 15)."""
        key = (revision, int(s), int(e))
        with state_lock:
            hit = state["map_cache"]
        if hit is not None and hit[0] == key:
            return hit[1]
        image = map_from_channels(c, s, e)
        with state_lock:
            state["map_cache"] = (key, image)
        return image

    def _cached_spectrum(c, revision, roi, pts):
        """Last (C,) sum for (revision, selection): same rule as _cached_map_image."""
        if pts:
            key = (revision, "pts", tuple(tuple(p) for p in pts))
            get = lambda: spectrum_from_points(c, [p[0] for p in pts],
                                               [p[1] for p in pts])
        else:
            y0, y1, x0, x1 = roi if roi else [0, c.shape[0], 0, c.shape[1]]
            key = (revision, "roi", y0, y1, x0, x1)
            get = lambda: spectrum_from_roi(c, y0, y1, x0, x1)
        with state_lock:
            hit = state["spec_cache"]
        if hit is not None and hit[0] == key:
            return hit[1]
        spec = get()
        with state_lock:
            state["spec_cache"] = (key, spec)
        return spec
    H, W, C = arr.shape if arr is not None else (0, 0, 0)
    app = _XrfDash(__name__, external_stylesheets=[dbc.themes.DARKLY])
    app.layout = dbc.Container(fluid=True, style=_DASH_DARK_VARS, children=[
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
                html.Div(dcc.Graph(id="map", responsive=True,
                                   config={"doubleClick": "reset"},
                                   style={"height": "100%", "width": "100%"}),
                         id="map-frame", style={"width": "100%",
                                                 "aspectRatio": W / H if H else 1}),
                dbc.Row(className="g-2 align-items-center mb-2", children=[
                    dbc.Col([dbc.Label("zmin", html_for="zmin"), dbc.Input(id="zmin", type="number")], width=2),
                    dbc.Col([dbc.Label("zmax", html_for="zmax"), dbc.Input(id="zmax", type="number")], width=2),
                    dbc.Col([dbc.Label("colormap", html_for="colorscale"), dcc.Dropdown(id="colorscale",
                                 options=[{"label": n, "value": n} for n in _COLORSCALES],
                                 value="Greys_r")], width=3),
                    dbc.Col(dcc.Checklist(id="log", options=[{"label": "log", "value": "log"}],
                                          value=[], labelStyle={"color": "var(--bs-body-color)"}),
                            width=1, className="pt-4"),
                    dbc.Col(dbc.Button("Apply scale", id="apply-scale", n_clicks=0,
                                       size="sm", outline=True, color="info"), width=2),
                    dbc.Col(dbc.Button("Reset scale", id="reset-scale", n_clicks=0,
                                       size="sm", outline=True, color="secondary"), width=2),
                ]),
                html.B("Interval → map"),
                dbc.Row(className="g-2 align-items-end mb-2", children=[
                    dbc.Col([dbc.Label("ch begin", html_for="ch-s"), dbc.Input(id="ch-s", type="number", value=0)], width=4),
                    dbc.Col([dbc.Label("ch end", html_for="ch-e"), dbc.Input(id="ch-e", type="number", value=C)], width=4),
                    dbc.Col(dbc.Button("Apply interval", id="apply-ch", n_clicks=0, outline=True, color="info"), width=4),
                ]),
                dbc.Button("Download current map", id="download-map", n_clicks=0,
                           outline=True, color="info", className="w-100"),
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
                    dbc.Col([dbc.Label("m", html_for="m"), dcc.Input(
                        id="m", type="text", value="", className="form-control")], width=6),
                    dbc.Col([dbc.Label("q", html_for="q"), dcc.Input(
                        id="q", type="text", value="", className="form-control")], width=6),
                    dbc.Col(dbc.Button("Apply calibration", id="apply-calibration", n_clicks=0, outline=True, color="info"), width=12),
                    dbc.Col(html.Div(id="cal-msg", className="text-info small"), width=12),
                    dbc.Col(dcc.Checklist(id="show-energy", options=[
                        {"label": "X-axis: Energy (keV) — swap ADC/Energy to check calibration",
                         "value": "energy"}], value=[],
                        labelStyle={"color": "var(--bs-body-color)"}), width=12),
                ]),
                html.B("Fit calibration from channel/energy points"),
                dbc.Row(className="g-2 align-items-end mt-1", children=[
                    dbc.Col([dbc.Label("x (integer channel)"), dbc.Input(
                        id="cal-point-x", type="number", step=1)], width=3),
                    dbc.Col([dbc.Label("y (energy, keV)"), dcc.Input(
                        id="cal-point-y", type="text", className="form-control",
                        placeholder="e.g. 6.404123456")], width=4),
                    dbc.Col(dbc.Button("Add point", id="cal-point-add", n_clicks=0,
                                       outline=True, color="info"), width=2),
                    dbc.Col(dbc.Button("Fit & apply", id="cal-point-fit", n_clicks=0,
                                       disabled=True, outline=True, color="info"), width=3),
                    dbc.Col(html.Div(id="cal-point-rows"), width=12),
                    dbc.Col(html.Div(id="cal-point-msg", className="text-info small"), width=12),
                ]),
                dbc.Row(className="g-2 align-items-end mt-1", children=[
                    dbc.Col([html.B("Pick line"), dbc.Label("Select Line"), dcc.Dropdown(
                        id="line-select", options=[
                            {"label": f"{line['element']} ({line['value']:g} keV)", "value": line["value"]}
                            for line in _XRF_LINES], placeholder="Search XRF line…",
                        searchable=True, clearable=False)], width=5),
                    dbc.Col([dbc.Label("Select width (keV)"), dbc.Input(
                        id="line-width", type="number", min=0, step="any", value=0.1)], width=4),
                    dbc.Col(dbc.Button("Apply", id="apply-line", n_clicks=0, disabled=True,
                                       outline=True, color="info"), width=3),
                    dbc.Col(html.Div(id="line-selected", className="text-info small"), width=12),
                    dbc.Col(html.Div(id="line-msg", className="text-info small"), width=12),
                ]),
            ]),
        ]),
        html.Hr(),
        html.H4("Elemental maps (full field)", style={"color": _ACCENT}),
        dbc.Row(className="g-2 align-items-end mb-2", children=[
            dbc.Col([dbc.Label("Add XRF line"), dcc.Dropdown(
                id="elemental-add-line", options=[
                    {"label": f"{line['element']} ({line['value']:g} keV)", "value": i}
                    for i, line in enumerate(_XRF_LINES)],
                placeholder="Search XRF line…", searchable=True, clearable=True)], width=5),
            dbc.Col(dbc.Button("Add line", id="elemental-add", n_clicks=0,
                               outline=True, color="info", disabled=True), width=2),
            dbc.Col(dcc.Checklist(id="elemental-shared-scale", options=[
                {"label": "Same scale for all", "value": "shared"}],
                value=["shared"], labelStyle={"color": "var(--bs-body-color)"}), width=3),
            dbc.Col(dbc.Button("Compute", id="elemental-compute", n_clicks=0,
                               outline=True, color="info", disabled=True), width=2),
        ]),
        html.Div(id="elemental-rows"),
        html.Div(id="elemental-msg", className="text-info small"),
        dbc.Row(className="g-2 align-items-center", children=[
            dbc.Col(dbc.Button("Previous", id="elemental-prev", n_clicks=0,
                               outline=True, color="secondary"), width="auto"),
            dbc.Col(html.Span(id="elemental-position"), width="auto"),
            dbc.Col(dbc.Button("Next", id="elemental-next", n_clicks=0,
                               outline=True, color="secondary"), width="auto"),
            dbc.Col(dbc.Button("Download all PNGs (ZIP)", id="elemental-download",
                               n_clicks=0, outline=True, color="info"), width="auto"),
        ]),
        dbc.Row(className="g-2", children=[
            dbc.Col(dcc.Graph(id="elemental-map", responsive=True), md=6),
            dbc.Col(dcc.Graph(id="elemental-spectrum", responsive=True), md=6),
        ]),
        html.Div(id="msg", style={"color": "orange"}),
        dcc.Store(id="roi-store", data=[0, H, 0, W]),
        dcc.Store(id="ch-store", data=[0, C]),
        dcc.Store(id="points-store", data=[]),
        dcc.Store(id="scale-store", data={"zmin": None, "zmax": None}),
        dcc.Store(id="cal-store", data=None),
        dcc.Store(id="xmode-store", data=False),
        dcc.Store(id="cal-points-store", data=[]),
        dcc.Store(id="elemental-rows-store", data=[]),
        dcc.Store(id="elemental-meta", data=[]),
        dcc.Store(id="elemental-index", data=0),
        dcc.Store(id="cal-input-store", data=None),
        dcc.Store(id="gen", data=0),
        dcc.Download(id="map-download"),
        dcc.Download(id="elemental-download-data"),
        dcc.Store(id="map-reset-store", data=0),
        dcc.Store(id="map-bind-store", data=0),
    ])

    @app.callback(Output("map", "figure"),
                  Input("ch-store", "data"), Input("roi-store", "data"),
                  Input("points-store", "data"), Input("colorscale", "value"), Input("log", "value"),
                  Input("scale-store", "data"), Input("cal-store", "data"),
                  Input("xmode-store", "data"), Input("gen", "data"))
    def _map_figure(ch, roi, points, cscale, log, scale, calibration, xmode, _gen):
        c, rev = state["cube"], state["revision"]
        if c is None:
            return _empty_figure("No cube loaded — upload a file or paste a link above")
        s, e = ch or [0, c.shape[2]]
        # F3: None means "not set" only; an empty interval (e <= s) stays
        # empty and renders zero maps instead of the full channel range.
        if s is None:
            s = 0
        if e is None:
            e = c.shape[2]
        # dash 4.5: single Output -> return the raw value (no [..] wrap)
        scale = scale or {}
        m, q = (calibration or {}).get("m"), (calibration or {}).get("q")
        grid = state["energy"] if m is None else None
        return map_figure(c, s, e, scale.get("zmin"), scale.get("zmax"),
                          cscale, bool(log), m, q, roi, points,
                          show_energy=_x_on(xmode, m, grid), grid=grid,
                          image=_cached_map_image(c, rev, s, e))

    app.clientside_callback(
        """function(figure) {
            const root = document.getElementById("map");
            if (root && !root.__mapResetBound) {
                root.__mapResetBound = true;
                root.addEventListener("dblclick", function(event) {
                    const plot = root.querySelector(".js-plotly-plot");
                    if (plot && plot.contains(event.target)) {
                        window.dash_clientside.set_props("map-reset-store", {data: Date.now()});
                    }
                }, true);
            }
            return Date.now();
        }""",
        Output("map-bind-store", "data"), Input("map", "figure"),
    )

    app.clientside_callback(
        """function(n) {
            if (!n) return window.dash_clientside.no_update;
            return {
                m: document.getElementById("m").value,
                q: document.getElementById("q").value
            };
        }""",
        Output("cal-input-store", "data"), Input("apply-calibration", "n_clicks"),
        prevent_initial_call=True,
    )

    @app.callback(Output("map-frame", "style"), Input("roi-store", "data"))
    def _map_frame_style(roi):
        c = state["cube"]
        if c is None:
            return {"width": "100%", "aspectRatio": 1}
        H, W = c.shape[:2]
        y0, y1, x0, x1 = _clamp_roi(H, W, *(roi or [0, H, 0, W]))
        if y1 <= y0 or x1 <= x0:
            y0, y1, x0, x1 = 0, H, 0, W
        return {"width": "100%", "aspectRatio": (x1 - x0) / (y1 - y0)}

    @app.callback(Output("map-download", "data"), Input("download-map", "n_clicks"),
                  State("ch-store", "data"), State("roi-store", "data"),
                  State("colorscale", "value"), State("log", "value"),
                  State("scale-store", "data"), prevent_initial_call=True)
    def _download_map(_n, channels, roi, colorscale, log, scale):
        c, rev = state["cube"], state["revision"]
        if c is None:
            return no_update
        start, end = channels or [0, c.shape[2]]
        image = _cached_map_image(c, rev, start, end)
        # F7: the limits come from the full map, before the ROI crop —
        # exactly what the on-screen figure normalizes with.
        full = np.log1p(image) if log else image
        scale = scale or {}
        zmin = scale.get("zmin")
        zmax = scale.get("zmax")
        if zmin is None:
            zmin = float(full.min())
        if zmax is None:
            zmax = float(full.max())
        if roi:
            y0, y1, x0, x1 = _clamp_roi(*image.shape, *roi)
            image = image[y0:y1, x0:x1]
        if image.size == 0:
            return no_update
        png = _map_png_bytes(image, colorscale or "Greys_r", zmin, zmax, bool(log))
        return dcc.send_bytes(png, "xrf_map.png")

    @app.callback(Output("spectrum", "figure"),
                  Input("roi-store", "data"), Input("points-store", "data"),
                  Input("ch-store", "data"), Input("log", "value"),
                  Input("cal-store", "data"), Input("xmode-store", "data"),
                  Input("gen", "data"))
    def _spec_figure(roi, pts, channels, log, calibration, xmode, _gen):
        c, rev = state["cube"], state["revision"]
        if c is None:
            return _empty_figure("No cube loaded")
        H, W, _ = c.shape
        if pts:  # drop points stale from a previous cube; all stale -> ROI
            keep = [p for p in pts if 0 <= p[0] < H and 0 <= p[1] < W]
            pts = keep or None
        # F3: an empty ROI stays empty (zero spectrum); no full-frame fallback.
        m, q = (calibration or {}).get("m"), (calibration or {}).get("q")
        grid = state["energy"] if m is None else None
        return spectrum_figure(c, roi=roi, points=pts,
                               m=m, q=q,
                               channel_range=channels, log=bool(log),
                               show_energy=_x_on(xmode, m, grid), grid=grid,
                               spectrum=_cached_spectrum(c, rev, roi, pts))

    @app.callback(Output("roi-store", "data"),
                  Input("map", "relayoutData"), Input("map", "selectedData"),
                  Input("map-reset-store", "data"), Input("apply-roi", "n_clicks"),
                  State("roi-store", "data"),
                  State("roi-y0", "value"), State("roi-y1", "value"),
                  State("roi-x0", "value"), State("roi-x1", "value"),
                  prevent_initial_call=True)
    def _set_roi(rl, selected, _map_reset, n, cur, y0, y1, x0, x1):
        c = state["cube"]
        if c is None:
            return cur
        H, W, _ = c.shape
        if "map-reset-store.data" in ctx.triggered_prop_ids:
            return [0, H, 0, W]
        if "apply-roi.n_clicks" in ctx.triggered_prop_ids:
            return _clamp_roi(H, W, y0, y1, x0, x1)
        if ctx.triggered_id == "map":
            if "map.relayoutData" in ctx.triggered_prop_ids:
                r = _roi_from_relayout(rl, H, W)
                if r is not None:
                    return r
            if "map.selectedData" in ctx.triggered_prop_ids:
                r = _roi_from_selection(selected, H, W)
                if r is not None:
                    return r
        return cur

    @app.callback(Output("ch-store", "data"),
                  Input("spectrum", "relayoutData"), Input("apply-ch", "n_clicks"),
                  State("ch-store", "data"),
                  State("ch-s", "value"), State("ch-e", "value"),
                  State("cal-store", "data"), State("xmode-store", "data"),
                  prevent_initial_call=True)
    def _set_ch(rl, n, cur, s, e, calibration, xmode):
        c = state["cube"]
        if c is None:
            return cur
        C = c.shape[2]
        if "apply-ch.n_clicks" in ctx.triggered_prop_ids:
            # F3: an explicit 0 is preserved; only None falls to the edge.
            s = 0 if s is None else max(0, min(int(s), C))
            e = C if e is None else max(0, min(int(e), C))
            return [min(s, e), max(s, e)]
        if ctx.triggered_id == "spectrum":
            cal = calibration or {}
            m, q = cal.get("m"), cal.get("q", 0)
            grid = state["energy"] if m is None else None
            if _x_on(xmode, m, grid) and m is not None:
                r = _ch_from_relayout(rl, C, m, q)
            elif _x_on(xmode, m, grid) and grid is not None:
                r = _ch_from_relayout(rl, C, energy=grid)
            else:
                r = _ch_from_relayout(rl, C)
            if r is not None:
                return r
        return cur

    def _elemental_rows_view(rows):
        return [dbc.Row(className="g-2 align-items-end mb-2", children=[
            dbc.Col(html.Span(
                f"{_XRF_LINES[row['line_id']]['element']} "
                f"({ _XRF_LINES[row['line_id']]['value']:g} keV)"), width=5),
            dbc.Col([dbc.Label("Width (keV)"), dbc.Input(
                id={"type": "elemental-width", "index": row["line_id"]},
                type="text", value=row["width"])], width=4),
            dbc.Col(dbc.Button("Remove", id={"type": "elemental-remove", "index": row["line_id"]},
                               n_clicks=0, outline=True, color="danger"), width=2),
        ]) for row in rows]

    @app.callback([Output("elemental-rows-store", "data"),
                   Output("elemental-rows", "children"),
                   Output("elemental-add-line", "value"),
                   Output("elemental-msg", "children")],
                  Input("elemental-add", "n_clicks"),
                  Input({"type": "elemental-remove", "index": ALL}, "n_clicks"),
                  State("elemental-add-line", "value"),
                  State("elemental-rows-store", "data"),
                  State({"type": "elemental-width", "index": ALL}, "value"),
                  prevent_initial_call=True)
    def _edit_elemental_rows(_add, _remove_clicks, line_id, rows, widths):
        rows = list(rows or [])
        current_widths = {r["line_id"]: v for r, v in zip(rows, widths or [])}
        rows = [{**r, "width": current_widths.get(r["line_id"], r["width"])} for r in rows]
        trigger = ctx.triggered_id
        message = ""
        if trigger == "elemental-add":
            if line_id is None:
                return rows, _elemental_rows_view(rows), no_update, "Select an XRF line first"
            if line_id in {r["line_id"] for r in rows}:
                message = "That line is already in the list"
            rows = _add_elemental_line(rows, line_id)
            return rows, _elemental_rows_view(rows), None, message
        if isinstance(trigger, dict) and trigger.get("type") == "elemental-remove":
            rows = [r for r in rows if r["line_id"] != trigger["index"]]
        return rows, _elemental_rows_view(rows), no_update, ""

    @app.callback([Output("elemental-add", "disabled"),
                   Output("elemental-compute", "disabled"),
                   Output("elemental-prev", "disabled"),
                   Output("elemental-next", "disabled"),
                   Output("elemental-download", "disabled")],
                  Input("cal-store", "data"), Input("elemental-add-line", "value"),
                  Input("elemental-rows-store", "data"), Input("elemental-meta", "data"))
    def _elemental_buttons(calibration, line_id, rows, meta):
        ready = bool(calibration)
        count = len(meta or [])
        return (not (ready and line_id is not None), not (ready and rows),
                count < 2, count < 2, count == 0)

    @app.callback([Output("elemental-meta", "data", allow_duplicate=True),
                   Output("elemental-index", "data", allow_duplicate=True),
                   Output("elemental-msg", "children", allow_duplicate=True)],
                  Input("cal-store", "data"), prevent_initial_call=True)
    def _invalidate_elemental_maps(_calibration):
        # A calibration change invalidates the current result (computed
        # with the old m/q); bumping the revision also rejects any
        # in-flight compute (F11).
        with state_lock:
            state["revision"] += 1
            state["result"] = None
        return [], 0, ""

    @app.callback([Output("elemental-meta", "data", allow_duplicate=True),
                   Output("elemental-msg", "children", allow_duplicate=True),
                   Output("elemental-index", "data", allow_duplicate=True)],
                  Input("elemental-compute", "n_clicks"),
                  State("elemental-rows-store", "data"),
                  State({"type": "elemental-width", "index": ALL}, "value"),
                  State("cal-store", "data"),
                  prevent_initial_call=True)
    def _compute_elemental_maps(_n, rows, widths, calibration):
        c = state["cube"]
        if c is None:
            return no_update, "Load a cube first", no_update
        if not calibration:
            return no_update, "Apply calibration first", no_update
        if len(rows or []) != len(widths or []):
            return no_update, "Each selected line needs a width", no_update
        # Dataset identity at Compute start; the result is committed only
        # if it is still current (F11).
        with state_lock:
            c, revision, source_file = state["cube"], state["revision"], state["name"]
        maps, metadata = [], []
        # Audit item 15: reuse the last reduction for (revision, selection).
        full_spectrum = _cached_spectrum(c, revision, None, None)
        try:
            for row, width in zip(rows or [], widths or []):
                line = _XRF_LINES[row["line_id"]]
                start, end = _line_channel_range(line["value"], width,
                                                 calibration, c.shape[2])
                image = _cached_map_image(c, revision, start, end)
                finite = image[np.isfinite(image)]
                if not finite.size:
                    raise ValueError(f"{line['element']}: map contains no finite values")
                maps.append(image)
                integration_width = float(str(width).strip().replace(",", "."))
                actual_low = calibration["m"] * start + calibration["q"]
                actual_high = calibration["m"] * (end - 1) + calibration["q"]
                zone = [line["value"] - integration_width / 2,
                        line["value"] + integration_width / 2]
                zone_spectrum = full_spectrum[start:end]
                zone_energies = calibrate(c.shape[2], calibration["m"], calibration["q"])[start:end]
                weights = np.where(np.isfinite(zone_spectrum), zone_spectrum, 0.0)
                count_sum = float(weights.sum())
                metadata.append({"label": line["element"], "line_name": line["element"],
                                 "source_file": source_file,
                                 "energy": line["value"], "line_centre_keV": line["value"],
                                 "integration_width_keV": integration_width,
                                 "integration_zone_keV": zone,
                                 "channel_start": start, "channel_end_exclusive": end,
                                 "channel_centres_keV": [float(actual_low), float(actual_high)],
                                 "spectrum_centroid_keV": (
                                     float(np.dot(weights, zone_energies) / count_sum)
                                     if count_sum > 0 else None),
                                 "minimum": float(finite.min()), "maximum": float(finite.max())})
        except (KeyError, IndexError, TypeError, ValueError) as error:
            return no_update, str(error), no_update
        record = ResultRecord(revision=revision, source_file=source_file,
                              cube_shape=tuple(c.shape), calibration=dict(calibration),
                              metadata=metadata, maps=maps, spectrum=full_spectrum)
        if not try_commit(state, state_lock, record):
            # The dataset changed while we computed: never publish a stale
            # result (F11).
            return no_update, "Dataset changed during compute — run Compute again", no_update
        return metadata, f"Computed {len(maps)} full-field elemental maps", 0

    @app.callback(Output("elemental-index", "data", allow_duplicate=True),
                  Input("elemental-prev", "n_clicks"), Input("elemental-next", "n_clicks"),
                  State("elemental-index", "data"), State("elemental-meta", "data"),
                  prevent_initial_call=True)
    def _elemental_navigate(_prev, _next, index, meta):
        count = len(meta or [])
        if not count:
            return 0
        step = -1 if ctx.triggered_id == "elemental-prev" else 1
        return (int(index or 0) + step) % count

    @app.callback(Output("elemental-position", "children"),
                  Input("elemental-meta", "data"), Input("elemental-index", "data"))
    def _elemental_position(meta, index):
        return f"{min(int(index or 0), len(meta) - 1) + 1} / {len(meta)}" if meta else ""

    @app.callback(Output("elemental-map", "figure"),
                  Input("elemental-meta", "data"), Input("elemental-index", "data"),
                  Input("elemental-shared-scale", "value"),
                  Input("colorscale", "value"), Input("log", "value"))
    def _elemental_figure(meta, index, shared, colorscale, log):
        record = state.get("result")
        if not meta or record is None:
            return _empty_figure("Apply calibration, add lines, then Compute")
        index = min(int(index or 0), len(meta) - 1)
        current = meta[index]
        if "shared" in (shared or []):
            low = min(item["minimum"] for item in meta)
            high = max(item["maximum"] for item in meta)
        else:
            low, high = current["minimum"], current["maximum"]
        title = (f"{current['label']} ({current['energy']:g} keV), width "
                 f"{current['integration_width_keV']:g} keV — range {current['minimum']:g}–"
                 f"{current['maximum']:g} counts; scale {low:g}–{high:g} counts"
                 f"{' (log1p)' if log and 'log' in log else ''}")
        return _elemental_map_figure(record.maps[index], title, low, high,
                                     colorscale, bool(log and "log" in log))

    @app.callback(Output("elemental-spectrum", "figure"),
                  Input("elemental-meta", "data"), Input("elemental-index", "data"),
                  Input("log", "value"))
    def _elemental_spectrum(meta, index, log):
        record = state.get("result")
        if not meta or record is None:
            return _empty_figure("Spectrum appears after Compute")
        index = min(int(index or 0), len(meta) - 1)
        return _elemental_spectrum_figure(record.spectrum, record.calibration,
                                          meta[index], bool(log and "log" in log))

    @app.callback([Output("elemental-download-data", "data"),
                   Output("elemental-msg", "children", allow_duplicate=True)],
                  Input("elemental-download", "n_clicks"),
                  State("elemental-meta", "data"),
                  State("elemental-shared-scale", "value"),
                  State("colorscale", "value"), State("log", "value"),
                  prevent_initial_call=True)
    def _download_elemental_maps(_n, meta, shared, colorscale, log):
        record = state.get("result")
        if not meta or record is None:
            return no_update, "Compute maps first"
        maps = record.maps
        calibration = record.calibration
        shared_range = (min(item["minimum"] for item in meta),
                        max(item["maximum"] for item in meta))
        archive = BytesIO()
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            manifest = []
            for item_index, (item, image) in enumerate(zip(meta, maps)):
                low, high = (shared_range if "shared" in (shared or []) else
                             (item["minimum"], item["maximum"]))
                safe_label = re.sub(r"[^A-Za-z0-9_-]+", "_", item["label"]).strip("_")
                filename = (f"{safe_label}_{item['energy']:g}keV_"
                            f"{item['integration_width_keV']:g}keV_{item_index + 1}.png")
                spectrum_filename = filename.removesuffix(".png") + "_spectra.svg"
                bundle.writestr(filename, _map_png_bytes(
                    image, colorscale or "Greys_r", low, high,
                    bool(log and "log" in log)))
                bundle.writestr(spectrum_filename, _spectrum_svg_bytes(
                    record.spectrum, calibration, item,
                    bool(log and "log" in log)))
                manifest.append({
                    "file": filename, "spectrum_file": spectrum_filename,
                    "source_file": item["source_file"],
                    "min_count": low, "max_count": high,
                    "map_min_count": item["minimum"], "map_max_count": item["maximum"],
                    "line_name": item["line_name"],
                    "line_centre_keV": item["line_centre_keV"],
                    "integration_width_keV": item["integration_width_keV"],
                    "integration_zone_keV": item["integration_zone_keV"],
                    "channel_range_adc": [item["channel_start"], item["channel_end_exclusive"]],
                    "channel_centres_keV": item["channel_centres_keV"],
                    "spectrum_centroid_keV": item["spectrum_centroid_keV"],
                    "calibration_keV_per_channel": calibration["m"],
                    "calibration_intercept_keV": calibration["q"],
                    "cube_shape": list(record.cube_shape),
                    "spatial_scope": "full field",
                    "scale_mode": "shared" if "shared" in (shared or []) else "per-line autoscale",
                    "colorscale": colorscale or "Greys_r", "log1p": bool(log and "log" in log),
                })
            bundle.writestr("metadata.json", json.dumps(manifest, indent=2).encode("utf-8"))
        return dcc.send_bytes(archive.getvalue(), "elemental_maps.zip"), \
            f"Downloaded {len(meta)} maps and spectra"

    @app.callback([Output("ch-store", "data", allow_duplicate=True),
                   Output("line-msg", "children")],
                  Input("apply-line", "n_clicks"),
                  State("line-select", "value"), State("line-width", "value"),
                  State("cal-store", "data"),
                  prevent_initial_call=True)
    def _apply_line(_n, line_energy, width, calibration):
        if line_energy is None:
            return no_update, "Select an XRF line"
        if not calibration:
            return no_update, "Apply calibration first"
        if state["cube"] is None:
            return no_update, "Load a cube first"
        try:
            channels = state["cube"].shape[2]
            grid = state["energy"]
            if grid is not None and not _cal_matches_grid(calibration, grid):
                grid = None  # explicit manual calibration overrides the file axis
            start, end = _line_channel_range(line_energy, width, calibration, channels,
                                             energy=grid)
            width = float(width)
            return [start, end], f"Integrating {line_energy:g} ± {width / 2:g} keV"
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as e:
            return no_update, str(e)

    @app.callback([Output("points-store", "data"), Output("msg", "children"),
                   Output("roi-store", "data", allow_duplicate=True),
                   Output("ch-store", "data", allow_duplicate=True)],
                  Input("apply-points", "n_clicks"), Input("clear", "n_clicks"),
                  State("pt-rows", "value"), State("pt-cols", "value"),
                  prevent_initial_call=True)
    def _set_points(n_pt, n_cl, rows_s, cols_s):
        c = state["cube"]
        if ctx.triggered_id == "clear":
            H, W, C = c.shape if c is not None else (0, 0, 0)
            return [], "", [0, H, 0, W], [0, C]
        if c is None:
            return no_update, "Load a cube first (upload or link)", no_update, no_update
        H, W, _ = c.shape
        try:
            pts = [[r, c_] for r, c_ in _parse_points(rows_s, cols_s)]
            bad = [p for p in pts if not (0 <= p[0] < H and 0 <= p[1] < W)]
            if bad:
                raise ValueError(f"point {bad[0]} out of bounds ({H}x{W})")
            return pts, "", no_update, no_update
        except ValueError as e:
            return no_update, str(e), no_update, no_update

    def _calibration_point_rows(points):
        return [dbc.Row(className="g-1 align-items-center mb-1", children=[
            dbc.Col(html.Span(f"x: {point['channel']}  |  y: {point['energy']:.17g} keV"), width=9),
            dbc.Col(dbc.Button("Remove", id={"type": "cal-point-remove", "index": i},
                               n_clicks=0, size="sm", outline=True, color="danger"), width=3),
        ]) for i, point in enumerate(points)]

    @app.callback([Output("cal-points-store", "data"), Output("cal-point-rows", "children"),
                   Output("cal-point-x", "value"), Output("cal-point-y", "value"),
                   Output("cal-point-msg", "children")],
                  Input("cal-point-add", "n_clicks"),
                  Input({"type": "cal-point-remove", "index": ALL}, "n_clicks"),
                  State("cal-points-store", "data"), State("cal-point-x", "value"),
                  State("cal-point-y", "value"), prevent_initial_call=True)
    def _edit_calibration_points(_add, _remove, points, channel, energy):
        points = list(points or [])
        trigger = ctx.triggered_id
        if trigger == "cal-point-add":
            try:
                point = _parse_calibration_point(channel, energy)
            except (TypeError, ValueError):
                return points, _calibration_point_rows(points), no_update, no_update, \
                    "Enter an integer channel and a finite energy"
            points.append(point)
            return points, _calibration_point_rows(points), None, "", ""
        if isinstance(trigger, dict) and trigger.get("type") == "cal-point-remove":
            points.pop(trigger["index"])
        return points, _calibration_point_rows(points), no_update, no_update, ""

    @app.callback(Output("cal-point-fit", "disabled"), Input("cal-points-store", "data"))
    def _calibration_fit_enabled(points):
        return len(points or []) < 2

    @app.callback([Output("cal-store", "data", allow_duplicate=True),
                   Output("cal-msg", "children", allow_duplicate=True),
                   Output("cal-point-msg", "children", allow_duplicate=True),
                   Output("apply-line", "disabled", allow_duplicate=True),
                   Output("m", "value"), Output("q", "value")],
                  Input("cal-point-fit", "n_clicks"), State("cal-points-store", "data"),
                  prevent_initial_call=True)
    def _fit_calibration(_n, points):
        try:
            m, q = _calibration_from_points(points or [])
        except (TypeError, ValueError, np.linalg.LinAlgError) as error:
            return no_update, no_update, str(error), no_update, no_update, no_update
        message = f"Applied fit from {len(points)} points: m = {m:.17g}, q = {q:.17g} keV"
        return {"m": m, "q": q}, message, message, False, format(m, ".17g"), format(q, ".17g")

    @app.callback([Output("cal-store", "data"), Output("cal-msg", "children"),
                   Output("apply-line", "disabled")],
                  Input("cal-input-store", "data"),
                  prevent_initial_call=True)
    def _calibration_msg(values):
        m, q = (values or {}).get("m"), (values or {}).get("q")
        if m is None or not str(m).strip():
            return no_update, "Enter m (keV/channel) to apply calibration", no_update
        try:
            m = float(str(m).strip().replace(",", "."))
            q = float(str(q).strip().replace(",", ".")) if q and str(q).strip() else 0.0
            if not math.isfinite(m) or m <= 0 or not math.isfinite(q):
                raise ValueError
        except (TypeError, ValueError):
            return no_update, "Enter finite m > 0 and q values", no_update
        return {"m": m, "q": q}, f"Applied: E = {m} × channel + {q} keV", False

    @app.callback(Output("show-energy", "value"),
                  Input("cal-store", "data"), Input("gen", "data"))
    def _xmode_toggle(calibration, _gen):
        """ADC/Energy swap (calibration check): checked while a calibration is
        active — manual or the file energy axis shown in the fields. dcc.Checklist
        has no `disabled` prop in dash 4.5, so the check state itself carries the
        'usable' state; _x_on stays a no-op without calibration."""
        if (calibration or {}).get("m") is not None or state["energy"] is not None:
            return ["energy"]
        return []

    @app.callback(Output("xmode-store", "data"), Input("show-energy", "value"))
    def _xmode_store(value):
        return bool(value)

    @app.callback(Output("line-selected", "children"), Input("line-select", "value"))
    def _show_selected_line(value):
        if value is None:
            return "No line selected"
        line = next((line for line in _XRF_LINES if line["value"] == value), None)
        return f"Selected: {line['element']} — {line['value']:g} keV" if line else ""

    @app.callback([Output("scale-store", "data"), Output("zmin", "value"),
                   Output("zmax", "value")],
                  Input("apply-scale", "n_clicks"), Input("reset-scale", "n_clicks"),
                  State("zmin", "value"), State("zmax", "value"),
                  prevent_initial_call=True)
    def _set_scale(_apply, _reset, zmin, zmax):
        if ctx.triggered_id == "reset-scale":
            return {"zmin": None, "zmax": None}, None, None
        return {"zmin": zmin, "zmax": zmax}, zmin, zmax

    # --- M6: file loading (upload / link / change) ---------------------------
    def _commit_load(arr, name, gen, cube=None):
        """One commit block for load and clear (audit 1, 19). The new dataset
        owns all selections, so no stale ROI, channels, points, scale or
        calibration state survives a load (audit 2). A file energy axis is
        shown in the m/q fields like a manual calibration (F4)."""
        h, w, c = arr.shape if arr is not None else (0, 0, 0)
        if arr is None:
            msg, cname = "Cube cleared — upload a file or paste a link.", "— no cube loaded"
        else:
            msg, cname = (f"Loaded {name} — {h} x {w} x {c}",
                          f"— {name} ({h} x {w} x {c})")
        file_m = file_q = None
        if cube is not None and getattr(cube, "energy", None) is not None:
            file_m, file_q = _fit_energy_grid(cube.energy)
        cal_msg = (f"File energy axis: m = {file_m:.17g}, q = {file_q:.17g} keV "
                   f"(linear fit) — Apply calibration to use it"
                   if file_m is not None else "")
        return (msg, cname, (gen or 0) + 1,
                None, None,
                None, [], 0,
                [0, h, 0, w], [0, c], [],
                {"zmin": None, "zmax": None},
                0, h, 0, w, 0, c,
                [], [], None, None, "", cal_msg,
                format(file_m, ".17g") if file_m is not None else None,
                format(file_q, ".17g") if file_q is not None else None)

    @app.callback([Output("load-msg", "children"), Output("cube-name", "children"),
                   Output("gen", "data"),
                   Output("zmin", "value", allow_duplicate=True),
                   Output("zmax", "value", allow_duplicate=True),
                   Output("cal-store", "data", allow_duplicate=True),
                   Output("elemental-meta", "data", allow_duplicate=True),
                   Output("elemental-index", "data", allow_duplicate=True),
                   Output("roi-store", "data", allow_duplicate=True),
                   Output("ch-store", "data", allow_duplicate=True),
                   Output("points-store", "data", allow_duplicate=True),
                   Output("scale-store", "data", allow_duplicate=True),
                   Output("roi-y0", "value"), Output("roi-y1", "value"),
                   Output("roi-x0", "value"), Output("roi-x1", "value"),
                   Output("ch-s", "value"), Output("ch-e", "value"),
                   Output("cal-points-store", "data", allow_duplicate=True),
                   Output("cal-point-rows", "children", allow_duplicate=True),
                   Output("cal-point-x", "value", allow_duplicate=True),
                   Output("cal-point-y", "value", allow_duplicate=True),
                   Output("cal-point-msg", "children", allow_duplicate=True),
                   Output("cal-msg", "children", allow_duplicate=True),
                   Output("m", "value", allow_duplicate=True),
                   Output("q", "value", allow_duplicate=True)],
                  Input("upload", "contents"), State("upload", "filename"),
                  Input("load-url", "n_clicks"), State("url", "value"),
                  Input("change-file", "n_clicks"),
                  State("gen", "data"),
                  prevent_initial_call=True)
    def _on_load(contents, filename, n_url, url, n_change, gen):
        """Single entry point for file loading (M6).

        Dispatch on ctx.triggered_id, not on the n_clicks counters: they are
        cumulative, so the first Change file discarded every later upload and
        a stale URL click re-downloaded over a new upload (audit 1). On
        success the dataset is committed in one block (audit 2, 19); on error
        the current dataset is untouched. The gen bump forces the figure
        callbacks to recompute.
        """
        if ctx.triggered_id == "change-file":
            with state_lock:
                state["cube"], state["name"] = None, ""
                state["energy"] = None
                state["revision"] += 1
                state["result"] = None
            return _commit_load(None, "", gen)
        if ctx.triggered_id == "load-url":
            if not (url and url.strip()):
                return ("Enter a URL first",) + (no_update,) * 25
            try:
                cube = download_cube(url.strip())
            except Exception as e:
                return (f"Download failed: {e}",) + (no_update,) * 25
            arr, name = cube.array, cube.path.name
        elif contents:
            try:
                _, b64 = contents.split(",", 1)
                cube = load_cube_bytes(base64.b64decode(b64), filename or "")
            except Exception as e:
                return (f"Upload failed: {e}",) + (no_update,) * 25
            arr, name = cube.array, Path(filename or "cube").name
        else:
            return (no_update,) * 26
        with state_lock:
            state["cube"], state["name"] = arr, name
            state["energy"] = cube.energy
            state["revision"] += 1
            state["result"] = None
        return _commit_load(arr, name, gen, cube)

    return app
