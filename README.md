# xrf-visualizer

Plotly Dash visualizer for XRF measurement cubes: a 2-D intensity map on the
left, an XRF spectrum on the right. Selecting a spectral interval recomputes
the map (plain sum over channels); selecting a spatial region recomputes the
spectrum (plain sum over pixels). Runs inside Jupyter/JupyterLab from a
notebook: `path_to_file = "/path/to/file"` — no hardcoded paths.

This is a **rewrite** of the legacy `dash-xrf-analyzer/` app (kept read-only,
git-ignored, never imported by the new runtime). The AIRES model/inference
parts were removed by decision; the custom CHNet raw reader is deferred.

## Status

- M0 (read-only audit), M1 (uv environment, fixture + golden tests, docs),
  M2 (readers, `XRFlines`), M3 (JAX float64 reductions, bit-exact vs
  goldens) completed 2026-11-10; M4 (Dash UI) completed 2026-11-11 — see
  `timeline.md`.
- M5 (deployment recipes) completed 2026-11-11: WSLC deployment reported
  working by the user; Docker and Apptainer recipes remain **NOT TESTED**.
- M6 (empty start + file loading) completed 2026-11-11: `build_app(None)`
  starts empty; upload (.npz/.h5/.npy) or http(s) link in the UI;
  `XRF_CUBE` optional in all containers.
  Scope and acceptance criteria: `operational_plan.md`; verified numerical
  conventions: `scientific_notes.md`.
- M7 (UI fix + legacy look) completed 2026-11-11: fixed the dash 4.5
  callback return convention (single `Output` → raw value; the `[value]`
  wrap double-wrapped the stores and 500'd on real WSLC traffic) and
  ported the legacy `dash-bootstrap-components` look (DARKLY theme,
  `#4196B4` navbar, 5/7 columns).
- M8 (selection labels + controls layout) completed 2026-11-12: compact
  horizontal map controls, square ROI inputs, wider point/calibration fields,
  explicit calibration button, and plot titles showing integrated channel/energy
  range and spectrum selection. Calibration changes only the plotted X axis.
- M9 (callback behavior + notebook compatibility) completed 2026-11-12: explicit
  selection buttons now take precedence over stale graph zoom state; HTTP
  callback tests cover ROI/interval propagation, color limits and calibration.
  Dash 4.x notebooks use `jupyter_mode="inline"` (not boolean `True`).
- M10 (selection zoom + controls) completed 2026-11-12: Apply ROI/interval also
  sets the corresponding plot zoom; Apply scale explicitly refreshes display
  limits; legacy `jupyter_mode=True` is accepted as an alias for `"inline"`.
- M11 (Plotly interactions) completed 2026-11-12: parse Plotly's actual
  `xaxis.range[0]` relayout keys, show selected points on the map, and apply
  log1p to spectrum counts as well as map counts.
- M12 (scale, clear, calibration) completed 2026-11-12: zmin/zmax apply to
  Plotly's shared color axis; Clear selection resets points, ROI and channels;
  calibration confirms and displays the active m/q values. In log mode, the
  raw-count zmin/zmax are transformed with the map using `log1p`.
- M13 (calibration refresh, ROI zoom, scale reset) completed 2026-11-13:
  Apply ROI disables `imshow` autorange so both numeric axes zoom to the
  requested box; Reset scale clears zmin/zmax.
- M14 (explicit calibration, scale reset, line integration) completed 2026-11-13:
  Apply buttons commit settings to figure state; searchable JSON XRF lines can
  be integrated over a user-set width centered on the selected line energy.
- M15 (calibrated spectrum selection) completed 2026-11-13: spectrum zoom ranges
  are converted from keV back to ADC channels; line Apply stays disabled until
  calibration is applied and the selected line is shown below its menu.
- M17 (square map pixels and arbitrary ROI) completed 2026-11-13: rectangular
  drag selection accepts any shape, and the responsive map frame follows the
  selected ROI ratio while keeping square pixels.
- M18 (download map PNG) completed 2026-11-13: Download current map exports
  the selected interval and visible spatial ROI as one exact-size PNG pixel
  per XRF pixel, with the chosen colorscale/limits/log transform and no axes
  or colorbar.

## Setup

```bash
uv sync
.venv/bin/python -m pytest -q             # latest: 60 passed + 1 GPU skip
.venv/bin/python -m ipykernel install --user --name xrf-viz \
    --display-name "Python 3.12 (xrf-viz)"
```

Requires [uv](https://docs.astral.sh/uv/); Python 3.12 is managed by uv via
`.python-version`. In Jupyter/JupyterLab select the kernel
**Python 3.12 (xrf-viz)**.

Dash's inline notebook frame defaults to 650 px tall. Increase it to avoid an
inner vertical scrollbar when the UI needs more space, for example:

```python
app.run(jupyter_mode="inline", debug=False, jupyter_height=1000)
```

`jupyter_height` is an integer number of pixels; raise it further if the
notebook still scrolls inside the embedded app.

If a notebook keeps showing old UI behavior after updating the checkout, check
which module the kernel imported:

```python
import xrf_visualizer.ui.app as ui_module
print(ui_module.__file__)
```

The path should resolve to this checkout's `src/xrf_visualizer/ui/app.py`.
If it points to another checkout or `site-packages`, sync that environment and
restart the kernel before rebuilding the app. Updating the file on disk does
not replace callbacks already captured by an existing `app = build_app(cube)`;
restart the kernel and recreate `app` after a code update.

## Data formats

- `.h5` / `.npz`: dataset/key `img`, 3-D array, shape (H, W, C), channels last;
  `.npz` may carry an optional `energy` vector (float64, keV).
- `.npy`: single 3-D array, channel (ADC) axis only.
- Uncalibrated files display the channel axis; the UI provides manual
  calibration `E(i) = m·i + q` (m in keV/channel, q in keV); m and q accept
  arbitrary decimal precision in plain text fields; decimal commas and points
  are accepted.
- Examples (real data, git-ignored): `data_example/`.

## Compute

Default: **CPU-only JAX with float64 reductions**. No GPU context is created
at import and no VRAM is pre-allocated (verified with `nvidia-smi` on a GPU
host, 2026-11-10). GPU is an opt-in, NOT TESTED variant —
`deploy/uv/README.md`.

## Layout / interactions (M4, implemented 2026-11-11)

- **Look (M7):** `dash-bootstrap-components` — DARKLY theme, `#4196B4`
  navbar, fluid container, 5/7 column split (map | controls, spectrum |
  controls), outline info buttons — ported from the legacy
  `dash-xrf-analyzer` app.
- Left: 2-D map (heatmap); zmin/zmax, colorscale, log controls.
- Right: spectrum (keV axis when `m` is set, otherwise channel (ADC) vs
  sum counts).
- **Selection:** rectangle-drag on the map updates the spectrum and crops the
  map view; the map frame follows the ROI aspect ratio. Zooming the spectrum
  updates the map interval. Double-click/autorange resets to the full field or
  spectrum range.
- Explicit selection: numeric ROI panel (y0, y1, x0, x1), multi-point
  (rows + cols), numeric channel interval. Manual calibration panel
  (`m`, `q`); invalid points report an error message and keep the previous
  selection. All selections are clamped before reaching `processing/`.
- M8: map controls share a horizontal row; ROI inputs use a 2×2 layout
  (x0/x1, then y0/y1); point and calibration fields are wider. Plot titles
  report the integrated interval (ADC or calibrated keV) and active spectrum
  box/points. “Apply calibration” rescales only the plotted X axis; XRF arrays
  and reductions are unchanged.
- M10: numeric ROI/channel actions zoom the associated plot as well as update
  the opposite plot; Apply scale is available for explicit color-limit refresh.
  The Dash wrapper accepts the legacy boolean notebook mode for compatibility.
- M11: Plotly zoom relayout payloads use `axis.range[n]`; selected map points
  are overlaid on the map, and the log checkbox transforms both plotted sums.
- M12: Apply scale commits `coloraxis.cmin/cmax`; Clear selection restores both
  plots to full bounds; calibration shows the active formula on the spectrum.
- M13: numeric ROI disables Plotly's reversed autorange and
  explicitly zooms both axes to x0..x1 and y0..y1.
- M14: Apply calibration commits m/q to both plots; Apply scale commits zmin/zmax,
  while Reset scale clears the committed limits and their input fields. The
  searchable Select Line menu reads `XRFlines.json`; Apply integrates the
  selected keV width centered on that line (requires an applied calibration).
- M15: when calibrated, spectrum selections are inverted through `E=m·channel+q`
  before updating the map's ADC interval. The line-selection Apply button is
  enabled by successful calibration; the selected line is displayed explicitly.
- M17: rectangle-drag accepts arbitrary ROI aspect ratios; the map frame's CSS
  aspect ratio follows the selected ROI while Plotly keeps x/y scales equal.
- M18: Download current map writes the current channel sum and visible spatial
  crop as a lossless PNG with one square image pixel per matrix element. The
  chosen colorscale, zmin/zmax and log transform are applied; axes/colorbar and
  point markers are not included.
- M20: map double-click resets the ROI in notebook and browser; calibration
  inputs accept arbitrary decimal precision.
- `example.ipynb`: set `path_to_file`, then `build_app(cube)` and
  `app.run(jupyter_mode="inline", jupyter_height=1000)` (Dash 4.x); the height
  is in pixels. `app.run()` also works from a terminal.

## File loading (M6, implemented 2026-11-11)

- `build_app(cube=None)` starts empty: both panels show "no cube loaded"
  until a file is loaded.
- Upload: drag & drop or click, `.npz` / `.h5` / `.npy` — validated with
  the same readers as `load_cube`. Unknown extensions are rejected;
  nameless files are recognized by magic bytes.
- Link: paste an http(s) direct download link (Zenodo, ownCloud/nextCloud
  share links, ...) and press "Load from link" — the app downloads (500 MB
  cap, 60 s timeout) and validates it the same way. Links that return HTML
  (a share page instead of the file) fail with a clear error.
- "Change file" clears the cube back to the empty state; on load failure
  the current cube is untouched.
- Containers: `XRF_CUBE` is now **optional** in Docker/WSLC/Apptainer —
  unset (and no `/data` bind) the app starts empty with the same
  upload/link UI.

## Repository map

```text
pyproject.toml / uv.lock / .python-version   uv project (root, versioned)
src/xrf_visualizer/                          data / processing / ui (M2–M4)
tests/                                       pytest; fixtures/ + golden.json
deploy/{docker,wslc,apptainer,uv}/           recipes & how-tos (Docker/Apptainer NOT TESTED)
dash-xrf-analyzer/                           legacy reference (git-ignored, read-only)
```
