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
  M2 (readers, `XRFlines`) and M3 (JAX float64 reductions, bit-exact vs
  goldens) completed 2026-11-10 — see `timeline.md`.
- **Not implemented yet:** Dash UI (M4) and deployment recipes (M5).
  Scope and acceptance criteria: `operational_plan.md`; verified numerical
  conventions: `scientific_notes.md`.

## Setup (verified 2026-11-10)

```bash
uv sync
.venv/bin/python -m pytest -q          # 5 tests, ~1 s, CPU only
.venv/bin/python -m ipykernel install --user --name xrf-viz \
    --display-name "Python 3.12 (xrf-viz)"
```

Requires [uv](https://docs.astral.sh/uv/); Python 3.12 is managed by uv via
`.python-version`. In Jupyter/JupyterLab select the kernel
**Python 3.12 (xrf-viz)**.

## Data formats

- `.h5` / `.npz`: dataset/key `img`, 3-D array, shape (H, W, C), channels last;
  `.npz` may carry an optional `energy` vector (float64, keV).
- `.npy`: single 3-D array, channel (ADC) axis only.
- Uncalibrated files display the channel axis; the UI provides manual
  calibration `E(i) = m·i + q` (m in keV/channel, q in keV).
- Examples (real data, git-ignored): `data_example/`.

## Compute

Default: **CPU-only JAX with float64 reductions**. No GPU context is created
at import and no VRAM is pre-allocated (verified with `nvidia-smi` on a GPU
host, 2026-11-10). GPU is an opt-in, NOT TESTED variant —
`deploy/uv/README.md`.

## Layout / interactions (target, M4)

- Left: 2-D map (`px.imshow`); zmin/zmax, colorscale, lin/log controls.
- Right: spectrum (keV or channel vs counts) with rangeslider.
- **Zoom = selection** (legacy semantics, confirmed by user): image zoom →
  new spectrum; spectrum zoom → new map; `autorange` resets to full field.
- Extra selection modes: drag-mode select, numeric ROI panel, multi-point.

## Repository map

```text
pyproject.toml / uv.lock / .python-version   uv project (root, versioned)
src/xrf_visualizer/                          data / processing / ui (M2–M4)
tests/                                       pytest; fixtures/ + golden.json
deploy/{docker,wslc,apptainer,uv}/           recipes & how-tos (M5; containers NOT TESTED)
dash-xrf-analyzer/                           legacy reference (git-ignored, read-only)
```