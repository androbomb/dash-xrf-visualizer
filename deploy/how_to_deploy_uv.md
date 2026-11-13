# How to deploy — local uv + Jupyter

**Status:** VERIFIED 2026-11-11 (WSL2, CPU-only JAX, RTX 5070 Ti present but unused); UI callback suite last verified 2026-11-13.
GPU variant is in `uv/README.md` and remains NOT TESTED.

## Steps

1. From the repository root:

   ```bash
   uv sync
   .venv/bin/python -m pytest -q
   ```

   Latest recorded result: `57 passed, 1 skipped` (GPU parity, not tested).

2. Register the Jupyter kernel (once):

   ```bash
   .venv/bin/python -m ipykernel install --user --name xrf-viz \
       --display-name "Python 3.12 (xrf-viz)"
   ```

3. Open Jupyter/JupyterLab, open `example.ipynb` with kernel
   **Python 3.12 (xrf-viz)**, set `path_to_file = "/path/to/cube.npz"` (or
   `.h5` / `.npy`) and Run All. The app renders inline in the notebook. To
   avoid a scrollbar inside the embedded app, set `jupyter_height` in pixels,
   e.g. `app.run(jupyter_mode="inline", debug=False, jupyter_height=1000)`.

4. Alternatively, standalone server (preloaded cube):

   ```bash
   uv run python -c "from xrf_visualizer.data import load_cube; from xrf_visualizer.ui import build_app; build_app(load_cube('path/to/cube.npz').array).run(host='127.0.0.1', port=8765)"
   ```

   Or start empty (M6): `build_app(None)` — the page opens with
   "no cube loaded" and you upload a `.npz`/`.h5`/`.npy` or paste a
   download link in the UI.

## Checklist (same for every deploy mode)

- [ ] Page loads at `http://localhost:8765` (map left, spectrum right).
- [ ] Upload a `.npz`/`.h5`/`.npy` (or paste a link, or "Change file") → loading works (M6).
- [ ] Zoom the map → spectrum updates to the zoomed region.
- [ ] Zoom the spectrum → map updates to the zoomed channel interval.
- [ ] `autorange` (double-click / reset) → back to full field/interval.
- [ ] Numeric ROI panel → Apply updates the spectrum.
- [ ] Multi-point rows/cols → Apply works; a non-numeric entry shows an error message and keeps the previous selection.
- [ ] Set `m` (and `q`) → spectrum x-axis switches to keV.
- [ ] After applying calibration, zoom the spectrum → matching channel interval updates the map.
- [ ] Select an XRF line and width → Apply selects the centered energy window; Reset scale clears zmin/zmax.

## Result log (fill in after your test)

| date | pytest | notebook inline | standalone | notes |
| ---- | ------ | --------------- | ---------- | ----- |
|      |        |                 |            |       |
