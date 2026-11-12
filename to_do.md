# XRF-only visualizer — project to-do

> Future requirements are user decisions, not verified legacy implementation.
> M0–M11 completed 2026-11-10/12 (evidence in `timeline.md`).

## Now — M1 baseline + approved docs (completed 2026-11-10)

- [x] uv project at root: `pyproject.toml`, `uv.lock`, `.python-version` (3.12); CPU-only JAX default (no GPU pre-allocation — verified with `nvidia-smi`).
- [x] Deterministic synthetic fixture (48×64×512 float32) as .npz/.h5/.npy + `golden.json`; generator `tests/make_fixture.py`.
- [x] Baseline tests: golden reductions (float64 rule), interval/ROI edge cases, encoding consistency — 5 passed in <1 s.
- [x] Docs sync: `scientific_notes.md` (verified conventions), `operational_plan.md` (M0 findings), this file, `timeline.md`, `AGENTS.md`, real `README.md` (bootstrap removed).
- [x] Jupyter kernel registered from the project venv (`xrf-viz`); instructions in `deploy/uv/README.md`.
- [x] Gate: user review of M1 → approve M2 (approved 2026-11-10).

## Done — M2: readers + source package (completed 2026-11-10)

- [x] `src/xrf_visualizer/data/`: readers for `.h5` (key `img`), `.npz` (`img` + optional `energy`), `.npy` (channel axis only); schema validation (ndim=3, dtype, shape) with informative errors; readers never import Dash.
- [x] Migrate `XRFlines.json` (347 CHNet lines, legacy `application/static/`) into the package as static data.
- [x] `tests/test_readers.py`: fixture round-trips + schema validation + bounded metadata check of `data_example/` files.
- [x] Gate: reader tests green (17/17), no AIRES coupling (grep clean).

## After M3 (approval-gated)

- [x] M3 — `processing/`: `backend.py` (x64 forced, native JAX device policy: CPU in this venv, GPU via `JAX_PLATFORM_NAME`, `backend_info`/`require_backend`) + `reductions.py` (`map_from_channels`, `spectrum_from_roi`, `spectrum_from_points`, `calibrate`, `energy_to_channel`, `channel_range_for_energy`; out-of-range raises ValueError); JAX parity vs `golden.json` **bit-exact** on all 7 keys; GPU agreement test marked NOT TESTED (skips here).
- [x] M4 — `ui/`: map left + spectrum right; zoom = selection via `relayoutData` (image zoom → spectrum, spectrum zoom → map, autorange = reset); explicit selection: numeric ROI panel, multi-point (rows/cols), numeric channel interval; zmin/zmax, colorscale, log; manual m/q calibration (keV axis when m set, else ADC); `example.ipynb` with `path_to_file`. Evidence: 31 passed + 1 GPU skip; live-server E2E (real Mantegna cube): figures bit-exact vs `processing/`, zoom→ROI `[19,30,10,50]`, zoom→interval `[100,120]`, point validation + error msg. Dash 4.5 single-Output callbacks return the raw value; multi-Output callbacks return tuples.
- [x] M5 — AIRES/TF absence gate: `grep -ri "aires|tensorflow|keras" src/ pyproject.toml` → 0 hits; `deploy/docker/` (Dockerfile + README), `deploy/wslc/` (Containerfile + README), `deploy/apptainer/` (xrf-viz.def + README) all marked NOT TESTED (CPU-only JAX image, `XRF_CUBE` env + `/data` bind, port 8765); final README pass.
- [x] M6 — Empty start + file loading: `build_app(cube=None)`; upload (`.npz`/`.h5`/`.npy`) or http(s) link in the UI (500 MB cap, magic-byte sniffing); `sniff_format`/`load_cube_bytes`/`download_cube` in `data/`; `XRF_CUBE` now optional in all containers (empty app when unset); markdown synced (README, how_to_deploy_*, container READMEs, plan, timeline). Evidence: 49 passed + 1 GPU skip; empty-app E2E over HTTP (upload → map bit-exact vs `map_from_channels`, bad link → clear error + state untouched, change-file → cleared).
- [x] M7 — UI fix + legacy look: (a) dash 4.5.0 return convention fixed — with a single `Output` the renderer sends `outputs` as a single object and the callback value is assigned as-is, so callbacks return the **raw value** (the previous `[value]` wrap double-wrapped the stores: `[[...]]` → 500 in `_spec_figure`/`_map_figure` on real WSLC traffic, figures came back as `[fig]` = "file not loaded"); (b) legacy look ported per user request — `dash-bootstrap-components` (DARKLY theme, `#4196B4` navbar, fluid container, 5/7 columns) added as a dependency. Evidence: 49 passed + 1 GPU skip; HTTP E2E using the real renderer body format (single-object `outputs`): initial figures 200 + DICT + bit-exact, zoom → flat stores (`[9,20,10,40]`, `[100,120]`), post-zoom figures bit-exact, double-click reset, empty state no 500.
- [x] M8 — UI selection and calibration clarity: aligned map controls, 2×2 ROI fields, wider rows/cols and m/q, explicit “Apply calibration” button; figure titles show active map interval (ADC or calibrated keV) and spectrum ROI/points. Calibration remains plot-axis-only. Evidence: 49 passed + 1 GPU skip; title/calibration assertions added.
- [x] M9 — Callback behavior + Dash 4 notebook compatibility: route ROI/channel changes by the actual triggering component (not stale `relayoutData`), preserve live figure refresh for zoom/selections and display controls, use `jupyter_mode="inline"`. Evidence: HTTP callback tests cover zoom→ROI→spectrum, explicit ROI override, spectrum zoom→map, explicit interval override, zmin/zmax including zero, and Apply calibration.
- [x] M10 — Follow-up UI action behavior: Apply ROI sets map axes and spectrum ROI; Apply interval sets spectrum axes and map interval; Apply scale explicitly refreshes color limits; legacy `jupyter_mode=True` is normalized to Dash 4's `"inline"`. Evidence: HTTP callback tests plus mode-normalization test.
- [x] M11 — Plotly interactions and shared display behavior: accept actual `axis.range[n]` relayout payloads; overlay selected points on the map; apply log1p to map and spectrum counts; zmin/zmax are committed by Apply scale while colormap/log remain live. Evidence: HTTP callback test with real relayout keys and bit-exact map/spectrum data assertions.

## Later — not executable in Hermes' current environment

- [ ] Drag-mode select on the map (deferred by user 2026-11-11): plotly drag-mode selection in addition to zoom = selection; zoom + numeric ROI panel cover the need for now.
- [ ] Ask user to manually test Docker, WSLC and Apptainer; document real test results (do not claim they passed).
- [ ] GPU validation on the user's WSL2 (RTX 5070 Ti / sm_120, cu12): `jax[cuda12]` variant + VRAM behavior with the documented XLA flags.
- [ ] Review removal of legacy `dash-xrf-analyzer/` **only after explicit final user approval**.

## Resolved (2026-11-10, user decisions)

- Data schema: legacy `img` key, (H, W, C); examples float32, C=500, no stored energy axis → channel axis by default, manual `E(i) = m·i + q` calibration in UI.
- Integration/aggregation: plain sums, no normalization (map over channels, spectrum over pixels).
- ROI interaction: zoom = selection (legacy semantics) + drag select + numeric panel + multi-point.
- `.npz` may carry an optional `energy` vector; `.npy` channel axis only.
- AIRES / model / inference / segmentation UI: removed (not re-asked). Custom raw reader: DEFER.
