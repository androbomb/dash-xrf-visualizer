# XRF refactor — project timeline

Retrospective record of completed work and **observed evidence only**.
No repository audit, migration, notebook test, JAX benchmark or GPU test has
been performed as part of creating these templates. Do not invent entries.

## Entry template — append after work actually occurs

### YYYY-MM-DD — Milestone ID and outcome

- **Objective:**
- **Work:** paths/commit or diff summary.
- **Evidence:** actual checks, commands, input fixture and observed outcomes.
- **Decisions:** approved retained features and any resulting scope changes.
- **Limitations / blockers:** unknowns, skipped validations, unsolved issues.
- **Next:** only if necessary.

---

### 2026-11-10 — M0 read-only audit (completed)

- **Objective:** inventory of the legacy Dash app against the target XRF-only visualizer scope; report only.
- **Work:** read `AGENTS.md`, `operational_plan.md`, `to_do.md`, `timeline.md`, bootstrap README; legacy `dash-xrf-analyzer/` mapped read-only (entrypoint `run.py`, `application/dash_app.py` ~2657 lines, `utils/RendererXRF_class.py` 660 lines, calibration/AIRES/custom-raw modules, `requirements.txt` pins). At audit time the root was **not** a git repository; the user created it during M1. No writes, installs, servers or commits by the agent.
- **Evidence:** legacy = Python 3.8, dash 1.21, plotly 4.14.3, numpy 1.16, h5py 2.10, TF 2.3.1; ingestion via base64 upload only, key `img` (.npz/.h5), no .npy support; energy axis hardcoded 0.5–38.5 keV; map = plain channel sum, spectrum = plain pixel sum; zoom fused with ROI selection; latent bugs (keV→channel indexing at edges, ROI clamp copy-paste, display↔ROI transposition); AIRES model reloaded on every upload (26.6 MB model tracked in legacy git; `.pyc` and `variables.env` also tracked); no tests, no pyproject/lock, no XRF data files in repo. Feature table KEEP/REMOVE/DEFER delivered in the M0 report.
- **Decisions:** user approved (2026-11-10): minimal tree (plan §2), uv, full docs sync, zoom = selection + drag select + numeric panel + multi-point, plain sums without normalization, channel-axis default + manual m/q calibration for uncalibrated cubes; sample npz supplied in `data_example/` (git-ignored).
- **Limitations / blockers:** no real file was in the repo at audit start; GPU JAX path not yet exercised.
- **Next:** M1.

### 2026-11-10 — M1 baseline + approved docs (completed)

- **Objective:** uv project at root, deterministic fixture + golden values, baseline tests, docs sync, kernel registration.
- **Work:** `pyproject.toml` / `.python-version` / `uv.lock` at root (uv 0.13.7; Python 3.12.16; numpy 2.5.4, h5py 3.16.0, jax 0.12.0.1 CPU, dash 4.5.0, plotly 7.1.0, pytest 9.2.0); `tests/make_fixture.py` + `tests/fixtures/synthetic_cube.{npz,h5,npy}` + `golden.json`; `tests/test_baseline_reductions.py`, `tests/test_fixture_consistency.py`; docs sync (real `README.md` replacing bootstrap, `scientific_notes.md`, `operational_plan.md`, `to_do.md`); `deploy/uv/README.md`; kernel `xrf-viz` registered from the project venv. The one-line `AGENTS.md` status update is **pending user approval** (protected file; exact diff in the M1 report).
- **Evidence:** `pytest` → 5 passed in 0.89 s (CPU only); `import jax; jax.devices()` → `CpuDevice(0)` with `nvidia-smi` unchanged before/after (no CUDA context, no VRAM pre-allocation, RTX 5070 Ti visible); `dash.Dash.run(jupyter_mode=...)` verified present in dash 4.5.0; `data_example/` npz inspected via mmap metadata + small slices only (float32, (76,177,500) and (145,212,500), key `img`, no stored energy axis).
- **Decisions:** default environment is CPU-only JAX (no GPU context at import); GPU = opt-in `jax[cuda12]` variant, documented, NOT TESTED.
- **Limitations / blockers:** GPU path, containers and notebook inline mode not exercised; no push performed (remote `origin` exists; push is user-only).
- **Next:** stop for user review; then M2 (readers + source package).

### 2026-11-10 — M2: readers + source package (completed)

- **Objective:** data readers for .h5/.npz/.npy with schema validation; XRFlines migrated into the package.
- **Work:** `src/xrf_visualizer/data/__init__.py` (`XrfCube`, `load_cube`, `load_xrf_lines`; 85 lines); `src/xrf_visualizer/data/XRFlines.json` (copied from legacy `application/static/`, 347 CHNet lines); `tests/test_readers.py` (12 tests: round-trip across the three encodings, optional `energy` contract, 6 validation-error cases, dash-import gate in a fresh subprocess, XRFlines spot checks, real `data_example/` metadata check).
- **Evidence:** pytest → 17 passed in 1.87 s (CPU); grep AIRES/TF in `src/` → 0 hits; both real example npz load successfully (bounded, 27/61 MB).
- **Notes:** dash 4.5.0 auto-registers a pytest plugin (`dash.testing.plugin`), so pytest itself imports dash; the "data module never imports dash" gate therefore runs in a fresh subprocess. h5 `img` must be a Dataset (Groups rejected with a clear error).
- **Next:** stop for user review; then M3 (processing: backend policy + JAX reductions).

### 2026-11-10 — M3: processing (JAX float64 reductions) (completed)

- **Objective:** `src/xrf_visualizer/processing/` with a device policy and plain float64 reductions (map from channel interval, spectrum from ROI/points), calibration helpers (m/q), exact parity vs `golden.json`.
- **Work:** `processing/backend.py` (x64 forced; native JAX device selection — CPU in this venv, GPU follows `JAX_PLATFORM_NAME`; `current_device`, `backend_info`, `require_backend`); `processing/reductions.py` (`map_from_channels`, `spectrum_from_roi` with y=row/x=col, `spectrum_from_points`, `calibrate`, `energy_to_channel`, `channel_range_for_energy`; out-of-range inputs raise ValueError). Spec: `tests/test_processing.py` (8 tests, incl. the GPU bit-parity test marked NOT TESTED via skipif).
- **Evidence:** pytest → 24 passed, 1 skipped (GPU, not tested) in ~8 s; all 7 golden keys bit-exact vs JAX output (float32 accumulation differs by ~2e-7 relative, so parity catches dtype regressions); real-data smoke (Mantegna npz, 76×177×500 float32): load 0.06 s, full map 0.17 s, ROI spectrum 0.09 s, all float64.
- **Notes:** goldens were re-specified in this milestone to a *sequential* float64 reference (`tests/make_fixture.py` and `tests/test_baseline_reductions.py` updated, uncommitted at milestone start); `golden.json` values unchanged (sequential == pairwise for these shapes). Shortcut: XLA's reduction order is not strictly sequential — bit-exactness verified on CPU for fixture and real-data shapes; the GPU path may drift by a few ulp (NOT TESTED).
- **Next:** stop for user review; then M4 (Dash UI).

### 2026-11-11 — M4: Dash UI (completed)

- **Objective:** visual-only UI: 2-D map (left) + spectrum (right), zoom = selection, explicit selection modes, m/q calibration, notebook start.
- **Work:** `src/xrf_visualizer/ui/app.py` (`build_app(cube)`, `map_figure`, `spectrum_figure`, `relayoutData`→ROI/interval helpers, numeric ROI/point/channel panels, clamping before `processing/`); `src/xrf_visualizer/ui/__init__.py`; `tests/test_ui_app.py` (11 tests: build+ids, helper parsing, clamping, figures vs goldens incl. log mode); `example.ipynb` (`path_to_file` variable, kernel `xrf-viz`, `app.run(jupyter_mode=True)`).
- **Evidence:** pytest → 31 passed, 1 skipped (GPU, NOT TESTED) in ~18 s; live server on the real Mantegna cube (76×177×500 float32): HTTP 200, all 20 component ids in the layout; E2E through the real `/_dash-update-component` endpoint (dash 4.5 wire format): map figure bit-exact vs `map_from_channels`, spectrum figure bit-exact vs `spectrum_from_roi`; zoom→ROI `[19, 30, 10, 50]`; zoom→interval `[100, 120]`; valid points stored + message cleared; invalid points → error message, store unchanged.
- **Notes:** dash 4.5 breaking change discovered empirically — callbacks must return a **list** of values even for a single `Output` (scalar return → HTTP 500); single-output key in `callback_map` is `id.property`, multi-output `..a...b..`; plotly serializes figure arrays binary (`bdata`/`dtype`/`shape`), so E2E checks decode them. Pan is not distinguishable from zoom in the `relayoutData` signal, so panning also moves the selection (documented limitation). Browser-level click testing not possible in this environment (no Chromium) — HTTP-level E2E used instead.
- **Next:** M5 (AIRES/TF absence gate, deploy recipes marked NOT TESTED, final README pass).

### 2026-11-11 — M5: absence gate + deployment recipes (completed, recipes NOT TESTED)

- **Objective:** verify no AIRES/TF/PyTorch residue in the new runtime; provide Docker/WSLC/Apptainer recipes marked NOT TESTED; final README pass.
- **Work:** `deploy/docker/` (Dockerfile + README), `deploy/wslc/` (Containerfile + README), `deploy/apptainer/` (xrf-viz.def + README) — all based on `ghcr.io/astral-sh/uv:python3.12-bookworm-slim`, `uv sync --frozen --no-dev`, CPU-only JAX, cube via `XRF_CUBE` env + read-only `/data` bind, port 8765; README final pass (M4/M5 status).
- **Evidence:** `grep -ri "aires|tensorflow|keras" src/ pyproject.toml` → 0 hits (runtime is AIRES/TF-free); `pytest` → 31 passed, 1 skipped (GPU, NOT TESTED).
- **Limitations / blockers:** no Docker, WSLC or Apptainer in this environment — recipes are documentation deliverables and **NOT TESTED**; user will run them. WSLC/Apptainer flag names follow the standard docker form and may need adjustment per CLI version (noted in the READMEs).
- **Deferred (user decision):** drag-mode select on the map (tracked in `to_do.md` → Later).
- **Next:** user-side container tests + GPU validation on WSL2 (RTX 5070 Ti); then optional legacy `dash-xrf-analyzer/` removal decision.

### 2026-11-11 — M5 fix: wslc build failure found by user (README.md missing in build context)

- **Observation:** user ran `wslc build --tag xrf-viz --file deploy/wslc/Containerfile .` → step 4 `uv sync --frozen --no-dev` failed: hatchling `OSError: Readme file does not exist: README.md`. Root cause: `pyproject.toml` declares `readme = "README.md"`, but the recipe copied only `pyproject.toml uv.lock` before `uv sync` (the editable project build validates metadata). Same bug in `deploy/docker/Dockerfile`.
- **Fix:** `COPY pyproject.toml uv.lock README.md ./` in both recipes; how-to notes added (harmless `git was not found` warning during build).
- **Evidence:** user-side first build attempt (WSLC), failure at the uv step; fix applied in this commit — rebuild pending on the user side.

### 2026-11-11 — M6: empty start + file loading (upload / link) (completed)

- **Objective:** app openable without `XRF_CUBE`; the user loads `.npz`/`.h5`/`.npy` by upload or by http(s) link (ownCloud/nextCloud share, Zenodo download, ...); containers start empty when the env var is unset.
- **Work:** `data/__init__.py`: `sniff_format` (magic bytes `.npy`/`.npz`/HDF5), `load_cube_bytes` (extension or magic sniff → temp file → `load_cube`, full schema validation), `download_cube` (http/https only, filename from Content-Disposition or URL, 500 MB cap, 60 s timeout). `ui/app.py`: `build_app(cube=None)` empty state; `dcc.Upload` + URL input + "Load from link" + "Change file"; single `_on_load` callback (priority change > link > upload; errors keep the current cube; a `gen` store bump re-triggers the figure callbacks; the cube stays in the Python closure, never in a dash Store). Container CMDs (Docker/WSLC/Apptainer): `XRF_CUBE` optional (`os.environ.get(...) or None`). `tests/test_transfer.py` (new: sniffing, bytes-vs-file equality, wrong-extension and non-http rejections, download round-trip over a local HTTP server, oversized and invalid content) and M6 tests in `tests/test_ui_app.py` (empty build, upload → figures, bad bytes, link, change).
- **Evidence:** pytest → 49 passed, 1 skipped (GPU, NOT TESTED) in ~22 s. Empty-app E2E over real HTTP (`build_app(None)` server, port 8766): GET / 200; upload callback → "Loaded synthetic_cube.npz — 48 x 64 x 512" + gen bump; map figure bit-exact vs `map_from_channels` after load; bad link → "Download failed: <urlopen error…>" with state untouched; "Change file" → cleared.
- **Notes:** links must be direct http(s) downloads; share links that return an HTML page fail validation with a clear error (documented). Upload size is browser-limited (base64 in the payload); the 500 MB cap applies to link downloads. `example.ipynb` intentionally unchanged (user decision). Dash 4.5 test pitfall: `callback_map` entries are wrappers requiring `outputs_list` — tests call the raw function via `__wrapped__`.
- **Next:** user-side container rebuilds (WSLC rebuild after the README fix still pending; Docker/Apptainer not started) + visual browser check.

### 2026-11-11 — M7: WSLC runtime 500 fixed + legacy dbc look ported (completed)

- **Observation:** user WSLC test (image built from `d729bf3`): the app started and the cube WAS loaded (the crash happened after the `if c is None` guard), but every zoom triggered `POST /_dash-update-component` 500s — `IndexError: list index out of range` in `_spec_figure` (`roi[1]`) and `ValueError: not enough values to unpack (expected 2, got 1)` in `_map_figure` (`s, e = ch`) — and the UI looked empty ("non ha caricato il file" = the graph never rendered a figure). User also requested: keep the legacy `dash_bootstrap_components` HTML look of the original app.
- **Root cause (verified in dash 4.5.0 source + renderer bundle):** for a callback with a single `Output`, `dash_renderer` v4.5.0 sends the request body `outputs` as a **single object** (`outputs: isMultiOutputProp(output) ? outputs : outputs[0]` in `dash_renderer.dev.js`); the server then assigns the callback return value **as-is** to that one output (`_prepare_response`, single path: `output_value, output_spec = [output_value], [output_spec]` with a bare spec dict). The M4/M6 code returned `[value]` for single-Output callbacks (the pre-4.5 convention), so the stores received `[[y0,y1,x0,x1]]` (1-element list → `roi[1]` IndexError) and the graphs received `[fig]` (not a figure dict → nothing rendered). The bug only surfaces on real HTTP dispatch: the pytest/E2E calls invoked the raw functions via `__wrapped__`, bypassing `_prepare_response`.
- **Fix:** `src/xrf_visualizer/ui/app.py` — single-Output callbacks (`map.figure`, `spectrum.figure`, `roi-store.data`, `ch-store.data`) now return the **raw value**; multi-Output callbacks unchanged (tuple). All original callbacks preserved (`_on_load` with priority change > link > upload, `_set_points`, ...).
- **M7 look (user request):** legacy look ported from read-only `dash-xrf-analyzer/application/dash_app.py` — `dash-bootstrap-components` added to `pyproject.toml`/`uv.lock` (2.0.4), `dbc.themes.DARKLY`, `#4196B4` navbar, fluid container, 5/7 columns with controls grouped per graph (display+interval under the map; ROI+points+calibration under the spectrum).
- **Evidence:** pytest → 49 passed, 1 skipped (GPU, NOT TESTED) in ~24 s. New HTTP E2E replicating the exact WSLC scenario with the **real renderer body format** (single-object `outputs` for single-Output callbacks): preloaded real cube (76x177x500) → initial map/spectrum 200, figure is a dict, bit-exact vs `map_from_channels`/`spectrum_from_roi`; map zoom `[10.2,39.8]x[19.5,9.4]` → `roi-store` FLAT `[9,20,10,40]` (the step that 500'd in the container) → post-zoom spectrum bit-exact; spectrum zoom → `ch-store` FLAT `[100,120]` → post-zoom map bit-exact; double-click → ROI reset `[0,76,0,177]`; empty-state app: spurious relayouts → 200, figures are dicts, no 500.
- **Limitations:** visual browser check still user-side (no Chromium in Hermes); container images must be **rebuilt** (new dependency + fixed code) — commands unchanged (`wslc build` then `wslc run -p 8765:8765 ...`).
- **Next:** user rebuilds the WSLC image and retests (preload + upload + zoom); Docker/Apptainer rebuilds when the user gets to them.

### 2026-11-12 — M8: selection labels and control layout (completed)

- **UI:** compact horizontal map controls; channel fields renamed “ch begin”/“ch end”; ROI inputs arranged x0/x1 then y0/y1; rows/cols and m/q aligned with wider inputs. Added an “Apply calibration” button and concise formula label.
- **Figure context:** map title states the integrated ADC channel range or its calibrated keV range; spectrum title reports the active ROI box or selected points. Calibration remains a plot-axis rescaling (`E = m·channel + q`) and does not change XRF samples or reduction results.
- **Verification:** `uv run pytest -q` → 49 passed, 1 skipped (GPU); added checks for map interval/unit titles and spectrum ROI/point titles.
- **Next:** rebuild and visually inspect the WSLC container as part of the user's existing deployment workflow.

### 2026-11-12 — M9: Dash callback triggers and notebook mode (completed)

- **Root causes:** explicit Apply ROI/interval callbacks preferred non-empty stale `relayoutData`; Dash callback declarations had State values interleaved before later Inputs, obscuring/incorrectly aligning callback parameters; the example passed boolean `jupyter_mode=True`, rejected by Dash 4.5 (requires string mode).
- **Fix:** branch ROI/channel changes on `ctx.triggered_id`; keep every Input before State and callback arguments in Dash's input-then-state order; include zero as a valid zmin/zmax; update notebook/README to `jupyter_mode="inline"`.
- **Evidence:** HTTP callback tests verify graph zoom→opposite figure update, numeric apply overrides old zoom, zmin=0/zmax apply, and calibration refreshes both plotted axes; `uv run pytest -q` → 52 passed, 1 skipped (GPU).

### 2026-11-12 — M10: numeric actions update both figures (completed)

- **Behavior:** Apply ROI now sets the map's x/y ranges and recomputes the spectrum; Apply interval sets the spectrum's channel/energy x range and recomputes the map. Added Apply scale for an explicit display refresh; zmin/zmax remain live Inputs as well. Calibrated channel zoom bounds are mapped to energy coordinates.
- **Notebook compatibility:** `_XrfDash.run` maps legacy `jupyter_mode=True` to Dash 4's required string mode `"inline"`; the example already uses the string form. This means the old notebook cell can work after updating the package in `.windows-env`.
- **Evidence:** `uv run pytest -q` → 53 passed, 1 skipped (GPU); HTTP callback assertions cover both axes, map/spectrum refresh, scale values including zero, calibration and legacy boolean mode.
