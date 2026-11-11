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
