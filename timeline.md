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
