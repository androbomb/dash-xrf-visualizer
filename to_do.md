# XRF-only visualizer — project to-do

> Future requirements are user decisions, not verified legacy implementation.
> M0, M1, M2 and M3 completed 2026-11-10; M4 completed 2026-11-11.

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
- [x] M4 — `ui/`: map left + spectrum right; zoom = selection via `relayoutData` (image zoom → spectrum, spectrum zoom → map, autorange = reset); explicit selection: numeric ROI panel, multi-point (rows/cols), numeric channel interval; zmin/zmax, colorscale, log; manual m/q calibration (keV axis when m set, else ADC); `example.ipynb` with `path_to_file`. Evidence: 31 passed + 1 GPU skip; live-server E2E (real Mantegna cube): figures bit-exact vs `processing/`, zoom→ROI `[19,30,10,50]`, zoom→interval `[100,120]`, point validation + error msg. dash 4.5 note: callbacks must return a list of values even for a single Output.
- [ ] M5 — AIRES/TF absence gate (grep), `deploy/{docker,wslc,apptainer}/` recipes marked NOT TESTED, final README pass.

## Later — not executable in Hermes' current environment

- [ ] Ask user to manually test Docker, WSLC and Apptainer; document real test results (do not claim they passed).
- [ ] GPU validation on the user's WSL2 (RTX 5070 Ti / sm_120, cu12): `jax[cuda12]` variant + VRAM behavior with the documented XLA flags.
- [ ] Review removal of legacy `dash-xrf-analyzer/` **only after explicit final user approval**.

## Resolved (2026-11-10, user decisions)

- Data schema: legacy `img` key, (H, W, C); examples float32, C=500, no stored energy axis → channel axis by default, manual `E(i) = m·i + q` calibration in UI.
- Integration/aggregation: plain sums, no normalization (map over channels, spectrum over pixels).
- ROI interaction: zoom = selection (legacy semantics) + drag select + numeric panel + multi-point.
- `.npz` may carry an optional `energy` vector; `.npy` channel axis only.
- AIRES / model / inference / segmentation UI: removed (not re-asked). Custom raw reader: DEFER.