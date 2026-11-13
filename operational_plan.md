# Operational plan — XRF-only visualizer (Dash / Jupyter / JAX)

> **Status:** requirements implemented — M0–M26 completed 2026-11-10/13 (M3–M26 evidence in `timeline.md`).
> **Remaining:** Docker/Apptainer user-side tests, GPU validation, deferred items in Section 7. WSLC was reported working by the user.
> **Repository path seen by Hermes:** `/workspace/exchange/dash-xrf-visualizer`.
> M0 (read-only audit, 2026-11-10) verified the legacy code and the user-supplied example data; findings are recorded in `scientific_notes.md` and `timeline.md`. The decisions explicitly fixed below must not be reopened unnecessarily.

## 1. Scope: decisions already made

The deliverable is **an XRF-only data visualizer**, NOT a machine-learning application.

| Feature | Target decision | Notes / questions for M0 |
| --- | --- | --- |
| HDF5 XRF input | **KEEP** | M0: legacy reads dataset/key `img`, shape (H,W,C); no per-file calibration in the main path. |
| `.npz` input | **ADD** | M0: already supported in legacy (key `img`). New contract: `img` + optional `energy` float64 vector (keV) — approved 2026-11-10. |
| `.npy` input | **ADD** | M0: absent in legacy. Contract: channel (ADC) axis only, no implicit keV; manual m/q calibration in UI — approved 2026-11-10. |
| Left panel: XRF image | **KEEP** | Derive 2-D intensity image from user-selected spectral interval. |
| Right panel: XRF spectrum | **KEEP** | Display spectrum for current spatial region (initially full field). |
| Spectral interval / energy ROI | **KEEP** | Changing interval recomputes the elemental/intensity map. Respect legacy bin/energy semantics. |
| Spatial ROI selection/zoom | **KEEP** | Map rectangle selection → new spectrum, spectrum zoom → new map; explicit numeric and multi-point selection; `autorange` = reset. |
| AIRES and all model/inference UI/code | **REMOVE** | Explicit user decision; preserve legacy baseline/evidence first. No need to ask again if a component is unambiguously AIRES-only. |
| Legacy custom-file reader | **DEFER** | Record findings; do not port now unless necessary for retained formats. |
| Everything else | **EVALUATE** | Classify only unknown/ambiguous functionality. |

The provided example XRF file may be inspected only via lightweight metadata or small slices in M0, when accessible and appropriate. Do NOT scan/load the whole dataset during M0.

### Retained behavior: functional contract

- Load a path supplied as a Python variable in a Jupyter notebook: `path_to_file = "/path/to/file"`. No hardcoded input paths or CLI-only workflow.
- Render a **left image panel** and a **right spectral panel**. Preserve axes, labels, units, and metadata as discoverable.
- Preserve square map pixels at every viewport size while allowing arbitrary rectangular ROI selection; resize the map frame to the selected ROI ratio.
- Download the current channel-sum map as a PNG cropped to the visible spatial ROI, with one image pixel per XRF pixel, current colorscale/limits/log display, and no plot axes or colorbar.
- A selected spectral interval updates the spatial map via a scientifically correct reduction/integration along the spectral axis.
- Manual calibration `E(i)=m·i+q` changes the plotted spectrum axis only. Plot zoom ranges in keV are converted back to ADC channel indices before updating the map interval.
- The searchable XRF line catalog supports selecting a line and positive width in keV; Apply selects channel bins whose calibrated channel-center energies fall within that window. The reduction remains a discrete plain sum, not energy-width-weighted integration.
- A selected spatial region updates the spectrum via a scientifically correct reduction over the selected pixels. Dragging an arbitrary rectangle on the map sets the spatial ROI; zooming the spectrum recomputes the map for the selected channel interval. Numeric ROI and multi-point selection are also supported; `autorange`/double-click resets to the full field/interval.
- Initial full-field/full-spectrum selection, empty ROIs, zero counts, NaNs, nonuniform energy grids, and repeated selections must be handled deterministically.
- Distinguish **sum of counts** over energy channels from **integral with energy-bin widths**; use the legacy convention or a documented, approved scientific choice. Avoid silent conversion between channel and energy units.

## 2. Architecture: target layout, subject to M0 findings

The repository root should be small. The example notebook is the only user-facing notebook in root. `pyproject.toml` and `uv.lock` are intentionally **root-level** files (uv project model), not placed under `deploy/`.

```text
.
├── .gitignore
├── AGENTS.md
├── README.md
├── operational_plan.md
├── to_do.md
├── timeline.md
├── scientific_notes.md                 # optional scientific context
├── pyproject.toml                     # uv project/dependencies (planned)
├── uv.lock                            # version-controlled lock (planned)
├── .python-version                    # optional interpreter pin
├── example.ipynb                      # notebook UX example
├── src/
│   └── xrf_visualizer/                # provisional import name
│       ├── __init__.py
│       ├── app.py                     # Dash app factory, never auto-run on import
│       ├── data/
│       │   ├── readers.py            # HDF5 and NumPy adapters
│       │   └── schema.py             # dimensions, coordinates, metadata validation
│       ├── processing/
│       │   ├── backend.py            # JAX availability / devices / precision
│       │   └── reductions.py         # spectrum/map reductions and ROI operations
│       └── ui/
│           ├── layout.py             # Dash HTML/components, nothing numerical
│           ├── callbacks.py          # orchestration / state mapping
│           └── figures.py            # Plotly figure construction
├── tests/                             # pytest convention (user said "test")
│   ├── test_readers.py
│   ├── test_reductions.py
│   └── test_interactions.py
└── deploy/
    ├── docker/                      # Dockerfile, compose.yaml, README
    ├── wslc/                        # Containerfile, README for Microsoft's wslc
    ├── apptainer/                   # .def recipe, README
    └── uv/                          # environment / Jupyter kernel how-to ONLY
```

Tree is an understandable **proposal**, not an instruction to create empty files or excessive abstractions. Keep the smallest number of modules that really separates data ingestion, numerical operations, UI layout, callbacks and Plotly. If a compelling reason exists to use a different test directory or package name, report it at the gate.

**Actual layout (M2–M4):** `data/__init__.py` (readers + schema validation in one module — the split into `readers.py`/`schema.py` was not justified), `ui/app.py` (app factory, figures and callbacks in one module; splitting into `layout.py`/`callbacks.py`/`figures.py` would mirror the diagram without buying separation, per `AGENTS.md` rule 4). `app.py` in the package root was not created — `build_app` lives in `ui/app.py`.

### Legacy folder

- Root-relative `dash-xrf-analyzer/` is a **bootstrap/reference-only** folder, to be ignored by Git (`/dash-xrf-analyzer/`).
- In M0, inspect read-only if needed to understand the old code; never treat it as the target package, install from it, import it in the new app, or make runtime code depend on its path.
- Do **not** remove the directory in M0. User will decide the final removal after migration and baseline checks. Ignoring it is not equivalent to deleting already-tracked Git files: first check `git ls-files dash-xrf-analyzer`.

## 3. Technical boundaries

### 3.1 Data access

- Public Python entry point may be `create_app(path_to_file, ...)` (name provisional). `str` and `pathlib.Path` accepted.
- Validate file existence, extension, dataset shape and dimensionality, metadata, energy/channel mapping and axes; specify informative exceptions.
- Data are conceptually a spatial + spectral cube, but actual axes/order must be discovered; do not assume `(height, width, channels)` from filename alone.
- `.npy` has no inherent energy labels; `.npz` may provide arrays/coordinate vectors, subject to a documented schema. Never invent physical calibration from an array.
- Keep raw data read-only; avoid unnecessary full-cube copies and repeated HDF5 opening on every callback. Evaluate bounded cache lifetime and GPU memory limits only after M0.

### 3.2 Numerical processing: JAX

- Use `jax.numpy` in retained numerical kernels **where it adds value**; use h5py, NumPy, and Plotly interoperability where required. No PyTorch as a retained dependency unless an explicitly retained function actually needs it (expected: none after removing AIRES).
- Detect the backend **visible to JAX from the running notebook kernel**. Prefer a supported GPU if accessible; otherwise CPU, and report the selected device. Do not equate host `nvidia-smi` with a working JAX GPU backend.
- Avoid silent fallback on a broken backend initialization. Surface meaningful diagnostics; document CPU-only and GPU installation paths separately, with compatible versions rather than a speculative universal command.
- **M1 verified (2026-11-10):** the default environment installs CPU-only JAX (no CUDA plugin); `import jax; jax.devices()` created no CUDA context and pre-allocated no VRAM (`nvidia-smi` before/after on an RTX 5070 Ti host). GPU is an opt-in documented variant (`jax[cuda12]`) in `deploy/uv/README.md` (NOT TESTED). Reductions always accumulate in float64 (example data is float32).
- Compare retained legacy calculations with small reproducible references. Record precision (`float32` / `float64`, `jax_enable_x64`), tolerances, count overflow, axes, binning, NaN handling and CPU/GPU agreement. Avoid repeated recompilation caused by changing shapes.
- Interactive callbacks must not read arbitrary files, trigger expensive compilation, or move large cubes CPU↔GPU without an explicit design.

### 3.3 Dash / Notebook

- Use supported built-in Dash-in-Jupyter mode (on a verified compatible Dash version); avoid deprecated `JupyterDash` unless there is a specific compatibility necessity. **M1 verified (2026-11-10):** `app.run(jupyter_mode=...)` is present in dash 4.5.0 in the project venv.
- The notebook starts with `path_to_file = "/path/to/file"`, creates app with a clean factory, then displays it inline. Package import must have no filesystem, server-start or GPU-compile side effect.
- Handle kernel reruns, duplicate callback registrations, port collisions, local bind address, and reverse proxies/container network mapping as appropriate. Do not assume Jupyter runs on the host; paths must resolve inside the kernel environment.

### 3.4 Deployment and reproducibility

- **uv** is the standard package/environment workflow; project metadata and lock file live in root. Commit `uv.lock`, not `.venv/`. Add `ipykernel` in the approved environment setup so JupyterLab can select the project kernel; document how to register it in `deploy/uv/README.md`.
- **Docker / Compose:** provide image, volume mounts for read-only XRF data, Jupyter/notebook access and documented port bindings. **WSLC:** its own `Containerfile` and documented `wslc` workflows. **Apptainer:** `.def` recipe with documented bind mounts and execution.
- Do not claim GPU access in any container target without actual user-side testing. Consider JAX CPU as portable baseline; GPU should be an optional well-documented variant, not an unconditional dependency.
- Hermes does not have Docker, WSLC or Apptainer available for this project: write/read-review recipes after approval and mark **NOT TESTED**. The user performs runtime validation and provides feedback. Do not run them in M0.
- Hermes may use `uv` for environment creation and kernel registration **after M0 and explicit authorization**, not during the read-only audit.

## 4. M0 audit (completed 2026-11-10 — findings in `scientific_notes.md` and `timeline.md`)

1. Check repository path `/workspace/exchange/dash-xrf-visualizer` and `git status`. Stop if absent; do not search elsewhere by guessing.
2. Map original Dash entrypoints, callbacks, data readers, visualization pipeline, AIRES dependencies, custom readers, existing tests and documented runtime requirements. Treat `dash-xrf-analyzer/` as read-only reference.
3. Inspect only file metadata/small bounded slices for example XRF files; record actual shapes, dtypes, calibration and schema if available, never invent them.
4. Identify what can be directly reused for the **already-defined** retained features, with exact paths/function names. Document `KEEP / REMOVE (AIRES) / DEFER (custom) / NEEDS CLARIFICATION` only for unknown features.
5. Propose a practical minimal tree (compare with Section 2), risks and tests for XRF sums, energy ROI, spatial ROI, CPU/GPU, file schemas and notebook interaction.
6. Propose M1 scope and synchronized updates to docs **without applying any edits**. Stop for user approval.

**M0 prohibited:** edit/write/move/delete files or docs; dependency/environment installation; downloads; start servers; costly tests, full data loading, jobs; commits/pushes. No code reformat/restructure yet. Inspect git-tracked content safely. Outputs are a report in the Hermes conversation only.

## 5. Validation criteria and milestones (future, approval-gated)

| ID | Acceptance check | Required evidence |
| --- | --- | --- |
| F1 | HDF5 file loads correctly | One representative schema fixture, axes, shape, metadata. |
| F2 | `.npy` and `.npz` load under documented schema | Small fixtures; explicit energy coordinates or channel-index fallback. |
| F3 | Energy ROI updates 2-D map | Golden arithmetic tests for channel/energy bounds and reduction. |
| F4 | Spatial ROI updates spectrum | Synthetic cube, selected pixels and expected sums; zoom/selection semantics tested. |
| F5 | Left image + right spectrum in Jupyter | Interactive user check and callback test on one allowed fixture. |
| F6 | JAX CPU and precision parity | Small repeatable JAX/reference comparisons, measured tolerances. |
| F7 | JAX GPU if available | Real device use + agreement; otherwise clearly **NOT TESTED**. |
| F8 | AIRES absent in new runtime | No model entry point, imports, UI, runtime deps or invocation paths. |
| F9 | Reproducible uv environment | `pyproject.toml`, versioned `uv.lock`, documented kernel commands, once user approves and allows test. |
| F10 | Container deployment recipes | Files/review done; Docker/WSLC/Apptainer marked **NOT TESTED** until user checks. |

| Milestone | Output | Gate |
| --- | --- | --- |
| **M0 — READ-ONLY** ✅ 2026-11-10 | Evidence report: existing code, formats, feature mapping, risks, precise M1 proposal | Approved 2026-11-10 |
| **M1 — Baseline + approved docs** ✅ 2026-11-10 | Representative small fixture, known XRF arithmetic results, coordinated docs plan | Approved 2026-11-10 |
| **M2 — Readers + source package** ✅ 2026-11-10 | HDF5 / NumPy readers, `src/` layout, minimal `tests/`, uv packaging | Test schemas and no AIRES coupling (17 passed) |
| **M3 — JAX + ROI reductions** ✅ 2026-11-10 | Device policy + float64 reductions, bit-exact vs goldens | CPU correctness before speed (24 passed, GPU skipped) |
| **M4 — Dash / Jupyter** ✅ 2026-11-11 | `ui/app.py` (factory, figures, callbacks), root example notebook, zoom = selection + explicit panels | Server-level E2E bit-exact vs processing (31 passed, GPU skipped) |
| **M5 — Cleanup + deploy docs** ✅ 2026-11-11 | AIRES/TF absence grep (0 hits), `deploy/{docker,wslc,apptainer}/` recipes **NOT TESTED**, final README | User tests unexecutable runtimes; no unapproved deletions |
| **M6 — Empty start + file loading** ✅ 2026-11-11 | `build_app(None)`; upload (`.npz`/`.h5`/`.npy`) or http(s) link in the UI; `sniff_format`/`load_cube_bytes`/`download_cube`; `XRF_CUBE` optional in containers | 49 passed + 1 GPU skip; empty-app E2E bit-exact via HTTP (upload → map vs `map_from_channels`) |
| **M7 — UI fix + legacy look** ✅ 2026-11-11 | dash 4.5 return convention fixed (single `Output` → raw value; the `[value]` wrap double-wrapped stores → WSLC 500); legacy dbc look ported (DARKLY, navbar `#4196B4`, 5/7 columns); `dash-bootstrap-components` added | 49 passed + 1 GPU skip; HTTP E2E with the real renderer contract (figure DICTs, flat stores, zoom → recompute bit-exact) |
| **M8 — Selection labels + controls layout** ✅ 2026-11-12 | Horizontal map controls, 2×2 ROI grid, wider rows/cols and calibration inputs, explicit calibration action; plot titles expose selected map interval and spectrum ROI/points; calibration remains X-axis-only | 49 passed + 1 GPU skip; assertions for interval units, ROI/point titles and calibrated axis |
| **M9 — Callback behavior + notebook compatibility** ✅ 2026-11-12 | Selection buttons use the actual callback trigger instead of stale `relayoutData`; figure callbacks retain live Input refresh and use ordered Input/State arguments; notebook example uses Dash 4.x `jupyter_mode="inline"` | HTTP callback regressions cover ROI/channel zoom, explicit overrides, map limits including zero, and calibration |
| **M10 — Numeric actions + plot zoom** ✅ 2026-11-12 | Apply ROI/interval also zoom their associated plot; Apply scale button explicitly refreshes display settings; `jupyter_mode=True` remains backward-compatible via normalization to `"inline"` | HTTP callback tests verify opposite-figure updates and axis ranges; suite: 53 passed + 1 GPU skip |
| **M11 — Plotly interaction and shared controls** ✅ 2026-11-12 | Parse real `axis.range[n]` relayout payloads; show selected points as map markers; apply log1p to spectrum counts; gate numeric zmin/zmax on Apply scale while colormap/log stay live | HTTP tests use actual Plotly relayout keys and compare serialized figure arrays to processing results |
| **M12 — Scale, clear, calibration** ✅ 2026-11-12 | Apply zmin/zmax to `layout.coloraxis.cmin/cmax`; Clear selection resets all stores; calibration reports and displays active m/q values | HTTP tests verify coloraxis limits, full-map/full-spectrum reset, and calibrated title/range |

These are planning checkpoints, NOT evidence of work performed. Use small reversible commits only when the user authorizes writes; never push without explicit approval. Do not delete `dash-xrf-analyzer/` automatically.

**F-status (evidence in `timeline.md`):** F1/F2 — M2 (round-trip fixtures + real-file metadata). F3/F4 — M3 (goldens) + M4 (zoom E2E). F5 — M4 at server level (figures bit-exact via callback endpoint, HTTP 200); *visual browser check still user-side* (no Chromium in Hermes). F6 — M3 (bit-exact CPU vs float64 reference). F7 — **NOT TESTED** (GPU skip in suite). F8 — M5 (grep 0 hits in `src/` + `pyproject.toml`). F9 — M1 (uv sync, kernel `xrf-viz`). F10 — M5 (recipes present, **NOT TESTED**).

## 6. Documentation ownership

- `AGENTS.md`: enduring constraints and resource policy.
- **This plan:** agreed scope, architecture, acceptance criteria and milestones.
- `to_do.md`: actionable prospective tasks (do not duplicate this whole plan).
- `timeline.md`: observed activity only; append when work actually happened.
- `scientific_notes.md`: only necessary physical/numerical assumptions or equations.
- `README.md`: human-facing project guide (replaced the bootstrap template in M1; kept in sync at M4/M5).
- `first_prompt_telegram.md`: initial read-only M0 request, not independent authorization to run later tasks.

## 7. Deferred / pending (2026-11-11)

- **TO DO (user decision, on hold):** drag-mode select on the map — plotly drag-mode selection in addition to zoom = selection. Tracked in `to_do.md` (Later); implement only on explicit request.
- User-side: run the `deploy/` recipes (`how_to_deploy_*.md`), GPU validation on WSL2 (RTX 5070 Ti, `jax[cuda12]` variant), visual browser pass of the UI, removal decision for `dash-xrf-analyzer/`.

### M26 — multi-line elemental-map and spectrum gallery (completed 2026-11-13)

- **UX:** an `<hr>`-separated panel has a searchable line-add dropdown, unique per-line rows with editable keV widths, and a manually navigable gallery. Duplicate additions are no-ops; adding/computing requires applied calibration. Each full-field map is shown beside its associated full-field spectrum; titles show line, integration width, map range and scale range. The spectrum marks the integration zone and catalog line center.
- **Numerics:** applied manual m/q only. For each line, select calibrated channel centers in `[E_line-width/2, E_line+width/2]` and sum with the existing unweighted float64 reduction over the whole H×W field, independent of ROI, points or spectrum selection. The associated spectrum is the full-field sum over all channels; include its count-weighted centroid inside the selected zone when defined.
- **Scale/export:** shared full-field count range by default, or per-line autoscale; reuse current colormap and log1p setting. ZIP contains exact-size, axis/colorbar-free map PNGs, matching vector `*_spectra.svg` plots, and `metadata.json` with filenames, scale/data ranges, line/window/channel bounds, calibration, cube shape, source filename, display settings and observed centroid. Arrays remain server-side; no dependency added.
- **Verification:** tests cover duplicate no-op, calibration gating, decimal widths, full-field reduction, gallery ranges, spectrum zone, shared/per-line scale, PNG/SVG export and manifest. Full suite: 63 passed, 1 skipped (GPU). Browser/Jupyter visual verification remains user-side.
