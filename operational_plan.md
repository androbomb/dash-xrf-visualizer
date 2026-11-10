# Operational plan — XRF-only visualizer (Dash / Jupyter / JAX)

> **Status:** user-approved *target requirements*, NOT verified repository behavior.
> **Current milestone:** M0, a READ-ONLY audit; no implementation or package installation.
> **Repository path seen by Hermes:** `/workspace/exchange/dash-xrf-visualizer`.
> Any facts about the old program, data schemas, dependencies, and tested deploy targets remain `TBD` until inspected. The decisions explicitly fixed below must not be reopened unnecessarily.

## 1. Scope: decisions already made

The deliverable is **an XRF-only data visualizer**, NOT a machine-learning application.

| Feature | Target decision | Notes / questions for M0 |
| --- | --- | --- |
| HDF5 XRF input | **KEEP** | Discover actual groups, datasets, calibration and axis order. |
| `.npz` input | **ADD** | Define supported key names, shapes and metadata contract; do not guess. |
| `.npy` input | **ADD** | Contains an array but generally no calibration metadata: determine how energy coordinates are supplied, or label channels. |
| Left panel: XRF image | **KEEP** | Derive 2-D intensity image from user-selected spectral interval. |
| Right panel: XRF spectrum | **KEEP** | Display spectrum for current spatial region (initially full field). |
| Spectral interval / energy ROI | **KEEP** | Changing interval recomputes the elemental/intensity map. Respect legacy bin/energy semantics. |
| Spatial ROI selection/zoom | **KEEP** | Changing image ROI recomputes the spectrum. Define selection mechanics and reset behavior. |
| AIRES and all model/inference UI/code | **REMOVE** | Explicit user decision; preserve legacy baseline/evidence first. No need to ask again if a component is unambiguously AIRES-only. |
| Legacy custom-file reader | **DEFER** | Record findings; do not port now unless necessary for retained formats. |
| Everything else | **EVALUATE** | Classify only unknown/ambiguous functionality. |

The provided example XRF file may be inspected only via lightweight metadata or small slices in M0, when accessible and appropriate. Do NOT scan/load the whole dataset during M0.

### Retained behavior: functional contract

- Load a path supplied as a Python variable in a Jupyter notebook: `path_to_file = "/path/to/file"`. No hardcoded input paths or CLI-only workflow.
- Render a **left image panel** and a **right spectral panel**. Preserve axes, labels, units, and metadata as discoverable.
- A selected spectral interval updates the spatial map via a scientifically correct reduction/integration along the spectral axis.
- A selected spatial region updates the spectrum via a scientifically correct reduction over the selected pixels. Image zoom is not automatically a change in ROI: preserve or explicitly define which user gesture changes data selection.
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
- Compare retained legacy calculations with small reproducible references. Record precision (`float32` / `float64`, `jax_enable_x64`), tolerances, count overflow, axes, binning, NaN handling and CPU/GPU agreement. Avoid repeated recompilation caused by changing shapes.
- Interactive callbacks must not read arbitrary files, trigger expensive compilation, or move large cubes CPU↔GPU without an explicit design.

### 3.3 Dash / Notebook

- Use supported built-in Dash-in-Jupyter mode (on a verified compatible Dash version); avoid deprecated `JupyterDash` unless there is a specific compatibility necessity.
- The notebook starts with `path_to_file = "/path/to/file"`, creates app with a clean factory, then displays it inline. Package import must have no filesystem, server-start or GPU-compile side effect.
- Handle kernel reruns, duplicate callback registrations, port collisions, local bind address, and reverse proxies/container network mapping as appropriate. Do not assume Jupyter runs on the host; paths must resolve inside the kernel environment.

### 3.4 Deployment and reproducibility

- **uv** is the standard package/environment workflow; project metadata and lock file live in root. Commit `uv.lock`, not `.venv/`. Add `ipykernel` in the approved environment setup so JupyterLab can select the project kernel; document how to register it in `deploy/uv/README.md`.
- **Docker / Compose:** provide image, volume mounts for read-only XRF data, Jupyter/notebook access and documented port bindings. **WSLC:** its own `Containerfile` and documented `wslc` workflows. **Apptainer:** `.def` recipe with documented bind mounts and execution.
- Do not claim GPU access in any container target without actual user-side testing. Consider JAX CPU as portable baseline; GPU should be an optional well-documented variant, not an unconditional dependency.
- Hermes does not have Docker, WSLC or Apptainer available for this project: write/read-review recipes after approval and mark **NOT TESTED**. The user performs runtime validation and provides feedback. Do not run them in M0.
- Hermes may use `uv` for environment creation and kernel registration **after M0 and explicit authorization**, not during the read-only audit.

## 4. M0 audit (only authorized current work)

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
| **M0 — READ-ONLY** | Evidence report: existing code, formats, feature mapping, risks, precise M1 proposal | Stop for approval |
| **M1 — Baseline + approved docs** | Representative small fixture, known XRF arithmetic results, coordinated docs plan | Approval for bounded tests/edits |
| **M2 — Readers + source package** | HDF5 / NumPy readers, `src/` layout, minimal `tests/`, uv packaging | Test schemas and no AIRES coupling |
| **M3 — JAX + ROI reductions** | GPU-aware device policy, validated spectrum/map algorithms | CPU correctness before speed |
| **M4 — Dash / Jupyter** | Clear layout/callback split, root example notebook, interactive ROI | Verify notebook behavior |
| **M5 — Cleanup + deploy docs** | Prune AIRES, ignore bootstrap folder, container recipes, final README | User tests unexecutable runtimes; no unapproved deletions |

These are planning checkpoints, NOT evidence of work performed. Use small reversible commits only when the user authorizes writes; never push without explicit approval. Do not delete `dash-xrf-analyzer/` automatically.

## 6. Documentation ownership

- `AGENTS.md`: enduring constraints and resource policy.
- **This plan:** agreed scope, architecture, acceptance criteria and milestones.
- `to_do.md`: actionable prospective tasks (do not duplicate this whole plan).
- `timeline.md`: observed activity only; append when work actually happened.
- `scientific_notes.md`: only necessary physical/numerical assumptions or equations.
- `README.md`: currently bootstrap-only; replace with verified human-facing project guide when authorized.
- `first_prompt_telegram.md`: initial read-only M0 request, not independent authorization to run later tasks.
