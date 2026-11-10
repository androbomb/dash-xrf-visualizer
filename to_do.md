# XRF-only visualizer — project to-do

> Future requirements are user decisions, not verified legacy implementation. **M0 is read-only**.

## Now — M0 audit, report only

- [ ] Verify `/workspace/exchange/dash-xrf-visualizer` exists and inspect `git status`.
- [ ] Inventory old Dash entrypoints, callbacks, image/spectrum calculation, energy and spatial ROI UI, format readers, and any code relying on `dash-xrf-analyzer/`.
- [ ] Identify AIRES-specific imports/callbacks/UI/code and propose removal boundaries; no need to re-ask whether AIRES stays (it does not).
- [ ] Inspect HDF5 and any permitted example file using metadata and bounded slices; record actual axis layout, energy calibration and data types.
- [ ] Inventory `.npz` / `.npy` support and determine the minimal new data contract, including external energy metadata if needed.
- [ ] Report which parts can be reused in proposed `src/`, `tests/`, `deploy/` structure and whether `dash-xrf-analyzer/` is tracked by Git.
- [ ] Propose M1 with small baseline fixture, feature/algorithm parity tests and a minimal edit set. **STOP for approval.**

## Next — requires authorization after M0

- [ ] Document known reference values for spectral integration and spatial ROI summation/aggregation.
- [ ] Create root uv packaging and importable source layout; migrate HDF5 reader and add `.npz` / `.npy` with validated schemas.
- [ ] Implement and validate JAX numerical reductions with explicit CPU/GPU reporting and dtype decisions.
- [ ] Split Dash UI layout, callbacks and plot construction; provide `example.ipynb` with `path_to_file`.
- [ ] Remove AIRES runtime code/dependencies and update actual human README.

## Later — not executable in Hermes' current environment

- [ ] Write `deploy/docker/` (Dockerfile/Compose), `deploy/wslc/` (Containerfile), `deploy/apptainer/` (`.def`) and `deploy/uv/` notebook-kernel instructions.
- [ ] Ask user to manually test Docker, WSLC and Apptainer; document real test results (do not claim they passed).
- [ ] Review removal of legacy `dash-xrf-analyzer/` **only after explicit final user approval**.

## Open technical questions (not feature-retention decisions)

- HDF5 group/key schema, dimension order, calibration units and spectral sampling.
- What `.npz` key schema and `.npy` metadata/energy-axis input are acceptable?
- Exact integration rule for elemental/intensity images and spatial ROI aggregation.
- ROI interaction: zoom versus selection, selectable shapes, default and reset behavior.
- Which user environment(s) will validate GPU, containers, notebook port/proxy access?
