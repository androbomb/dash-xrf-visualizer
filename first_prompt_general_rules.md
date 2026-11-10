# Rules for a future first task message (fallback, not primary instructions)

`AGENTS.md` and `operational_plan.md` are the authoritative repository rules;
this file is a compact reminder and never grants extra permission.

- First verify the user-supplied repository root. For this project Hermes sees
  `/workspace/exchange/dash-xrf-visualizer` (not the later XRF *data file* path).
- In **M0**: READ ONLY; check git status, architecture, callbacks, readers and
  small metadata. No edits, installs, downloads, servers, commits or push.
- User has already decided the retained functionality: XRF viewer only; HDF5,
  additional `.npy`/`.npz`, spectral map and spatial ROI spectrum interactions.
  **Remove AIRES** when implementation is approved. Defer custom readers.
- `dash-xrf-analyzer/` is bootstrap reference only. No runtime dependencies on it.
  Ignore via .gitignore; do not delete it in M0 or without explicit future consent.
- Proposed packaging: `src/xrf_visualizer`, `tests/`, `deploy/{docker,wslc,apptainer,uv}`,
  one root example notebook, root `pyproject.toml` and tracked `uv.lock`.
- Use JAX only for suitable numerical calculations; keep necessary NumPy/HDF5
  interoperability. Check real devices, validate CPU math, report GPU if tested.
- Validate actual axis order, energy/channel calibration and integration/ROI
  aggregation rules; never invent scientific conventions.
- After authorized implementation, run only bounded tests, avoid destructive
  refactors, make a simplification pass, synchronize docs and report real results.
- Container recipes can be written after approval; runtime tests with Docker,
  WSLC or Apptainer are for the user if Hermes does not have those tools.
- End M0 with evidence, unresolved technical questions, minimal M1 recommendation
  and **STOP** for approval.
