# Scientific notes — XRF visualizer (optional)

This is a reference for future documentation, not a task for M0 and not evidence
about the legacy application. Do not create LaTeX files without a reason.

When inspecting/implementing the retained visualizer, document only verified or
explicitly approved scientific conventions:

- Input data-cube axis order, units, acquisition metadata, energy calibration,
  energy-channel correspondence and missing metadata handling.
- **Spectral ROI → map:** whether reduction is a discrete sum of counts over
  channels or a width-weighted integral over energy, and how boundary bins count.
- **Spatial ROI → spectrum:** whether the selected pixels are summed, averaged,
  or normalized; pixel coordinates, masks, image zoom versus actual selection.
- Handling of invalid/empty selections, NaNs, saturated/negative values,
  integer overflow, `float32` vs `float64` and JAX x64 configuration.
- Baseline/golden test values and numerical CPU/GPU comparisons, with observed
  tolerances and actual input provenance. Never invent calibration or results.

No AIRES model, training, inference or custom reader derivations belong in the
retained XRF-only deliverable. Keep a separate historical note only if necessary
for explaining a deliberate removal.

`operational_plan.md` owns scope and acceptance criteria; `AGENTS.md` owns rules;
`timeline.md` records only work actually completed.
