# Scientific notes — XRF visualizer

Verified conventions and numerical contracts. Facts marked **M0** come from
the read-only audit of 2026-11-10 (legacy code + example files); facts marked
**approved** were fixed by the user on 2026-11-10. Nothing here is invented.

## Data cube schema (M0, from legacy code + `data_example/`)

- Cube: 3-D array, **shape (H, W, C)** — rows, columns, spectral channels last.
  Legacy key: `img` (HDF5 dataset / .npz key); the key name was env-overridable
  in legacy (`NAME_OF_IMAGE_IN_FILE`, default `img`).
- **Example files** (user-supplied, in git-ignored `data_example/`, inspected
  via mmap metadata + small slices only):
  - `Mantegna_mappa3b_centro-sx_35kV-90uA.npz`: `img` float32, (76, 177, 500)
  - `Ragazzo-triste_area4_centrodx_38-80_calibrated.npz`: `img` float32, (145, 212, 500)
- **No energy axis is stored in the files** (no `energy` key, no attributes).
  Peak channels (~130–160 of 500) map to different keV under different ranges,
  so the per-file energy range cannot be recovered from the file alone.
- Storage dtype in the examples is **float32** (not integer counts).

## Energy axis and calibration (approved 2026-11-10)

- **Uncalibrated cube → channel (ADC) axis only.** The app must never invent a
  keV calibration from an array.
- **Manual calibration:** `E(i) = m·i + q` with `m` in keV/channel and `q` in
  keV, entered by the user in the UI. For C channels this spans `q` to
  `m·(C−1)+q`.
- Historical notes (context only, not reused):
  - Legacy hardcoded `0.5–38.5 keV` over all channels (`RendererXRF._E_int`,
    `BASIC_ENERGY_RANGE_FOR_CALIBRATION`); user-confirmed as a remnant of
    manual calibration for the (removed) AIRES network.
  - Legacy CHNet XML convention (deferred custom-raw path): `E = m·i·32 + q`
    (m per 32 ADC). When reusing old XML values, divide m by 32.

## Reduction rules (approved 2026-11-10)

- **Spectral interval → map:** `map[h, w] = Σ_{c=s}^{e-1} cube[h, w, c]` —
  **plain sum of counts, no energy-bin-width weighting, no normalization.**
- **Spatial ROI → spectrum:** `spec[c] = Σ_{(h,w)∈ROI} cube[h, w, c]` —
  **plain sum over pixels, no averaging, no normalization.**
- Multi-point selection (M4): sum over the listed pixel coordinates (same rule).
- **Empty selection → all zeros.** Out-of-range bounds are clamped at the UI
  boundary; the public API raises `ValueError` (M3).
- **Accumulation dtype: always float64**, regardless of storage dtype. float32
  accumulation gives different results (pinned by
  `tests/test_baseline_reductions.py::test_float64_promotion_rule`).
- NaN policy: propagate (no silent zeroing); stated explicitly in M3.

## Axis orientation (M0)

- Display: **x-axis = column (dim 1)**, y-axis = row (dim 0), y increasing
  downward (plotly imshow default `origin='upper'`).
- Legacy bug (do NOT port): `update_spectra` fed the display x-range into dim 0
  and the display y-range into dim 1 — transposed for non-square regions.
  The new app fixes the semantics above and pins it with non-square ROI tests.

## Golden baseline (M1, 2026-11-10)

- Deterministic synthetic cube (H, W, C) = (48, 64, 512), float32, closed form
  (no RNG): spatially varying background + Gaussian peaks at channels
  100/250/400; region [10:20, 30:40] doubled.
- Files: `tests/fixtures/synthetic_cube.{npz,h5,npy}` + `golden.json`
  (regenerate: `.venv/bin/python tests/make_fixture.py`).
- Total (float64): **23415048.096206427**.
- M3 (JAX) and M4 (UI) must match `golden.json` exactly.

## JAX backend policy (M1, 2026-11-10)

- Default environment: **CPU-only JAX** (plain `jax` package, no CUDA plugin).
  Verified on this host (RTX 5070 Ti visible): `import jax; jax.devices()` →
  `CpuDevice(0)`, no CUDA process, no VRAM change (`nvidia-smi` before/after).
- GPU is an **opt-in variant** (`jax[cuda12]`), documented in
  `deploy/uv/README.md`, marked **NOT TESTED**. Known XLA behavior to guard:
  the GPU allocator pre-reserves a fraction of VRAM on first use (see that doc).

`operational_plan.md` owns scope and acceptance criteria; `AGENTS.md` owns
rules; `timeline.md` records only work actually completed.