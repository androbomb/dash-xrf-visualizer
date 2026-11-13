# How to deploy — WSLC (WSL Container)

**Status:** The user reports successful WSLC deployment and UI use; the recipe
has not been run in this environment. CPU-only JAX: no GPU passthrough.
`wslc run` uses the docker short-flag form `-p` / `-v` / `-e` (verified
against Microsoft's WSLC docs 2026-11-11); the `wslc build --tag --file`
form worked in a real user build. If your version differs, check
`wslc build --help` / `wslc run --help`.

## Prerequisites

- Microsoft WSLC (WSL Container) installed and running (`wslc --version`).

## Steps

1. From the repository root:

   ```bash
   wslc build --tag xrf-viz --file deploy/wslc/Containerfile .
   ```

2. Run — with a preloaded cube (adjust the path and cube filename):

   ```bash
   wslc run --name dash-xrf-viewer -p 8765:8765 \
       -v C:\Users\<you>\xrf-data:/data:ro \
       -e XRF_CUBE=/data/Mantegna_mappa3b_centro-sx_35kV-90uA.npz \
       xrf-viz
   ```

   (`-v` takes a **Windows** source path or a named volume; the `:ro`
   suffix follows the docker form — drop it if your wslc version rejects it.)

   Or empty (M6 — no env, no volume): the app starts with "no cube
   loaded" and you upload a file or paste a download link in the UI:

   ```bash
   wslc run --name dash-xrf-viewer -p 8765:8765 xrf-viz
   ```

3. Open `http://localhost:8765`. First request compiles the JAX reductions —
   a few seconds of wait is expected.

## Notes

- The `WARNING: current commit information was not captured by the build: git
  was not found` line is harmless (metadata only; git is not in the image) —
  the build proceeds.
- The `Containerfile` is content-identical to `deploy/docker/Dockerfile`
  (WSLC accepts the same OCI build format); keep them in sync if you change one.
- The WSLC image store is separate from Docker's; building both is cheap since
  the layers are the same, but the first build downloads the base image again.
- GPU variant: NOT TESTED; see `uv/README.md` for the `jax[cuda12]` path.

## Checklist

- [ ] Build succeeds.
- [ ] Page loads at `http://localhost:8765` (map left, spectrum right).
- [ ] Empty start (no `XRF_CUBE`) → upload a `.npz`/`.h5`/`.npy` (or paste a link) → map and spectrum appear (M6).
- [ ] Draw a rectangular selection on the map → spectrum updates; zoom the spectrum → map updates.
- [ ] `autorange` resets to full field/interval.
- [ ] Numeric ROI panel and multi-point panels work (error msg on bad input).
- [ ] `m`/`q` calibration switches the spectrum axis to keV.
- [ ] With calibration applied, zooming the spectrum updates the matching channel interval on the map.
- [ ] Select an XRF line and width → Apply selects its centered window; Reset scale clears zmin/zmax.
- [ ] Download current map → PNG dimensions equal selected matrix dimensions, with no axes/colorbar.

## Result log (fill in after your test)

| date | wslc version | build | run | UI checks | notes |
| ---- | ------------ | ----- | --- | --------- | ----- |
| 2026-11-13 | not recorded | user-reported | user-reported | user reports expected behavior | exact version/checklist not recorded |
|      |              |       |     |           |       |
