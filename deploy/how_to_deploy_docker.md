# How to deploy — Docker

**Status:** NOT TESTED (recipe drafted 2026-11-11; run this in your
environment and fill the result log). CPU-only JAX: no GPU passthrough.

## Prerequisites

- Docker (BuildKit default is fine).
- A data directory on the host, e.g. `/home/<you>/xrf-data/` containing the cube.

## Steps

1. From the repository root (context must be the root so `COPY` resolves):

   ```bash
   docker build -t xrf-viz -f deploy/docker/Dockerfile .
   ```

2. Run — with a preloaded cube (adjust the path and cube filename):

   ```bash
   docker run --rm -p 8765:8765 \
       -v /home/<you>/xrf-data:/data:ro \
       -e XRF_CUBE=/data/Mantegna_mappa3b_centro-sx_35kV-90uA.npz \
       xrf-viz
   ```

   Or empty (M6 — no env, no bind): the app starts with "no cube loaded"
   and you upload a file or paste a download link in the UI:

   ```bash
   docker run --rm -p 8765:8765 xrf-viz
   ```

3. Open `http://localhost:8765`. First request compiles the JAX reductions —
   a few seconds of wait is expected.

## Notes

- `README.md` is copied alongside `pyproject.toml`: hatchling validates the
  declared `readme = "README.md"` during the editable build of the project.
- The `git was not found in the system` warning during the build is harmless
  (metadata only).
- Image is `ghcr.io/astral-sh/uv:python3.12-bookworm-slim` + `uv sync --frozen --no-dev`
  (no pytest/ipykernel inside).
- To iterate on code: rebuild, or `docker run -it --entrypoint uv ...` with a
  shell — the app code lives at `/app` in the image.
- GPU variant: NOT TESTED; see `uv/README.md` for the `jax[cuda12]` path
  (a different base image would be needed).

## Checklist

- [ ] Build succeeds.
- [ ] Page loads at `http://localhost:8765` (map left, spectrum right).
- [ ] Empty start (no `XRF_CUBE`) → upload a `.npz`/`.h5`/`.npy` (or paste a link) → map and spectrum appear (M6).
- [ ] Zoom the map → spectrum updates; zoom the spectrum → map updates.
- [ ] `autorange` resets to full field/interval.
- [ ] Numeric ROI panel and multi-point panels work (error msg on bad input).
- [ ] `m`/`q` calibration switches the spectrum axis to keV.
- [ ] With calibration applied, zooming the spectrum updates the matching channel interval on the map.
- [ ] Select an XRF line and width → Apply selects its centered window; Reset scale clears zmin/zmax.

## Result log (fill in after your test)

| date | docker version | build | run | UI checks | notes |
| ---- | -------------- | ----- | --- | --------- | ----- |
|      |                |       |     |           |       |
