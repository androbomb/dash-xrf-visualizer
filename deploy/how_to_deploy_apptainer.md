# How to deploy — Apptainer (SIF)

**Status:** NOT TESTED (recipe drafted 2026-11-11; run this in your
environment and fill the result log). CPU-only JAX: no GPU passthrough.
The SIF boots from the Docker image, so you need Docker only for the build
step.

## Prerequisites

- Docker (build step) and Apptainer (`apptainer --version`).
- A data directory on the host.

## Steps

1. Build the base image (from the repository root):

   ```bash
   docker build -t xrf-viz -f deploy/docker/Dockerfile .
   ```

2. Build the SIF:

   ```bash
   apptainer build xrf-viz.sif deploy/apptainer/xrf-viz.def
   ```

   (`xrf-viz.def` uses `Bootstrap: docker / From: xrf-viz:latest`.)

3. Run — with a preloaded cube (adjust the path and cube filename):

   ```bash
   apptainer run --bind /home/<you>/xrf-data:/data \
       -e XRF_CUBE=/data/Mantegna_mappa3b_centro-sx_35kV-90uA.npz \
       xrf-viz.sif
   ```

   Or empty (M6 — no bind, no env): the app starts with "no cube loaded"
   and you upload a file or paste a download link in the UI:

   ```bash
   apptainer run xrf-viz.sif
   ```

4. Open `http://localhost:8765`. First request compiles the JAX reductions —
   a few seconds of wait is expected.

## Notes

- Apptainer bind syntax differs from Docker (`--bind src:dst`, no `:ro` in
  older versions — check `apptainer run --help` for read-only bind support).
- If Apptainer cannot reach the local Docker daemon for the bootstrap, export
  the image instead: `docker save xrf-viz:latest -o xrf-viz.tar` and point
  `From:` in the `.def` at the tar (or use `Bootstrap: localimage`).
- GPU variant: NOT TESTED; see `uv/README.md` for the `jax[cuda12]` path.

## Checklist

- [ ] `apptainer build` succeeds.
- [ ] Page loads at `http://localhost:8765` (map left, spectrum right).
- [ ] Empty start (no `XRF_CUBE`) → upload a `.npz`/`.h5`/`.npy` (or paste a link) → map and spectrum appear (M6).
- [ ] Zoom the map → spectrum updates; zoom the spectrum → map updates.
- [ ] `autorange` resets to full field/interval.
- [ ] Numeric ROI panel and multi-point panels work (error msg on bad input).
- [ ] `m`/`q` calibration switches the spectrum axis to keV.

## Result log (fill in after your test)

| date | apptainer version | build | run | UI checks | notes |
| ---- | ----------------- | ----- | --- | --------- | ----- |
|      |                   |       |     |           |       |