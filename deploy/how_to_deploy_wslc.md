# How to deploy — WSLC (WSL Container)

**Status:** NOT TESTED (recipe drafted 2026-11-11; run this in your
environment and fill the result log). CPU-only JAX: no GPU passthrough.
Flag names mirror the standard `docker run` form — if your `wslc` version
differs, check `wslc build --help` / `wslc run --help`.

## Prerequisites

- Microsoft WSLC (WSL Container) installed and running (`wslc --version`).

## Steps

1. From the repository root:

   ```bash
   wslc build --tag xrf-viz --file deploy/wslc/Containerfile .
   ```

2. Run (adjust the path and cube filename):

   ```bash
   wslc run --port 8765:8765 \
       --volume /home/<you>/xrf-data:/data:ro \
       --environment XRF_CUBE=/data/Mantegna_mappa3b_centro-sx_35kV-90uA.npz \
       xrf-viz
   ```

3. Open `http://localhost:8765`. First request compiles the JAX reductions —
   a few seconds of wait is expected.

## Notes

- The `Containerfile` is content-identical to `deploy/docker/Dockerfile`
  (WSLC accepts the same OCI build format); keep them in sync if you change one.
- The WSLC image store is separate from Docker's; building both is cheap since
  the layers are the same, but the first build downloads the base image again.
- GPU variant: NOT TESTED; see `uv/README.md` for the `jax[cuda12]` path.

## Checklist

- [ ] Build succeeds.
- [ ] Page loads at `http://localhost:8765` (map left, spectrum right).
- [ ] Zoom the map → spectrum updates; zoom the spectrum → map updates.
- [ ] `autorange` resets to full field/interval.
- [ ] Numeric ROI panel and multi-point panels work (error msg on bad input).
- [ ] `m`/`q` calibration switches the spectrum axis to keV.

## Result log (fill in after your test)

| date | wslc version | build | run | UI checks | notes |
| ---- | ------------ | ----- | --- | --------- | ----- |
|      |              |       |     |           |       |