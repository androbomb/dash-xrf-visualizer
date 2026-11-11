# Docker recipe — **NOT TESTED** (no Docker in the Hermes environment; the
# user will run these). CPU-only JAX: no GPU passthrough here; the GPU variant
# is documented in `../uv/README.md` and is also NOT TESTED.

Build (context = repository root, so `COPY pyproject.toml ...` resolves):

```bash
docker build -t xrf-viz -f deploy/docker/Dockerfile .
```

Run (bind the data directory read-only, point `XRF_CUBE` at the mounted
file — **optional**, M6):

```bash
docker run --rm -p 8765:8765 \
    -v /abs/path/to/xrf/data:/data:ro \
    -e XRF_CUBE=/data/Mantegna_mappa3b_centro-sx_35kV-90uA.npz \
    xrf-viz
```

Without `XRF_CUBE` (and no bind) the app starts empty: upload a
`.npz`/`.h5`/`.npy` or paste a download link in the UI.

Then open http://localhost:8765. Tests are not installed in the image
(`uv sync --frozen --no-dev`); add the `dev` group inside the container if
needed.