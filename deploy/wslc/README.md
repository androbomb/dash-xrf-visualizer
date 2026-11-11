# WSLC (WSL Container) recipe — **NOT TESTED** (no WSLC in the Hermes
# environment; the user will run these). Same image as `../docker/`, built with
# the `wslc` CLI. CPU-only JAX; GPU variant NOT TESTED.

Build (context = repository root):

```bash
wslc build --tag xrf-viz --file deploy/wslc/Containerfile .
```

Run:

```bash
wslc run --port 8765:8765 \
    --volume /abs/path/to/xrf/data:/data:ro \
    --environment XRF_CUBE=/data/Mantegna_mappa3b_centro-sx_35kV-90uA.npz \
    xrf-viz
```

Then open http://localhost:8765. Check `wslc --help` for exact flag names on
your WSLC version; this recipe mirrors the standard `docker run` form.