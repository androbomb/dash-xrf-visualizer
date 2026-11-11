# Apptainer recipe — **NOT TESTED** (no Apptainer in the Hermes environment;
# the user will run these). Builds a SIF from the Docker image, so make the
# image first (see `../docker/`):

```bash
docker build -t xrf-viz -f deploy/docker/Dockerfile .
apptainer build xrf-viz.sif deploy/apptainer/xrf-viz.def
```

Run (bind the data directory, point `XRF_CUBE` at the mounted file):

```bash
apptainer run --bind /abs/path/to/xrf/data:/data \
    -e XRF_CUBE=/data/Mantegna_mappa3b_centro-sx_35kV-90uA.npz \
    xrf-viz.sif
```

Then open http://localhost:8765. Apptainer binds work differently from Docker
(`--bind src:dst`); adjust flags to your Apptainer version if needed.