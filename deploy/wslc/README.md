# WSLC (WSL Container) recipe — user reports successful deployment; not run in
# this environment. Same image as `../docker/`, built with the `wslc` CLI.
# CPU-only JAX; GPU variant NOT TESTED.

Build (context = repository root):

```powershell
wslc build --tag xrf-viz --file deploy/wslc/Containerfile .
```

Run (`XRF_CUBE` **optional**, M6 — without it the app starts empty and
you upload a file or paste a download link in the UI):

```powershell
wslc run --name dash-xrf-viewer -p 8765:8765 \
    -v C:\Users\<you>\xrf-data:/data:ro \
    -e XRF_CUBE=/data/Mantegna_mappa3b_centro-sx_35kV-90uA.npz \
    xrf-viz
```

In one line: 
```powershell
wslc run --name dash-xrf-viewer -p 8765:8765 -v /abs/path/to/xrf/data:/data:ro -e XRF_CUBE=/data/Mantegna_mappa3b_centro-sx_35kV-90uA.npz xrf-viz
```


(Flags verified against Microsoft's WSLC docs 2026-11-11: `-p`/`-v`/`-e`,
docker short-flag form. `-v` takes a Windows source path; drop `:ro` if
your wslc version rejects it.)

Then open http://localhost:8765. The user reports that the WSLC deployment and
UI work correctly. It supports arbitrary rectangular map selections and
exact-pixel PNG downloads. Docker/Apptainer runtime recipes have not been tested.
