# deploy/uv — environment and Jupyter kernel

Only instructions live here; the actual manifest/lock are at the repository
root (`pyproject.toml`, `uv.lock`).

## Setup (verified 2026-11-10 on this machine)

```bash
uv sync                      # creates .venv/ (git-ignored) from uv.lock
.venv/bin/python -m pytest -q
.venv/bin/python -m ipykernel install --user --name xrf-viz \
    --display-name "Python 3.12 (xrf-viz)"
```

On windows:
```powershell
# install if needed
powershell -c "irm https://astral.sh/uv/install.ps1 | more"
# allow
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
# activate
.\.venv\Scripts\Activate.ps1
# test
python -m pytest -q
# save kernel
python -m ipykernel install --user --name xrf-viz --display-name "Python 3.12 (xrf-viz)"
```

In Jupyter/JupyterLab select the kernel **Python 3.12 (xrf-viz)**.

## Compute backend: CPU by default (verified), GPU opt-in (NOT TESTED)

- The default install is **CPU-only JAX** (plain `jax` package, no CUDA
  plugin). Verified 2026-11-10 on a host with a visible RTX 5070 Ti:
  `import jax; jax.devices()` prints `CpuDevice(0)` and `nvidia-smi` shows no
  new CUDA process and no VRAM change — importing JAX does **not**
  pre-allocate GPU memory in this setup.
- All reductions run in float64 (see `scientific_notes.md`); CPU is also the
  fast path for interactive cube sizes (no CPU↔GPU transfer per callback).

### GPU variant (NOT TESTED — validate on your WSL2/Blackwell host)

```bash
uv add "jax[cuda12]"         # replaces plain jax; pulls jax-cuda12 + NVIDIA wheels
```

Known behavior to check/limit (NOT TESTED here):

- On first GPU use the XLA allocator pre-reserves a fraction of VRAM
  (default ~75%). Cap it before launching the kernel:

  ```bash
  export XLA_FLAGS=--xla_gpu_global_memory_fraction=0.3
  ```

- Verify a `CudaDevice` is actually usable:
  `.venv/bin/python -c "import jax; print(jax.devices())"`.
  A host-visible GPU is not a working JAX backend (Blackwell sm_120 needs a
  recent `jax-cuda12` build). The app reports the selected device and never
  falls back silently (M3 contract).