# End-user setup: virtual environment with `uv` + Jupyter notebook

Set up an isolated Python environment named `xrf-visualizer-env` with **uv**,
install the packages needed to run the visualizer in a Jupyter notebook
(`xrf-visualizer`, `ipykernel`, `jupyterlab`), and run the self-contained
`quickstart.ipynb` — no `example.ipynb` or example data file to distribute.

## 0. Install `uv`

`uv` is a fast (Rust) Python package and environment manager; it replaces
`pip` + `virtualenv` for this workflow. Install it once per machine.

macOS / Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify:

```bash
uv --version
```

If the command is not found, reopen the terminal or make sure
`~/.local/bin` (Linux/macOS) / `%USERPROFILE%\.local\bin` (Windows) is in your
`PATH`.

## 1. Create the virtual environment

In a working directory (e.g. `~/xrf` or `C:\xrf`):

```bash
uv venv xrf-visualizer-env
```

This creates a folder `xrf-visualizer-env/` containing the Python
interpreter and the environment. Pin a specific Python version if needed —
`uv` downloads it automatically if missing:

```bash
uv venv xrf-visualizer-env --python 3.12
```

## 2. Activate, install packages, register the kernel

### 2.1 Activate the environment

macOS / Linux:

```bash
source xrf-visualizer-env/bin/activate
```

Windows (PowerShell):

```powershell
.\xrf-visualizer-env\Scripts\Activate.ps1
```

Windows (cmd):

```cmd
xrf-visualizer-env\Scripts\activate.bat
```

After activation the prompt shows the environment name
(`(xrf-visualizer-env)`).

### 2.2 Windows: "script not signed" error in PowerShell

On the first activation attempt you may see:

```
.\xrf-visualizer-env\Scripts\Activate.ps1 : Impossibile caricare il file ...
non è firmato digitalmente. Lo script non può essere eseguito ...
    + FullyQualifiedErrorId : UnauthorizedAccess
```

PowerShell blocks unsigned scripts by default, and `Activate.ps1` is one of
them. Solutions:

- **Permanent, current user (recommended)** — run once, no admin needed:

  ```powershell
  Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
  ```

  Allows locally created scripts; downloaded scripts still need a signature.

- **Temporary, current session only:**

  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  ```

- **Settings UI:** *Settings → Privacy and security → For developers →
  PowerShell* → enable the option to allow unsigned local scripts → Apply.

- **If it persists:** right-click `xrf-visualizer-env\Scripts\Activate.ps1`
  → *Properties* → tick **Unblock** (bottom of the General tab) → OK.

### 2.3 Install the packages

With the environment active:

```bash
uv pip install xrf-visualizer ipykernel jupyterlab
```

- **`xrf-visualizer`**: this package (brings its own deps: numpy, h5py, jax,
  dash, plotly, dash-bootstrap-components).
- **`ipykernel`**: registers the environment as a Jupyter kernel.
- **`jupyterlab`**: the notebook interface.

### 2.4 Register the environment as a Jupyter kernel

With the environment active:

```bash
python -m ipykernel install --user --name xrf-visualizer --display-name "Python (xrf-visualizer)"
```

- `--user`: installs to the user's kernel directory
  (`~/.local/share/jupyter/kernels/` on Linux/macOS,
  `%APPDATA%\jupyter\kernels` on Windows), no admin rights.
- `python -m ipykernel` uses the active environment's interpreter, so no
  `VIRTUAL_ENV` path juggling is needed.

The kernel **Python (xrf-visualizer)** is now available in Jupyter.

## 3. Use the notebook

### 3.1 Start JupyterLab

With `xrf-visualizer-env` active:

```bash
jupyter lab
```

It opens in the browser (usually `http://localhost:8888/lab`).

### 3.2 Run `quickstart.ipynb` (self-contained)

`quickstart.ipynb` (in this repository's root) generates its own synthetic
test cube, so no example data is needed:

1. Put `quickstart.ipynb` in the folder you opened JupyterLab from
   (e.g. download it from the repository).
2. Open it and select the kernel **Python (xrf-visualizer)** (top-right).
3. Run all cells: the app appears inline — 2-D map on the left, spectrum on
   the right. Try a channel interval, an ROI, and the log-scale toggle.

Verify the kernel uses the right environment with:

```python
import sys
print("Python:", sys.version)
print("Executable:", sys.executable)
```

The path must contain `xrf-visualizer-env`.

> Version note: on the current 0.1.2 wheel on PyPI,
> `xrf_visualizer.__version__` still prints `0.1.0` (stale hard-coded value
> in the built wheel); it is fixed on master and will be correct from 0.1.3.

### 3.3 No notebook file? Paste this instead

```python
path_to_file = None  # or e.g. "my_cube.h5"

from xrf_visualizer.data import load_cube
from xrf_visualizer.ui   import build_app

cube = load_cube(path_to_file).array if path_to_file else None
app  = build_app(cube)
app.run(jupyter_mode="inline", debug=False, jupyter_height=800)
```

`build_app(None)` starts the app with **Upload** / **Load from link** in the
UI; from a terminal use `app.run()` instead (serves on
http://127.0.0.1:8050).

## Summary

| Step | Command |
|------|---------|
| Install `uv` | `curl -LsSf https://astral.sh/uv/install.sh \| sh` (macOS/Linux) or `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 \| iex"` (Windows) |
| Create environment | `uv venv xrf-visualizer-env` |
| Activate | `source xrf-visualizer-env/bin/activate` (macOS/Linux) / `.\xrf-visualizer-env\Scripts\Activate.ps1` (Windows) |
| PowerShell fix (once, Windows) | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` |
| Install packages | `uv pip install xrf-visualizer ipykernel jupyterlab` |
| Register kernel | `python -m ipykernel install --user --name xrf-visualizer --display-name "Python (xrf-visualizer)"` |
| Start JupyterLab | `jupyter lab` |
| Run | open `quickstart.ipynb` with kernel **Python (xrf-visualizer)**, Run All |