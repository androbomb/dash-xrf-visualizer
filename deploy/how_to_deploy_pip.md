# Deploy with pip / PyPI

This is the pip-based alternative to the repository's `uv` workflow. It is a
deployment guide and release checklist; **the project is not published on
   PyPI yet**, so `pip install xrf-visualizer` will work only after the
first release.

## End-user install (after publication)

```bash
python -m venv .venv
# Linux/macOS
. .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install xrf-visualizer
python -m pip install ipykernel  # needed to register/use this environment as a Jupyter kernel
python -m ipykernel install --user --name xrf-viz \
  --display-name "Python (xrf-visualizer)"
```

Select the named kernel in Jupyter, then follow the notebook pattern in
`example.ipynb`: load the cube, call `build_app(cube)`, and use
`app.run(jupyter_mode="inline", debug=False, jupyter_height=1000)`.

The regular PyPI install uses the project's default `jax` dependency. GPU JAX
is platform/CUDA-specific; use the current JAX installation instructions
rather than assuming that the generic pip package enables a GPU. The existing
`deploy/uv/README.md` documents the project's tested CPU path and untested GPU
variant.

## Before the first release

1. PyPI's JSON endpoint returned HTTP 404 for both `chnet-xrf-visualizer` and
   `xrf-visualizer` on 2026-11-13, so neither project was listed then. This
   does not reserve the name; recheck immediately before publishing. The
   distribution name is now set in
   `pyproject.toml`; the source import remains `xrf_visualizer`, so the source
   directory does not need renaming.
2. MIT is now recorded in `LICENSE` and `pyproject.toml`. Before the first
   release, verify the copyright notice matches the actual rights holder and
   review licenses of any material added to the package. Apache-2.0 is an
   alternative if an explicit patent grant is important.
3. Confirm the project URLs and supported Python classifiers in
   `pyproject.toml`. Keep `requires-python = ">=3.12,<3.14"` unless CI and
   dependencies are verified for another range; do not invent an individual
   author/copyright holder if the project has not confirmed one.
4. Build wheel and source distribution using the existing Hatchling backend;
   inspect both archives to confirm `xrf_visualizer/data/XRFlines.json` ships
   with the package (verified in the current wheel and sdist). Run `twine
   check`, install the wheel in a clean venv, and smoke-test imports,
   `load_xrf_lines()`, `load_cube()` and `build_app()`.
5. Run the test suite on each supported Python version and verify dependency
   resolution with pip. The current runtime dependencies include JAX, NumPy,
   h5py, Dash, Plotly and Dash Bootstrap Components; there is no need to
   restructure the source tree or add a second package layout.

## Release workflow

1. CI tests Python 3.12/3.13 and builds wheel + sdist on pushes and pull
   requests. Release only from a reviewed `v<version>` tag matching
   `[project].version`; PyPI releases are immutable, so do not reuse a
   published version.
2. Rehearse the artifact on TestPyPI and install it in a clean environment.
3. Configure PyPI Trusted Publishing for owner `androbomb`, repository
   `dash-xrf-visualizer`, workflow `publish.yml`, and GitHub environment
   `pypi`. Use GitHub Actions OIDC and PyPA's supported publishing action
   instead of storing a long-lived PyPI token. See the
   [PyPI Trusted Publishing guide](https://docs.pypi.org/trusted-publishers/).
4. For each release, build and validate artifacts, create/push the version
   tag, then manually run the protected `Publish to PyPI` workflow for that
   tag. It is intentionally not triggered by an ordinary push or tag push.
   Verify the project page and install the exact version with pip.

The recommended build/upload flow is documented by the
[Python Packaging User Guide](https://packaging.python.org/en/latest/guides/tool-recommendations/).
Publishing is a separate, explicit release action; preparing this guide does
not upload anything to PyPI.
