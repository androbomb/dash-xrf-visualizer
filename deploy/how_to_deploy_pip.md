# Deploy with pip / PyPI

This is the pip-based alternative to the repository's `uv` workflow. It is a
deployment guide and release checklist; **the project is not published on
PyPI yet**, so `pip install chnet-xrf-visualizer` will work only after the
first release.

## End-user install (after publication)

```bash
python -m venv .venv
# Linux/macOS
. .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install chnet-xrf-visualizer
python -m ipykernel install --user --name chnet-xrf \
  --display-name "Python (CHNet XRF visualizer)"
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

1. The PyPI JSON endpoint returned HTTP 404 for `chnet-xrf-visualizer` on
   2026-11-13, so no project was listed then. This does not reserve the name;
   recheck immediately before publishing. The distribution name is now set in
   `pyproject.toml`; the source import remains `xrf_visualizer`, so the source
   directory does not need renaming.
2. MIT is now recorded in `LICENSE` and `pyproject.toml`. Before the first
   release, verify the copyright notice matches the actual rights holder and
   review licenses of any material added to the package. Apache-2.0 is an
   alternative if an explicit patent grant is important.
3. Complete remaining package metadata (`authors`, project URLs, Python
   classifiers). Keep `requires-python = ">=3.12,<3.14"` unless CI and
   dependencies are verified for another range.
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

1. Add CI for tests and distribution builds. Release only from a reviewed Git
   tag whose version matches `[project].version`; PyPI releases are immutable,
   so do not reuse a published version.
2. Rehearse the artifact on TestPyPI and install it in a clean environment.
3. Configure PyPI Trusted Publishing for this GitHub repository/workflow and
   a protected `pypi` environment. Use GitHub Actions OIDC and PyPA's supported
   publishing action instead of storing a long-lived PyPI token. See the
   [PyPI Trusted Publishing guide](https://docs.pypi.org/trusted-publishers/).
4. For each release, build and validate artifacts, create/push the version
   tag, let the protected workflow publish, then verify the project page and
   install the exact version with pip.

The recommended build/upload flow is documented by the
[Python Packaging User Guide](https://packaging.python.org/en/latest/guides/tool-recommendations/).
Publishing is a separate, explicit release action; preparing this guide does
not upload anything to PyPI.
