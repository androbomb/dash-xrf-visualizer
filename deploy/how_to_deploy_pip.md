# Deploy with pip / PyPI

This is the pip-based alternative to the repository's `uv` workflow. It is a
deployment guide and release checklist. The project is published on PyPI
(first release `v0.1.0`); `pip install xrf-visualizer` works for every
version listed on the project page.

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

## Admin: cut a release (vX.Y.Z)

Both publish workflows are manual (`workflow_dispatch` with a required
`tag` input); an ordinary push or tag push never publishes. Each run
checks out the tag, verifies the tag matches `[project].version`, runs
`uv sync --locked` + `pytest -q`, builds wheel + sdist, runs `twine
check`, and smoke-tests the wheel in a clean venv before uploading.

### 1. Bump version and lock (locally)

```bash
# pyproject.toml: version = "X.Y.Z"
uv lock          # required: the publish gate runs `uv sync --locked`
uv run --locked pytest -q   # must be green
git add pyproject.toml uv.lock
git commit -m "release: bump to X.Y.Z"
```

A stale `uv.lock` (bump without `uv lock`) fails the publish workflow at
`uv sync --locked`.

### 2. Push + tag

```bash
git push origin master
git tag vX.Y.Z
git push origin vX.Y.Z
```

The tag must point at the commit whose `pyproject.toml` carries `X.Y.Z`;
the workflow fails otherwise.

### 3. TestPyPI first

GitHub → *Actions* → **Publish to TestPyPI** → *Run workflow* →
`tag: vX.Y.Z`. If it is red, stop — do not continue to PyPI.

Verify the artifact from outside. TestPyPI is the primary index, PyPI is
an extra index for the dependencies, and `unsafe-best-match` is needed
because the project also exists on PyPI (without it uv applies its
first-index-only rule and hides the TestPyPI version). This flag is safe
here: both indexes are official PyPA indexes.

```bash
uv venv /tmp/xrf-testpypi
uv pip install --python /tmp/xrf-testpypi/bin/python \
  --index-url https://test.pypi.org/simple \
  --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match \
  xrf-visualizer==X.Y.Z
/tmp/xrf-testpypi/bin/python -c "from xrf_visualizer.ui import build_app; build_app(None); print('ok')"
```

Index gotchas:

- `--index-url` alone *replaces* PyPI, so dependency resolution breaks
  (TestPyPI has no `setuptools` and only a stale `dash` rc).
- Plain pip-style two-index installs also need `--extra-index-url`
  `https://pypi.org/simple` for the same reason.

### 4. PyPI

GitHub → *Actions* → **Publish to PyPI** → *Run workflow* →
`tag: vX.Y.Z`. Publishing uses PyPI Trusted Publishing (OIDC, no token)
configured for owner `androbomb`, workflow `publish.yml`, GitHub
environment `pypi` — see the
[PyPI Trusted Publishing guide](https://docs.pypi.org/trusted-publishers/).
Then check the version is listed:

```bash
curl -s https://pypi.org/pypi/xrf-visualizer/json | grep -o '"X\.Y\.Z"'
```

### 5. GitHub Release (optional)

```bash
gh release create vX.Y.Z --title "vX.Y.Z" --notes "<release notes>"
```

### Pitfalls

- Published versions are immutable: a broken release means bumping to the
  next version and re-running from step 1 (a TestPyPI version can be
  yanked instead).
- Release only from a reviewed tag; never reuse a published version.

The recommended build/upload flow is documented by the
[Python Packaging User Guide](https://packaging.python.org/en/latest/guides/tool-recommendations/).
Publishing is a separate, explicit release action; preparing this guide does
not upload anything to PyPI.
