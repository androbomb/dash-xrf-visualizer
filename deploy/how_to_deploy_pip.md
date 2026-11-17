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
  --no-cache \
  xrf-visualizer==X.Y.Z
/tmp/xrf-testpypi/bin/python -c "from xrf_visualizer.ui import build_app; build_app(None); print('ok')"
```

Index gotchas:

- `--index-url` alone *replaces* PyPI, so dependency resolution breaks
  (TestPyPI has no `setuptools` and only a stale `dash` rc).
- Plain pip-style two-index installs also need `--extra-index-url`
  `https://pypi.org/simple` for the same reason.
- A just-published version can still be "not found" even when it is on
  the TestPyPI page: uv caches index listings in a *global* cache
  (`~/.cache/uv`, not per-venv) left by the previous release's install.
  Keep `--no-cache` (or `uv cache clean` once).

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

Without `gh`: repo → *Releases* → **Draft a new release** → pick the
existing tag `vX.Y.Z`.

### Pitfalls

- Published versions are immutable: a broken release means bumping to the
  next version and re-running from step 1 (a TestPyPI version can be
  yanked instead).
- Release only from a reviewed tag; never reuse a published version.

The recommended build/upload flow is documented by the
[Python Packaging User Guide](https://packaging.python.org/en/latest/guides/tool-recommendations/).
Publishing is a separate, explicit release action; preparing this guide does
not upload anything to PyPI.
