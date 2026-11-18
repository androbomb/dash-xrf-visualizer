"""xrf_visualizer: Plotly Dash visualizer for XRF cubes (HDF5/.npz/.npy).

Subpackages: ``data`` (cube readers), ``processing`` (JAX/NumPy reductions),
``ui`` (Dash app factory).
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("xrf-visualizer")
except PackageNotFoundError:  # source checkout without an install
    __version__ = "0+unknown"