"""Package metadata sanity: __version__ must match the installed distribution."""
import xrf_visualizer
from importlib.metadata import PackageNotFoundError, version


def test_version_matches_distribution():
    try:
        expected = version("xrf-visualizer")
    except PackageNotFoundError:
        assert xrf_visualizer.__version__.endswith("+unknown")
        return
    assert xrf_visualizer.__version__ == expected