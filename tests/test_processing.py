"""M3: JAX reductions match the goldens exactly; device policy; energy helpers."""
import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

from xrf_visualizer.processing import (
    backend_info,
    calibrate,
    channel_range_for_energy,
    current_device,
    energy_to_channel,
    map_from_channels,
    require_backend,
    spectrum_from_points,
    spectrum_from_roi,
)

FIX = Path(__file__).parent / "fixtures"
ROOT = Path(__file__).parent.parent

GOLDEN_KEYS = ("map_full", "map_interval_100_120", "map_single_channel_250",
               "map_empty_interval_50_50", "spectrum_full",
               "spectrum_roi_10_20_30_40", "spectrum_single_pixel_10_10")


@pytest.fixture(scope="module")
def cube() -> np.ndarray:
    return np.load(FIX / "synthetic_cube.npz", allow_pickle=False)["img"]


@pytest.fixture(scope="module")
def golden() -> dict:
    g = json.loads((FIX / "golden.json").read_text())
    for k in GOLDEN_KEYS:
        g[k] = np.asarray(g[k])
    return g


def test_jax_matches_golden_exactly(cube, golden):
    H, W, C = cube.shape
    np.testing.assert_array_equal(map_from_channels(cube, 0, C), golden["map_full"])
    np.testing.assert_array_equal(map_from_channels(cube, 100, 120),
                                  golden["map_interval_100_120"])
    np.testing.assert_array_equal(map_from_channels(cube, 250, 251),
                                  golden["map_single_channel_250"])
    np.testing.assert_array_equal(map_from_channels(cube, 50, 50),
                                  golden["map_empty_interval_50_50"])
    np.testing.assert_array_equal(spectrum_from_roi(cube, y0=0, y1=H, x0=0, x1=W),
                                  golden["spectrum_full"])
    np.testing.assert_array_equal(spectrum_from_roi(cube, y0=10, y1=20, x0=30, x1=40),
                                  golden["spectrum_roi_10_20_30_40"])
    np.testing.assert_array_equal(spectrum_from_roi(cube, y0=10, y1=11, x0=10, x1=11),
                                  golden["spectrum_single_pixel_10_10"])


def test_results_are_float64(cube):
    assert map_from_channels(cube, 0, cube.shape[-1]).dtype == np.float64
    assert spectrum_from_roi(cube, y0=0, y1=1, x0=0, x1=1).dtype == np.float64


def test_out_of_range_raises_value_error(cube):
    H, W, C = cube.shape
    with pytest.raises(ValueError):
        map_from_channels(cube, -1, 10)
    with pytest.raises(ValueError):
        map_from_channels(cube, 0, C + 1)
    with pytest.raises(ValueError):
        map_from_channels(cube, 5, 4)
    with pytest.raises(ValueError):
        spectrum_from_roi(cube, y0=0, y1=H + 1, x0=0, x1=W)
    with pytest.raises(ValueError):
        spectrum_from_roi(cube, y0=-1, y1=5, x0=0, x1=5)
    with pytest.raises(ValueError):
        spectrum_from_points(cube, [0], [])
    with pytest.raises(ValueError):
        spectrum_from_points(cube, [H], [0])


def test_points_rules(cube):
    C = cube.shape[-1]
    a64 = cube.astype(np.float64)
    ref = np.zeros(C)
    for r, c in [(10, 30), (19, 39)]:
        ref += a64[r, c, :]
    np.testing.assert_array_equal(spectrum_from_points(cube, [10, 19], [30, 39]), ref)
    np.testing.assert_array_equal(spectrum_from_points(cube, [], []), np.zeros(C))
    np.testing.assert_array_equal(
        spectrum_from_points(cube, [7], [5]),
        spectrum_from_roi(cube, y0=7, y1=8, x0=5, x1=6))


def test_calibrate_and_energy_helpers():
    C = 512
    m, q = 0.0746, 0.5
    e = calibrate(C, m, q)
    assert e.dtype == np.float64
    np.testing.assert_allclose(e[0], q)
    np.testing.assert_allclose(e[-1], q + m * (C - 1))
    assert energy_to_channel(e, q + m * 250) == 250
    assert channel_range_for_energy(e, e[0], e[-1]) == (0, C)
    assert channel_range_for_energy(e, e[3], e[7]) == (3, 8)
    with pytest.raises(ValueError):
        calibrate(C, 0.0, q)
    with pytest.raises(ValueError):
        energy_to_channel(e, e[-1] + 1.0)
    with pytest.raises(ValueError):
        channel_range_for_energy(e, 2.0, 1.0)
    with pytest.raises(ValueError):
        channel_range_for_energy(e, -0.1, 1.0)


def test_no_dash_import():
    code = "import sys, xrf_visualizer.processing; assert 'dash' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, timeout=60)


def test_backend_reporting():
    info = backend_info()
    assert info["platform"] == current_device().platform
    assert info["platform"] in {"cpu", "cuda", "gpu", "tpu"}
    if info["platform"] != "cuda":
        with pytest.raises(RuntimeError, match="not available"):
            require_backend("cuda")


def _hex_map(platform_env: str) -> str:
    snippet = (
        "import numpy as np\n"
        "from xrf_visualizer.data import load_cube\n"
        "from xrf_visualizer.processing import map_from_channels\n"
        f"c = load_cube({str(FIX / 'synthetic_cube.npz')!r}).array\n"
        "print(map_from_channels(c, 0, c.shape[-1]).tobytes().hex())\n"
    )
    env = {**os.environ, "JAX_PLATFORM_NAME": platform_env}
    r = subprocess.run([sys.executable, "-c", snippet], env=env, cwd=ROOT,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip().splitlines()[-1]


HAS_CUDA = any(d.platform == "cuda" for d in jax.devices())


@pytest.mark.skipif(not HAS_CUDA,
                    reason="NOT TESTED: needs a CUDA JAX build (deploy/uv/README.md)")
def test_gpu_agrees_bitwise_with_cpu():
    """GPU/CPU must agree bit-exactly (sequential order + float64, x64 enabled)."""
    assert _hex_map("cuda") == _hex_map("cpu")