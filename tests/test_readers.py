"""Reader tests: fixtures round-trip, schema validation, XRFlines, real data_example metadata."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from xrf_visualizer.data import XrfCube, load_cube, load_xrf_lines

FIXTURES = Path(__file__).parent / "fixtures"
DATA_EXAMPLE = Path(__file__).parent.parent / "data_example"
C = 512


def test_load_npz_fixture():
    cube = load_cube(FIXTURES / "synthetic_cube.npz")
    assert isinstance(cube, XrfCube)
    assert cube.array.shape == (48, 64, C)
    assert cube.array.dtype == np.float32
    assert cube.energy is None
    assert cube.path.name == "synthetic_cube.npz"


def test_load_h5_and_npy_match_npz():
    ref = load_cube(FIXTURES / "synthetic_cube.npz").array
    for name in ("synthetic_cube.h5", "synthetic_cube.npy"):
        cube = load_cube(FIXTURES / name)
        assert cube.array.dtype == np.float32
        assert cube.energy is None
        np.testing.assert_array_equal(cube.array, ref)


def test_npz_with_energy(tmp_path):
    src = np.load(FIXTURES / "synthetic_cube.npz", allow_pickle=False)
    energy = np.linspace(0.5, 38.5, C)
    p = tmp_path / "cal.npz"
    np.savez_compressed(p, img=src["img"], energy=energy)
    cube = load_cube(p)
    assert cube.energy is not None
    assert cube.energy.dtype == np.float64
    np.testing.assert_allclose(cube.energy, energy)


CASES = [
    ("bad.npy", lambda p: np.save(p, np.zeros((4, 5), np.float32)), "3-D array"),
    ("bad.npy", lambda p: np.save(p, np.zeros((4, 5, 6), np.int16)), "floating dtype"),
    ("bad.npz", lambda p: np.savez(p, wrong=np.zeros((4, 5, 6), np.float32)), "missing 'img'"),
    ("bad.npz", lambda p: np.savez(p, img=np.zeros((4, 5, 6), np.float32), energy=np.zeros(3)), "one value per channel"),
    (
        "bad.npz",
        lambda p: np.savez(p, img=np.zeros((4, 5, 6), np.float32),
                           energy=np.array([3.0, 1.0, 2.0, 0.5, 4.0, 9.0])),
        "strictly increasing",
    ),
    ("bad.txt", lambda p: p.write_text("not a cube"), "unsupported XRF file"),
]


@pytest.mark.parametrize("name,make,msg", CASES)
def test_validation_errors(tmp_path, name, make, msg):
    p = tmp_path / name
    make(p)
    with pytest.raises(ValueError, match=msg):
        load_cube(p)


def test_data_module_does_not_import_dash():
    # dash 4.5.0 registers a pytest plugin, so pytest itself imports dash;
    # the contract is that the data module alone must not, hence a fresh subprocess.
    code = "import sys, xrf_visualizer.data; assert 'dash' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, timeout=60)


def test_xrf_lines():
    lines = load_xrf_lines()
    assert len(lines) == 347
    assert all(isinstance(v["value"], float) and v["value"] > 0 for v in lines)
    by_el = {v["element"]: v["value"] for v in lines}
    assert by_el["Fe (Ka)"] == pytest.approx(6.405)
    assert by_el["Cu (Ka)"] == pytest.approx(8.046)


@pytest.mark.skipif(not DATA_EXAMPLE.exists(), reason="data_example/ is git-ignored and user-supplied")
def test_real_example_metadata():
    files = sorted(DATA_EXAMPLE.glob("*.npz"))
    assert files, "expected at least one .npz in data_example/"
    for p in files:
        cube = load_cube(p)
        assert cube.array.ndim == 3
        assert np.issubdtype(cube.array.dtype, np.floating)
        assert cube.energy is None  # no stored energy axis in the example files