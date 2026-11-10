"""The three fixture encodings (.npz/.h5/.npy) must carry bit-identical arrays."""
from pathlib import Path

import h5py
import numpy as np

FIX = Path(__file__).parent / "fixtures"


def test_encodings_identical():
    a = np.load(FIX / "synthetic_cube.npz", allow_pickle=False)["img"]
    b = np.asarray(h5py.File(FIX / "synthetic_cube.h5", "r")["img"])
    c = np.load(FIX / "synthetic_cube.npy")
    assert a.shape == b.shape == c.shape == (48, 64, 512)
    assert a.dtype == b.dtype == c.dtype == np.float32
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, c)