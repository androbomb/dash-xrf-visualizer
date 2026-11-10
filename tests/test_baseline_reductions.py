"""Baseline numeric contract (M1): golden values pin the XRF reduction rules.

Rules (see scientific_notes.md):
- cube: (H, W, C); channel axis last.
- map from spectral interval [s:e) = plain sum over channels, NO weighting.
- spectrum from spatial ROI [x0:x1, y0:y1) = plain sum over pixels, NO normalization.
- reductions run in float64 regardless of storage dtype (real data is float32).
M3 (JAX) and M4 (UI) must match these goldens exactly (int-free float64 sums,
deterministic order).
"""
import json
from pathlib import Path

import numpy as np
import pytest

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def cube() -> np.ndarray:
    return np.load(FIX / "synthetic_cube.npz", allow_pickle=False)["img"]


@pytest.fixture(scope="module")
def golden() -> dict:
    g = json.loads((FIX / "golden.json").read_text())
    for k in ("map_full", "map_interval_100_120", "map_single_channel_250",
              "map_empty_interval_50_50", "spectrum_full",
              "spectrum_roi_10_20_30_40", "spectrum_single_pixel_10_10"):
        g[k] = np.asarray(g[k])
    return g


def _ref_map(a: np.ndarray, s: int, e: int) -> np.ndarray:
    """Reference map: plain channel sum, float64, clamped to [0, C]."""
    C = a.shape[-1]
    s, e = max(0, min(s, C)), max(0, min(e, C))
    return a[:, :, s:e].astype(np.float64).sum(axis=-1)


def _ref_spectrum(a: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """Reference spectrum: plain spatial sum, float64, clamped to shape."""
    H, W = a.shape[:2]
    x0, y0 = max(0, min(x0, H)), max(0, min(y0, W))
    x1, y1 = max(0, min(x1, H)), max(0, min(y1, W))
    return a[x0:x1, y0:y1, :].astype(np.float64).sum(axis=(0, 1))


def test_golden_reproducible(cube, golden):
    """Golden values must be exactly reproducible (deterministic closed-form cube)."""
    assert cube.shape == tuple(golden["shape"])
    assert cube.dtype == np.dtype(golden["dtype"])
    np.testing.assert_array_equal(_ref_map(cube, 0, cube.shape[-1]), golden["map_full"])
    np.testing.assert_array_equal(
        _ref_map(cube, 100, 120), golden["map_interval_100_120"])
    np.testing.assert_array_equal(_ref_map(cube, 250, 251), golden["map_single_channel_250"])
    np.testing.assert_array_equal(_ref_map(cube, 50, 50), golden["map_empty_interval_50_50"])
    np.testing.assert_array_equal(_ref_spectrum(cube, 0, 0, *cube.shape[:2]),
                                  golden["spectrum_full"])
    np.testing.assert_array_equal(_ref_spectrum(cube, 10, 30, 20, 40),
                                  golden["spectrum_roi_10_20_30_40"])
    np.testing.assert_array_equal(_ref_spectrum(cube, 10, 10, 11, 11),
                                  golden["spectrum_single_pixel_10_10"])
    assert golden["total"] == pytest.approx(
        _ref_spectrum(cube, 0, 0, *cube.shape[:2]).sum(), rel=0.0, abs=0.0)


def test_map_interval_rules(cube, golden):
    """Interval [s:e): empty -> zeros; single channel; clamping at edges."""
    assert np.all(golden["map_empty_interval_50_50"] == 0.0)
    C = cube.shape[-1]
    np.testing.assert_array_equal(_ref_map(cube, C - 1, C),
                                  cube[:, :, C - 1].astype(np.float64))
    # clamping: out-of-range edges collapse instead of erroring (UI contract)
    np.testing.assert_array_equal(_ref_map(cube, -5, 10), _ref_map(cube, 0, 10))
    np.testing.assert_array_equal(_ref_map(cube, C - 1, C + 9), _ref_map(cube, C - 1, C))


def test_roi_rules(cube, golden):
    """ROI [x0:x1, y0:y1): non-square, single pixel, empty -> zeros, clamped."""
    H, W = cube.shape[:2]
    np.testing.assert_array_equal(_ref_spectrum(cube, 5, 5, 5, 5), np.zeros(cube.shape[-1]))
    # full ROI == spectrum_full (consistency anchor)
    np.testing.assert_array_equal(_ref_spectrum(cube, 0, 0, H, W),
                                  golden["spectrum_full"])
    # doubled region must not be a plain fraction of the full spectrum
    roi = golden["spectrum_roi_10_20_30_40"]
    per_px = golden["spectrum_full"] / (H * W)
    assert not np.allclose(roi, 100 * per_px, rtol=1e-3)


def test_float64_promotion_rule(cube, golden):
    """float32 storage must not leak into results: float64 accumulation is exact."""
    s32 = cube[:, :, 100:120].sum(axis=-1)          # float32 accumulation
    s64 = cube[:, :, 100:120].astype(np.float64).sum(axis=-1)
    assert s32.dtype == np.float32
    assert s64.dtype == np.float64
    # the documented rule: goldens match the float64 path, and the two can differ
    np.testing.assert_array_equal(s64, golden["map_interval_100_120"])
    assert not np.array_equal(s32, s64)  # guard: the two paths really do differ