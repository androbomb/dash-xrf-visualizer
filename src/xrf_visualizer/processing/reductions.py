"""Plain float64 reductions over (H, W, C) XRF cubes (M3).

Contract pinned by tests/fixtures/golden.json, tests/test_processing.py and
scientific_notes.md:
- map = plain sum over channels [ch_start:ch_end); empty interval -> zero map.
- spectrum = plain sum over ROI pixels (y = row, x = col) or explicit points.
- accumulation is float64 regardless of storage dtype (real data is float32).
- out-of-range selections raise ValueError (the UI clamps before calling).

The CPU path is bit-exact vs the documented sequential float64 reference
(M3, all golden keys); shortcut: XLA's reduction order is not sequential in
general, so other shapes or a GPU backend may drift by a few ulp — the GPU
parity test is marked NOT TESTED (tests/test_processing.py).
"""
import jax
import jax.numpy as jnp
import numpy as np

from .backend import current_device

_F64 = jnp.float64


def _on_device(cube):
    return jax.device_put(jnp.asarray(cube), current_device())


def map_from_channels(cube, ch_start, ch_end):
    """Plain float64 sum of channels [ch_start:ch_end] -> (H, W) map."""
    c = cube.shape[-1]
    s, e = int(ch_start), int(ch_end)
    if not (0 <= s <= e <= c):
        raise ValueError(f"channel interval [{s}:{e}] outside [0:{c}]")
    a = _on_device(cube)[..., s:e]
    return np.asarray(jnp.sum(a, axis=-1, dtype=_F64))


def spectrum_from_roi(cube, y0, y1, x0, x1):
    """Plain float64 sum of ROI rows y [y0:y1), cols x [x0:x1) -> (C,) spectrum."""
    h, w = cube.shape[:2]
    y0, y1, x0, x1 = int(y0), int(y1), int(x0), int(x1)
    if not (0 <= y0 <= y1 <= h):
        raise ValueError(f"row range [{y0}:{y1}] outside [0:{h}]")
    if not (0 <= x0 <= x1 <= w):
        raise ValueError(f"col range [{x0}:{x1}] outside [0:{w}]")
    a = _on_device(cube)[y0:y1, x0:x1, :]
    return np.asarray(jnp.sum(a, axis=(0, 1), dtype=_F64))


def spectrum_from_points(cube, rows, cols):
    """Plain float64 sum of the given (row, col) pixels -> (C,) spectrum."""
    h, w, c = cube.shape
    if len(rows) != len(cols):
        raise ValueError(f"rows and cols must have equal length ({len(rows)} != {len(cols)})")
    rows = [int(r) for r in rows]
    cols = [int(x) for x in cols]
    if any(r < 0 or r >= h for r in rows) or any(x < 0 or x >= w for x in cols):
        raise ValueError("point out of cube bounds")
    if not rows:
        return np.zeros(c, dtype=np.float64)
    a = _on_device(cube)[np.asarray(rows), np.asarray(cols), :]
    return np.asarray(jnp.sum(a, axis=0, dtype=_F64))


def calibrate(n_channels, m, q):
    """Manual calibration axis E[i] = m*i + q (keV), float64, i in [0, n_channels)."""
    if m == 0:
        raise ValueError("m must be non-zero")
    return m * np.arange(n_channels, dtype=np.float64) + q


def energy_to_channel(energy_axis, energy):
    """Channel index on a calibrated energy_axis for a keV value.

    Nearest channel, within half a channel step; otherwise ValueError.
    """
    axis = np.asarray(energy_axis, dtype=np.float64)
    i = int(np.argmin(np.abs(axis - float(energy))))
    step = abs(axis[1] - axis[0]) if len(axis) > 1 else 1.0
    if abs(axis[i] - energy) > step / 2:
        raise ValueError(f"energy {energy} keV not on the calibrated axis")
    return i


def channel_range_for_energy(energy_axis, e_lo, e_hi):
    """Half-open channel range [s, e) covering [e_lo, e_hi] keV on the axis."""
    if e_lo > e_hi:
        raise ValueError(f"energy range [{e_lo}, {e_hi}] is inverted")
    s = energy_to_channel(energy_axis, e_lo)
    e = energy_to_channel(energy_axis, e_hi) + 1
    return (s, e)