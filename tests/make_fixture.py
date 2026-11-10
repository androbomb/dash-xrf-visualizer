"""Generate the deterministic synthetic XRF fixture and its golden values.

Run once:  .venv/bin/python tests/make_fixture.py
Regenerates tests/fixtures/synthetic_cube.{npz,h5,npy} and golden.json.

Design (closed form, no RNG):
- shape (H, W, C) = (48, 64, 512), float32 (real example data is float32).
- base = 1.0 + 0.001 * ((i*W + j) % 7)  -> spatially varying background
- region [10:20, 30:40] is doubled (non-square, so transposition bugs show)
- 3 Gaussian peaks at channels 100 (A=1000, s=2), 250 (A=500, s=3), 400 (A=250, s=1.5)
Golden reductions are computed in float64 (the documented reduction dtype).
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
FIX = HERE / "fixtures"


def make_cube() -> np.ndarray:
    H, W, C = 48, 64, 512
    i, j = np.mgrid[0:H, 0:W]          # (H, W)
    c = np.arange(C)                   # (C,)
    base = 1.0 + 0.001 * ((i * W + j) % 7)
    peaks = (1000.0 * np.exp(-(((c - 100.0) / 2.0) ** 2))
             + 500.0 * np.exp(-(((c - 250.0) / 3.0) ** 2))
             + 250.0 * np.exp(-(((c - 400.0) / 1.5) ** 2)))
    cube = base[:, :, None] + peaks[None, None, :]
    cube[10:20, 30:40, :] *= 2.0
    return cube.astype(np.float32)


def _seq_map(a64: np.ndarray, s: int, e: int) -> np.ndarray:
    """Sequential channel sum (the documented reference order)."""
    out = np.zeros(a64.shape[:2])
    for c in range(s, e):
        out += a64[:, :, c]
    return out


def _seq_spec(a64: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    """Sequential pixel sum, row-major (row, then column)."""
    out = np.zeros(a64.shape[-1])
    for h in range(y0, y1):
        for w in range(x0, x1):
            out += a64[h, w, :]
    return out


def golden(cube: np.ndarray) -> dict:
    a64 = cube.astype(np.float64)  # documented rule: reduce in float64
    H, W, C = a64.shape
    return {
        "shape": list(cube.shape),
        "dtype": str(cube.dtype),
        "map_full": _seq_map(a64, 0, C),
        "map_interval_100_120": _seq_map(a64, 100, 120),
        "map_single_channel_250": a64[:, :, 250].copy(),
        "map_empty_interval_50_50": _seq_map(a64, 50, 50),
        "spectrum_full": _seq_spec(a64, 0, H, 0, W),
        "spectrum_roi_10_20_30_40": _seq_spec(a64, 10, 20, 30, 40),
        "spectrum_single_pixel_10_10": a64[10, 10, :].copy(),
        "total": float(_seq_spec(a64, 0, H, 0, W).sum()),
    }


def main() -> None:
    FIX.mkdir(exist_ok=True)
    cube = make_cube()
    np.savez(FIX / "synthetic_cube.npz", img=cube)
    import h5py

    with h5py.File(FIX / "synthetic_cube.h5", "w") as f:
        f.create_dataset("img", data=cube)
    np.save(FIX / "synthetic_cube.npy", cube)
    g = golden(cube)
    out = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in g.items()}
    (FIX / "golden.json").write_text(json.dumps(out, indent=1))
    print("wrote", FIX)
    print("total counts (float64):", g["total"])


if __name__ == "__main__":
    main()