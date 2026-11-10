"""Read XRF measurement cubes from .h5/.hdf5, .npz and .npy files.

Contract (see scientific_notes.md):
- cube: 3-D floating array, shape (H, W, C), channels last, stored as-is
  (no implicit keV conversion: an uncalibrated cube stays on the ADC axis)
- .h5/.hdf5 and .npy: top-level ``img`` dataset/array only
- .npz: key ``img`` plus an optional ``energy`` vector (keV), float64,
  strictly increasing, one value per channel

Readers never import Dash; numeric reductions live in ``processing`` (M3).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

__all__ = ["XrfCube", "load_cube", "load_xrf_lines"]


@dataclass(frozen=True)
class XrfCube:
    path: Path
    array: np.ndarray          # (H, W, C), floating dtype as stored
    energy: np.ndarray | None  # (C,) float64 keV, or None = channel (ADC) axis


def _validate_array(arr: np.ndarray, path: Path) -> None:
    if arr.ndim != 3:
        raise ValueError(f"{path.name}: expected a 3-D array (H, W, C), got shape {arr.shape}")
    if not np.issubdtype(arr.dtype, np.floating):
        raise ValueError(f"{path.name}: expected a floating dtype (counts), got {arr.dtype}")


def _validate_energy(energy: np.ndarray, n_channels: int, path: Path) -> np.ndarray:
    energy = np.asarray(energy, dtype=np.float64)
    if energy.ndim != 1 or energy.shape[0] != n_channels:
        raise ValueError(
            f"{path.name}: 'energy' must be a 1-D vector with one value per channel "
            f"({n_channels}), got shape {energy.shape}"
        )
    if not (np.all(np.isfinite(energy)) and np.all(np.diff(energy) > 0)):
        raise ValueError(f"{path.name}: 'energy' must be finite and strictly increasing")
    return energy


def load_cube(path: str | Path) -> XrfCube:
    """Load an XRF cube from .h5/.hdf5, .npz or .npy (key/dataset ``img``)."""
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".npz":
        with np.load(p, allow_pickle=False) as z:
            if "img" not in z:
                raise ValueError(f"{p.name}: missing 'img' key (found: {sorted(z.files)})")
            arr = z["img"]
            energy = _validate_energy(z["energy"], arr.shape[-1], p) if "energy" in z else None
        _validate_array(arr, p)
        return XrfCube(p, arr, energy)

    if suffix == ".npy":
        arr = np.load(p, allow_pickle=False)
        _validate_array(arr, p)
        return XrfCube(p, arr, None)

    if suffix in (".h5", ".hdf5"):
        with h5py.File(p, "r") as f:
            if "img" not in f:
                raise ValueError(f"{p.name}: missing 'img' dataset (found: {list(f.keys())})")
            ds = f["img"]
            if not isinstance(ds, h5py.Dataset):
                raise ValueError(f"{p.name}: 'img' must be a dataset, found {type(ds).__name__}")
            arr = np.asarray(ds[...])
        _validate_array(arr, p)
        return XrfCube(p, arr, None)

    raise ValueError(f"unsupported XRF file: {p.name} (expected .h5, .hdf5, .npz or .npy)")


def load_xrf_lines() -> list[dict[str, str | float]]:
    """CHNet XRF line energies: list of {"element": "<El> (Kx)", "value": keV}."""
    return json.loads((Path(__file__).parent / "XRFlines.json").read_text())