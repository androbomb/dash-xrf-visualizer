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
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from email.parser import BytesParser
from pathlib import Path
from urllib.parse import unquote, urlparse

import h5py
import numpy as np

__all__ = [
    "XrfCube",
    "load_cube",
    "load_cube_bytes",
    "download_cube",
    "sniff_format",
    "load_xrf_lines",
]

# Magic bytes -> extension, used when a file (e.g. a download link) has no
# usable extension.
_MAGIC = (
    (b"\x93NUMPY", ".npy"),
    (b"PK\x03\x04", ".npz"),
    (b"\x89HDF\r\n\x1a\n", ".h5"),
)


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


def sniff_format(data: bytes) -> str:
    """'.npy' | '.npz' | '.h5' from magic bytes; ValueError on unknown data."""
    for magic, ext in _MAGIC:
        if data[: len(magic)] == magic:
            return ext
    raise ValueError("unrecognized file format (expected .npy, .npz or HDF5 magic bytes)")


def load_cube_bytes(data: bytes, name: str = "") -> XrfCube:
    """Load a cube from in-memory bytes (uploads/downloads).

    Format comes from the name's extension when it is a known one, otherwise
    from magic-byte sniffing. Reuses :func:`load_cube` on a temp file, so the
    full schema validation applies.
    """
    suffix = Path(name).suffix.lower() if name else ""
    if suffix and suffix not in (".h5", ".hdf5", ".npz", ".npy"):
        raise ValueError(
            f"unsupported XRF file '{Path(name).name}' (expected .h5, .hdf5, .npz or .npy)"
        )
    if suffix not in (".h5", ".hdf5", ".npz", ".npy"):
        suffix = sniff_format(data)
    fd, tmp = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        cube = load_cube(tmp)
    finally:
        os.unlink(tmp)
    return XrfCube(Path(name) if name else Path("in-memory"), cube.array, cube.energy)


def _name_from_response(headers, url: str) -> str:
    cd = headers.get("Content-Disposition")
    if cd:
        msg = BytesParser().parsebytes(cd.encode())
        name = msg.get_filename()
        if name:
            return name
    return Path(unquote(urlparse(url).path)).name or ""


def download_cube(url: str, max_bytes: int = 500 * 1024 * 1024,
                  timeout: float = 60.0) -> XrfCube:
    """Download an XRF cube from an http(s) URL and validate it as a cube.

    Works with direct download links (Zenodo, ownCloud/nextCloud share links,
    ...). Authenticated/redirect-only links that return HTML instead of the
    file fail validation with a clear error.
    """
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError(f"only http(s) URLs are supported, got {url[:50]!r}")
    req = urllib.request.Request(url, headers={"User-Agent": "xrf-visualizer/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        name = _name_from_response(r.headers, url)
        data = r.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"download exceeds {max_bytes // (1024 * 1024)} MB limit")
    return load_cube_bytes(data, name)


def load_xrf_lines() -> list[dict[str, str | float]]:
    """CHNet XRF line energies: list of {"element": "<El> (Kx)", "value": keV}."""
    return json.loads((Path(__file__).parent / "XRFlines.json").read_text())