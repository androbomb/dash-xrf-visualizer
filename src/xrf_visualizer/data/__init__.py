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
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

import h5py
import numpy as np

__all__ = [
    "MAX_CUBE_BYTES",
    "XrfCube",
    "load_cube",
    "load_cube_bytes",
    "download_cube",
    "sniff_format",
    "validate_cube_array",
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


def validate_cube_array(arr: np.ndarray, label: str) -> None:
    """3-D floating cube with strictly positive axes; ValueError otherwise.
    Shape is checked before anything indexed by shape (e.g. the energy axis),
    so a scalar 'img' fails with ValueError, not IndexError."""
    if arr.ndim != 3:
        raise ValueError(f"{label}: expected a 3-D array (H, W, C), got shape {arr.shape}")
    if any(size == 0 for size in arr.shape):
        raise ValueError(f"{label}: all axes must be positive, got shape {arr.shape}")
    if not np.issubdtype(arr.dtype, np.floating):
        raise ValueError(f"{label}: expected a floating dtype (counts), got {arr.dtype}")


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


MAX_CUBE_BYTES = 1024 ** 3  # expanded-cube cap (bytes); default for all readers


def _npy_expanded_bytes(f) -> int:
    """Expanded byte count of an NPY stream, from its header only (no data read).

    Works on any binary file-like (plain file or a .npz member via ZipExtFile),
    so a cube whose header declares gigabytes (e.g. compressed zeros) is
    rejected before any allocation.
    """
    major, minor = np.lib.format.read_magic(f)
    read_header = (np.lib.format.read_array_header_2_0 if (major, minor) >= (2, 0)
                   else np.lib.format.read_array_header_1_0)
    shape, _fortran, dtype = read_header(f)
    return int(np.prod(shape, dtype=np.int64)) * dtype.itemsize


def _require_within_budget(expanded: int, name: str, max_bytes: int) -> None:
    if expanded > max_bytes:
        raise ValueError(
            f"{name}: expanded cube is ~{expanded // (1024 * 1024)} MB, exceeds "
            f"the {max_bytes // (1024 * 1024)} MB limit")


def load_cube(path: str | Path, max_bytes: int = MAX_CUBE_BYTES) -> XrfCube:
    """Load an XRF cube from .h5/.hdf5, .npz or .npy (key/dataset ``img``).

    The expanded cube must fit ``max_bytes`` (default MAX_CUBE_BYTES); the
    check runs on file metadata (NPY/NPZ array header, HDF5 shape/dtype)
    before any array is materialized, so a tiny file that expands huge
    (e.g. compressed zeros) is rejected without allocating it.
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".npz":
        with zipfile.ZipFile(p) as zf:
            if "img.npy" not in zf.namelist():
                raise ValueError(f"{p.name}: missing 'img' key (found: {sorted(zf.namelist())})")
            with zf.open("img.npy") as img_f:
                _require_within_budget(_npy_expanded_bytes(img_f), p.name, max_bytes)
        with np.load(p, allow_pickle=False) as z:
            arr = z["img"]
            # validate shape/axes before touching arr.shape[-1] (scalar img)
            validate_cube_array(arr, p.name)
            energy = _validate_energy(z["energy"], arr.shape[-1], p) if "energy" in z else None
        return XrfCube(p, arr, energy)

    if suffix == ".npy":
        with open(p, "rb") as f:
            _require_within_budget(_npy_expanded_bytes(f), p.name, max_bytes)
        arr = np.load(p, allow_pickle=False)
        validate_cube_array(arr, p.name)
        return XrfCube(p, arr, None)

    if suffix in (".h5", ".hdf5"):
        with h5py.File(p, "r") as f:
            if "img" not in f:
                raise ValueError(f"{p.name}: missing 'img' dataset (found: {list(f.keys())})")
            ds = f["img"]
            if not isinstance(ds, h5py.Dataset):
                raise ValueError(f"{p.name}: 'img' must be a dataset, found {type(ds).__name__}")
            _require_within_budget(
                int(np.prod(ds.shape, dtype=np.int64)) * ds.dtype.itemsize, p.name, max_bytes)
            arr = np.asarray(ds[...])
        validate_cube_array(arr, p.name)
        return XrfCube(p, arr, None)

    raise ValueError(f"unsupported XRF file: {p.name} (expected .h5, .hdf5, .npz or .npy)")


def sniff_format(data: bytes) -> str:
    """'.npy' | '.npz' | '.h5' from magic bytes; ValueError on unknown data."""
    for magic, ext in _MAGIC:
        if data[: len(magic)] == magic:
            return ext
    raise ValueError("unrecognized file format (expected .npy, .npz or HDF5 magic bytes)")


def load_cube_bytes(data: bytes, name: str = "", max_bytes: int = MAX_CUBE_BYTES) -> XrfCube:
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
        cube = load_cube(tmp, max_bytes)
    finally:
        os.unlink(tmp)
    return XrfCube(Path(name) if name else Path("in-memory"), cube.array, cube.energy)


def _name_from_response(headers, url: str) -> str:
    # response headers are an email.message: get_filename() handles both
    # filename and RFC 5987 filename*; parsing the raw header line parsed
    # nothing and the name was always lost to the URL fallback.
    name = headers.get_filename()
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