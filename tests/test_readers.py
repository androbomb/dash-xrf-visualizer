"""Reader tests: fixtures round-trip, schema, size limits, XRFlines, real data_example metadata."""
import http.server
import io
import socketserver
import subprocess
import sys
import threading
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pytest

from xrf_visualizer.data import (
    XrfCube,
    _SafeRedirectHandler,
    download_cube,
    load_cube,
    load_cube_bytes,
    load_xrf_lines,
)

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
    # F13: zero-length axes and non-3-D cubes must fail with ValueError
    ("bad.npy", lambda p: np.save(p, np.zeros((0, 5, 6), np.float32)), "positive"),
    ("bad.npy", lambda p: np.save(p, np.zeros((4, 5, 0), np.float32)), "positive"),
    ("bad.npz", lambda p: np.savez(p, img=np.zeros((4, 5, 0), np.float32)), "positive"),
    ("bad.npz", lambda p: np.savez(p, img=np.array(3.0), energy=np.array(1.0)), "3-D array"),
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


def test_download_cube_name_from_content_disposition(monkeypatch):
    # F14: the Content-Disposition filename must win over the URL path
    monkeypatch.setattr("xrf_visualizer.data._assert_public_url",
                        lambda url: None)  # F10 gate tested separately
    payload = (FIXTURES / "synthetic_cube.npz").read_bytes()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Disposition", 'attachment; filename="cube.npz"')
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            pass

    httpd = socketserver.TCPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        cube = download_cube(f"http://127.0.0.1:{httpd.server_address[1]}/download.php")
        assert cube.path.name == "cube.npz"
        assert cube.array.shape == (48, 64, C)
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_download_cube_refuses_non_public_destinations():
    # F10: the server-side fetch must not reach loopback/internal
    # addresses, so it is refused before any request leaves the machine.
    for url in ("http://127.0.0.1:8765/x.npz", "http://10.0.0.4/x.npz",
                "http://192.168.1.10/x.npz", "http://localhost/x.npz"):
        with pytest.raises(ValueError, match="non-public"):
            download_cube(url)


def test_download_redirect_to_non_public_is_rechecked():
    # F10: a public URL must not be able to redirect the fetch to an
    # internal address; every hop is re-checked.
    handler = _SafeRedirectHandler()
    req = urllib.request.Request("https://example.com/cube.npz")
    with pytest.raises(ValueError, match="non-public"):
        handler.redirect_request(req, None, 302, "Found", {},
                                 "http://127.0.0.1/x.npz")
    follow = handler.redirect_request(req, None, 302, "Found", {},
                                      "https://example.com/files/cube.npz")
    assert follow.full_url == "https://example.com/files/cube.npz"


# F12: expanded-cube size limit, checked on file metadata before allocation.


def _npy_header_only(shape, dtype="float32", version=(1, 0)):
    """A syntactically complete NPY header (magic+version+dict), zero data bytes.

    write_array_header_*_0 emits the magic itself, so the result is a valid
    file header followed by nothing: a real file would continue with the
    declared data, but the loader rejects it before ever reading data.
    """
    buf = io.BytesIO()
    header = {"descr": np.dtype(dtype).descr, "fortran_order": False, "shape": shape}
    if version >= (2, 0):
        np.lib.format.write_array_header_2_0(buf, header)
    else:
        np.lib.format.write_array_header_1_0(buf, header)
    return buf.getvalue()


HUGE = (1, 1, 1_000_000_000)  # float32 -> 4 GB declared, 0 bytes present


def test_npy_header_only_rejected_before_allocation(tmp_path):
    p = tmp_path / "huge.npy"
    p.write_bytes(_npy_header_only(HUGE))
    with pytest.raises(ValueError, match="exceeds"):
        load_cube(p)


def test_npy_v2_header_only_rejected(tmp_path):
    p = tmp_path / "huge.npy"
    p.write_bytes(_npy_header_only(HUGE, version=(2, 0)))
    with pytest.raises(ValueError, match="exceeds"):
        load_cube(p)


def test_npz_header_only_rejected_before_allocation(tmp_path):
    # the zeros case: a tiny file on disk declaring gigabytes when expanded
    p = tmp_path / "huge.npz"
    with zipfile.ZipFile(p, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("img.npy", _npy_header_only(HUGE))
    with pytest.raises(ValueError, match="exceeds"):
        load_cube(p)


def test_load_cube_bytes_header_only_rejected(tmp_path):
    # the upload path has no raw-byte cap, so the expanded check is its guard
    p = tmp_path / "huge.npz"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("img.npy", _npy_header_only(HUGE))
    with pytest.raises(ValueError, match="exceeds"):
        load_cube_bytes(p.read_bytes(), "huge.npz")


def test_max_bytes_parameter_npz():
    expanded = 48 * 64 * C * 4  # float32, synthetic_cube.npz
    load_cube(FIXTURES / "synthetic_cube.npz", max_bytes=expanded)
    with pytest.raises(ValueError, match="exceeds"):
        load_cube(FIXTURES / "synthetic_cube.npz", max_bytes=expanded - 1)


def test_max_bytes_parameter_h5():
    expanded = 48 * 64 * C * 4  # float32, synthetic_cube.h5
    load_cube(FIXTURES / "synthetic_cube.h5", max_bytes=expanded)
    with pytest.raises(ValueError, match="exceeds"):
        load_cube(FIXTURES / "synthetic_cube.h5", max_bytes=expanded - 1)
