"""M6: in-memory cube loading (upload bytes) and http(s) link download."""
import http.server
import socketserver
import threading
from functools import partial
from pathlib import Path

import numpy as np
import pytest

from xrf_visualizer.data import (
    download_cube,
    load_cube,
    load_cube_bytes,
    sniff_format,
)

FIX = Path(__file__).parent / "fixtures"
NPZ = FIX / "synthetic_cube.npz"
NPy = FIX / "synthetic_cube.npy"
H5 = FIX / "synthetic_cube.h5"


def test_sniff_format():
    assert sniff_format(NPy.read_bytes()[:16]) == ".npy"
    assert sniff_format(NPZ.read_bytes()[:8]) == ".npz"
    assert sniff_format(H5.read_bytes()[:16]) == ".h5"
    with pytest.raises(ValueError, match="unrecognized"):
        sniff_format(b"this is not a cube file")


@pytest.mark.parametrize("src, name", [
    (NPZ, "cube.npz"), (NPy, "cube.npy"), (H5, "cube.h5"),
    (NPZ, ""), (NPy, ""), (H5, ""),  # nameless: magic-byte sniffing
])
def test_load_cube_bytes_matches_file(src, name):
    data = src.read_bytes()
    cube = load_cube_bytes(data, name)
    ref = load_cube(src)
    np.testing.assert_array_equal(cube.array, ref.array)
    np.testing.assert_array_equal(cube.energy, ref.energy)
    assert cube.path.name == (Path(name).name if name else "in-memory")


def test_load_cube_bytes_rejects_bad_extension():
    with pytest.raises(ValueError, match="unsupported XRF file"):
        load_cube_bytes(b"\x93NUMPY" + b"\x00" * 32, "cube.txt")
    with pytest.raises(ValueError, match="unsupported XRF file"):
        load_cube_bytes(b"PK\x03\x04" + b"\x00" * 32, "cube.zip")


def test_download_cube_rejects_non_http():
    with pytest.raises(ValueError, match="only http"):
        download_cube("ftp://example.com/cube.npz")
    with pytest.raises(ValueError, match="only http"):
        download_cube("file:///tmp/cube.npz")


def test_download_cube_bad_url():
    with pytest.raises(Exception):
        download_cube("http://127.0.0.1:1/definitely/closed.npz", timeout=2)


@pytest.fixture
def http_server():
    """Serve the fixtures dir on 127.0.0.1:<ephemeral port>."""
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(FIX))
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def test_download_cube_roundtrip(http_server):
    cube = download_cube(f"{http_server}/synthetic_cube.npz")
    ref = load_cube(NPZ)
    np.testing.assert_array_equal(cube.array, ref.array)
    assert cube.path.name == "synthetic_cube.npz"


def test_download_cube_invalid_content(http_server, tmp_path):
    (tmp_path / "index.html").write_text("<html>share page</html>")
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path))
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        with pytest.raises(ValueError):
            download_cube(f"http://127.0.0.1:{httpd.server_address[1]}/index.html")
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_download_cube_size_cap(http_server):
    # the fixture npz is a few KB; a 1024-byte cap must trip the limit
    with pytest.raises(ValueError, match="exceeds"):
        download_cube(f"{http_server}/synthetic_cube.npz", max_bytes=1024)