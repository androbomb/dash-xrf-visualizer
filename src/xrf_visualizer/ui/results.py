"""Elemental gallery results tied to a dataset revision (audit F11).

Compute is long (per-line JAX reductions) and runs on a Dash request
thread, so the user may load a different cube before it finishes. A result
must only be published if the dataset it was computed from is still
current: the record carries the server-side revision captured at Compute
start, and try_commit rejects it under the same short lock that load
commits use to bump the revision — check and assign cannot interleave with
a dataset swap. Once committed, the record is the single source for gallery
figures, spectra and the ZIP export: never a mix of browser metadata and a
mutable server array.
"""
from dataclasses import dataclass


@dataclass
class ResultRecord:
    """One Compute's output, stamped with the dataset it belongs to."""

    revision: int      # server-side revision captured at Compute start
    source_file: str   # dataset name at Compute start
    cube_shape: tuple  # (H, W, C) at Compute start
    calibration: dict  # m/q used for the computation
    metadata: list     # one entry per line, as returned to the UI
    maps: list         # one (H, W) float array per line
    spectrum: list     # full-field spectrum (C,)


def try_commit(state, lock, record) -> bool:
    """Commit ``record`` if the dataset revision is unchanged.

    ``lock`` is also held by load commits (revision bump + result reset),
    so check and assign cannot interleave with a dataset swap. Returns
    False and writes nothing when the record is stale.
    """
    with lock:
        if state["revision"] != record.revision:
            return False
        state["result"] = record
        return True