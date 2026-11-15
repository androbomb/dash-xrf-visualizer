"""F11: revision-gated result records (unit level, no Dash app).

try_commit is the gate between a long Compute and the dataset it belongs
to: loads bump the server-side revision under the same lock, so a record
computed from an older revision is rejected instead of published."""
import threading

from xrf_visualizer.ui.results import ResultRecord, try_commit


def _state(revision=1, result=None):
    return {"revision": revision, "result": result}


def _record(revision=1):
    return ResultRecord(revision=revision, source_file="a.npz", cube_shape=(2, 3, 4),
                        calibration={"m": 0.1, "q": 0.2},
                        metadata=[{"label": "Fe (Ka)"}], maps=[[[1.0]]],
                        spectrum=[1.0])


def test_commit_fresh_succeeds():
    state, lock = _state(), threading.Lock()
    assert try_commit(state, lock, _record(revision=1)) is True
    assert state["result"].source_file == "a.npz"
    assert state["result"].cube_shape == (2, 3, 4)


def test_commit_stale_rejected():
    state, lock = _state(revision=1), threading.Lock()
    state["revision"] = 2  # a load committed while compute was running
    assert try_commit(state, lock, _record(revision=1)) is False
    assert state["result"] is None


def test_stale_commit_cannot_survive_concurrent_load():
    # Real concurrency: whichever of the two critical sections runs first,
    # a stale record must never end up as the committed result.
    for _ in range(200):
        state, lock = _state(revision=1), threading.Lock()

        def load():
            with lock:
                state["revision"] += 1
                state["result"] = None

        def compute():
            try_commit(state, lock, _record(revision=1))

        t1, t2 = threading.Thread(target=load), threading.Thread(target=compute)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert state["result"] is None