from __future__ import annotations
from pathlib import Path

from facekit.pipeline.checkpoint import CheckpointManager

class _StubCollector:
    def __init__(self, n: int):
        self._n = n
        self.dump_calls = 0

    def count(self) -> int:
        return int(self._n)

    def dump_npz(self, _path: Path) -> None:
        self.dump_calls += 1

class _StubAggregator:
    tracks = []

def test_checkpoint_now_updates_anchor_bookkeeping_even_when_write_disabled(monkeypatch, tmp_path: Path):
    run_dir = tmp_path / "run-000001"
    run_dir.mkdir()
    dummy_video = tmp_path / "dummy.mp4"
    dummy_video.touch()

    mgr = CheckpointManager(run_dir, video_path=dummy_video, resume=True, write_disabled=True)
    mgr._obs = _StubCollector(123)
    mgr._emb = _StubCollector(77)

    # Spy: ensure no disk writes are attempted
    called_dump = {"n": 0}
    called_status = {"n": 0}

    def _fake_dump_npz_atomic(_collector, _final_path):
        called_dump["n"] += 1

    def _fake_atomic_write_text(_dst, _text):
        called_status["n"] += 1

    monkeypatch.setattr("facekit.pipeline.checkpoint._dump_npz_atomic", _fake_dump_npz_atomic)
    monkeypatch.setattr("facekit.pipeline.checkpoint._atomic_write_text", _fake_atomic_write_text)

    mgr.checkpoint_now(
        frame_idx=180,
        shot_number=2,
        aggregator=_StubAggregator(),
        shot_first_frame=103,
        note="checkpoint",
    )

    # In-memory bookkeeping should be updated
    assert mgr._last_det_frame == 180
    assert mgr._obs_rows_at_det == 123
    assert mgr._emb_rows_at_det == 77

    # But no disk writes should occur
    assert called_dump["n"] == 0
    assert called_status["n"] == 0