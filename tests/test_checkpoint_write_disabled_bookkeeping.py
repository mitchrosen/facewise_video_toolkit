from __future__ import annotations
from pathlib import Path
import json

import pytest

import facekit.pipeline.checkpoint as checkpoint_mod
from facekit.pipeline.checkpoint import CheckpointManager


class _StubCollector:
    def __init__(self, n: int) -> None:
        self._n = int(n)

    def count(self) -> int:
        return int(self._n)

    def dump_npz(self, path: Path) -> None:
        # This should never be called in write-disabled tests.
        raise AssertionError("dump_npz must not be called when write_disabled=True")


def test_flush_commit_is_noop_when_write_disabled(monkeypatch, tmp_path: Path) -> None:
    """
    Spine: mainline may notify checkpoint that embeddings are safe up to a frame,
    but checkpoint write is strictly disabled under --no-checkpoint-write.
    """
    run_dir = tmp_path / "run-000001"
    # Intentionally do NOT create any directories here.
    dummy_video = tmp_path / "dummy.mp4"
    dummy_video.write_bytes(b"")

    mgr = CheckpointManager(run_dir, video_path=dummy_video, resume=True, write_disabled=True)
    mgr._obs = _StubCollector(123)
    mgr._emb = _StubCollector(77)

    called_dump = {"n": 0}
    called_status = {"n": 0}

    def _fake_dump_npz_atomic(_collector, _final_path):
        called_dump["n"] += 1

    def _fake_atomic_write_text(_dst, _text):
        called_status["n"] += 1

    monkeypatch.setattr(checkpoint_mod, "_dump_npz_atomic", _fake_dump_npz_atomic)
    monkeypatch.setattr(checkpoint_mod, "_atomic_write_text", _fake_atomic_write_text)

    mgr.mark_embedding_safe(frame_idx=180, shot_number=2, shot_first_frame=103, note="test")

    assert called_dump["n"] == 0
    assert called_status["n"] == 0


def test_flush_commit_writes_sidecars_and_status_when_enabled(monkeypatch, tmp_path: Path) -> None:
    """
    Spine: when checkpoint writing is enabled, marking the embedding-safe anchor
    makes the checkpoint durable.
    """
    run_dir = tmp_path / "run-000001"
    run_dir.mkdir(parents=True, exist_ok=True)
    dummy_video = tmp_path / "dummy.mp4"
    dummy_video.write_bytes(b"")

    mgr = CheckpointManager(run_dir, video_path=dummy_video, resume=True, write_disabled=False)

    class _OKCollector(_StubCollector):
        def dump_npz(self, path: Path) -> None:
            # write a minimal file so fs checks don’t explode if someone inspects it later
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"npz")

    mgr._obs = _OKCollector(123)
    mgr._emb = _OKCollector(77)

    called_dump = {"n": 0}

    def _fake_dump_npz_atomic(_collector, _final_path):
        called_dump["n"] += 1

    monkeypatch.setattr(checkpoint_mod, "_dump_npz_atomic", _fake_dump_npz_atomic)

    mgr.mark_embedding_safe(frame_idx=180, shot_number=2, shot_first_frame=103, note="test")

    # Two sidecars: obs + emb
    assert called_dump["n"] == 2
    assert mgr.status_path.exists()

    st = json.loads(mgr.status_path.read_text())
    assert st["last_embedding_safe_frame"] == 180