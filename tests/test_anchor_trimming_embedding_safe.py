from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from facekit.pipeline.checkpoint import CheckpointManager

class _StubObs:
    def __init__(self, n: int):
        self._n = n
        self.trim_calls: list[int] = []
        self.trim_frame_calls: list[int] = []

        # pretend f increases 1:1 with row index
        self._arr = np.zeros((n,), dtype=[("f", "i4"), ("shot", "i4"), ("track_id", "i4"), ("src", "i4"), ("emb_idx", "i4")])
        self._arr["f"] = np.arange(n, dtype=np.int32)

    def count(self) -> int:
        return int(self._n)

    def load_npz(self, _path: Path) -> int:
        return int(self._n)

    def dump_npz(self, _path: Path) -> None:
        return

    def trim_to(self, n: int) -> None:
        self.trim_calls.append(int(n))
        self._n = int(n)
        self._arr = self._arr[: self._n]

    def to_array(self):
        return self._arr

    def trim_to_frame(self, frame_max: int) -> None:
        self.trim_frame_calls.append(int(frame_max))
        self._arr = self._arr[self._arr["f"] <= int(frame_max)]
        self._n = int(self._arr.shape[0])

class _StubEmb:
    def __init__(self, n: int):
        self._n = n
        self.trim_calls: list[int] = []
        self.trim_frame_calls: list[int] = []

    def count(self) -> int:
        return int(self._n)

    def load_npz(self, _path: Path) -> int:
        return int(self._n)

    def dump_npz(self, _path: Path) -> None:
        return

    def trim_to(self, n: int) -> None:
        self.trim_calls.append(int(n))
        self._n = int(n)

    def trim_to_frame(self, frame_max: int) -> None:
        self.trim_frame_calls.append(int(frame_max))

def _write_status(run_dir: Path, status: dict) -> None:
    (run_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    # create placeholder NPZs so resume_available() is true
    np.savez(run_dir / "ckpt" / "obs_ckpt.npz", observations=np.zeros((0,), dtype=[("f","i4")]))
    np.savez(run_dir / "ckpt" / "emb_ckpt.npz", embeddings=np.zeros((0, 512), dtype=np.float32))
    (run_dir / "status.json").write_text(json.dumps(status, indent=2))

def test_load_and_anchor_collectors_prefers_embedding_safe_anchor(tmp_path: Path):
    parent = tmp_path / "ckpt_parent"
    parent.mkdir()
    run_dir = parent / "run-000001"
    run_dir.mkdir()

    status = {
        "schema_version": "2.3",
        "video_path": "/tmp/dummy.mp4",
        # later DET checkpoint exists but is not safe
        "last_detection_frame": 180,
        "obs_rows_at_last_detection": 900,
        "emb_rows_at_last_detection": 800,
        # Design B (safe)
        "last_embedding_safe_frame": 170,
        "last_embedding_safe_shot_number": 2,
        "last_embedding_safe_shot_first_frame": 103,
        "obs_rows_at_last_embedding_safe": 700,
        "emb_rows_at_last_embedding_safe": 650,
        "track_order": [],
        "frames_done": 0,
        "shots_done": 0,
        "tracks_seen": 0,
        "detect_interval": 10,
        "embedding_batch_size_max": 32,
        "device": "cpu",
        "emb_store": "sidecar",
        "emb_sidecar_path": None,
        "obs_sidecar_path": None,
        "detector_model_path": "",
        "embedding_model_path": "",
        "yolo_config_path": "",
        "shot_segmentation_path": None,
        "checkpoint_dir": str(run_dir),
        "log_level": "INFO",
        "log_file": None,
    }
    _write_status(run_dir, status)

    dummy_video = tmp_path / "dummy.mp4"
    dummy_video.touch()

    mgr = CheckpointManager.open(
        parent_dir=parent,
        video_path=dummy_video,
        options_snapshot={"detect_interval": 10, "embedding_batch_size_max": 32},
        no_resume=False,
        resume_latest=True,
        force_new_run=False,
        write_disabled=True,
    )

    obs = _StubObs(1000)
    emb = _StubEmb(900)
    mgr.load_and_anchor_collectors(obs, emb, trim_to_anchor=True)

    # Should trim to embedding-safe counts, not detection counts.
    assert obs.trim_calls and obs.trim_calls[-1] == 700
    assert emb.trim_calls and emb.trim_calls[-1] == 650
    # Frame trim should use safe frame (keep strictly < 170)
    assert obs.trim_frame_calls and obs.trim_frame_calls[-1] == 169



def test_checkpoint_now_is_noop_when_write_disabled(tmp_path):
    dummy_video = tmp_path / "dummy.mp4"
    dummy_video.touch()

    run_dir = tmp_path / "run-000001"
    run_dir.mkdir()

    mgr = CheckpointManager(run_dir, video_path=dummy_video, resume=True, write_disabled=True)

    # Pretend collectors exist
    mgr._obs = _StubCollector(123)
    mgr._emb = _StubCollector(77)

    # Anchor fields should not change under write_disabled
    assert mgr._last_det_frame is None
    assert mgr._obs_rows_at_det == 0
    assert mgr._emb_rows_at_det == 0

    mgr.checkpoint_now(frame_idx=180, shot_number=2, aggregator=_StubAggregator(), shot_first_frame=103)

    # Still unchanged
    assert mgr._last_det_frame is None
    assert mgr._obs_rows_at_det == 0
    assert mgr._emb_rows_at_det == 0