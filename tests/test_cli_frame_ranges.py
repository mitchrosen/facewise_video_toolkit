# tests/test_cli_frame_range_resume.py

from __future__ import annotations

import json
import types
from pathlib import Path
import inspect
import pytest

from facekit.cli import resolve_face_ids_v2_cli as cli
from facekit.errors import ResumeSafetyError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeReader:
    def __init__(self, path: str, total_frames: int = 100):
        self.path = path
        self._fps = 30.0
        self._size = (640, 360)
        self._total_frames = total_frames

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def fps(self):
        return self._fps

    def size(self):
        return self._size

    def total_frames(self):
        return self._total_frames

class _FakeEmbedder:
    def __init__(self, *args, **kwargs):
        self.max_batch_size = None

    def set_max_batch_size(self, n: int):
        self.max_batch_size = n

class _FakeCkpt:
    def __init__(
        self,
        *,
        root: Path,
        resume_enabled: bool = False,
        validate_exc: Exception | None = None,
        anchor_frame: int | None = None,
    ):
        self.root = root
        self.resume_enabled = resume_enabled
        self._validate_exc = validate_exc
        self._anchor_frame = anchor_frame

        self.validate_called = 0
        self.start_called = 0
        self.rehydrate_called = 0
        self.finalize_called = 0
        self.mark_completed_called = 0

    def read_status(self):
        return {}

    def validate_resume_or_raise(self, *args, **kwargs):
        self.validate_called += 1
        if self._validate_exc is not None:
            raise self._validate_exc

    def start(self, *args, **kwargs):
        self.start_called += 1

    def rehydrate_runtime(self, *args, **kwargs):
        self.rehydrate_called += 1
        if self._anchor_frame is None:
            return {}
        return {
            "anchor_frame": self._anchor_frame,
            "anchor_shot": 1,
            "anchor_shot_first_frame": 0,
            "track_order_entries": 0,
        }

    def finalize(self):
        self.finalize_called += 1

    def copy_ckpt_sidecars_to_final(self, *args, **kwargs):
        return None

    def mark_completed(self):
        self.mark_completed_called += 1

def _write_minimal_shots(path: Path, *, first_frame: int = 0, last_frame: int = 99) -> None:
    payload = {
        "shots": [
            {
                "shot_number": 1,
                "first_frame": first_frame,
                "last_frame": last_frame,
            }
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

def _base_args(tmp_path: Path, **overrides):
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"FAKE")

    shots_path = tmp_path / "shots.json"
    _write_minimal_shots(shots_path, last_frame=99)

    args = types.SimpleNamespace(
        input=str(video_path),
        detector_model="models/detector/yolov5n_state_dict.pt",
        embedding_model="models/embedding/glintr100_dynamic.onnx",
        config="models/detector/yolov5n.yaml",
        min_face=10,
        shot_segmentation=str(shots_path),
        schema_version="2.1",
        schema_dir=None,
        output_segment_json=None,
        output_global_json=None,
        output_video=None,
        device="auto",
        detect_interval=30,
        embedding_queue_max_pending=1024,
        embedding_batch_size_max=32,
        track_sample_interval=4,
        post_min_gap_len=210,
        post_min_track_len=70,
        post_iou_threshold=0.2,
        emb_store="sidecar",
        emb_sidecar_path=None,
        obs_sidecar_path=None,
        checkpoint_dir=str(tmp_path / "ckpt_parent"),
        no_resume=False,
        force=False,
        resume_latest=False,
        checkpoint_run_id=None,
        new_run=False,
        no_checkpoint_write=True,
        log="INFO",
        log_file=None,
        # NEW CONTRACT
        start_frame=0,
        end_frame=None,
    )

    for k, v in overrides.items():
        setattr(args, k, v)

    return args

def _patch_minimal_pipeline(monkeypatch, tmp_path: Path, *, ckpt: _FakeCkpt, total_frames: int = 100, track_impl=None):
    monkeypatch.setattr(cli, "ReaderCoordinator", lambda p: _FakeReader(p, total_frames=total_frames))
    monkeypatch.setattr(cli, "load_yolo5face_model", lambda *a, **k: object())
    monkeypatch.setattr(cli, "FaceDetector", lambda yolo: object())
    monkeypatch.setattr(cli, "FaceEmbedder", lambda *a, **k: _FakeEmbedder())
    monkeypatch.setattr(cli.CheckpointManager, "compute_parent_dir", lambda *a, **k: tmp_path / "computed_ckpts")
    monkeypatch.setattr(cli.CheckpointManager, "open", lambda *a, **k: ckpt)

    if track_impl is None:
        monkeypatch.setattr(
            cli.track_across_segments,
            "track_across_segments",
            lambda **kwargs: [],
        )
    else:
        monkeypatch.setattr(
            cli.track_across_segments,
            "track_across_segments",
            track_impl,
        )

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_start_frame_defaults_to_none(monkeypatch):
    """
    CLI contract:
      --start-frame exists and defaults to None.

    None means "no explicit requested start"; downstream range validation
    interprets that as beginning-of-video for bounds checking.
    """
    captured = {}

    def fake_run_pipeline(args):
        captured["start_frame"] = args.start_frame
        captured["end_frame"] = args.end_frame
        return None

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(
        cli,
        "sys",
        types.SimpleNamespace(argv=["prog", "--input", "dummy.mp4"]),
    )

    cli.main()

    assert captured["start_frame"] is None
    assert captured["end_frame"] is None

def test_end_frame_before_start_frame_exits_fast(monkeypatch, tmp_path: Path):
    """
    New run contract:
      end_frame < start_frame is invalid and should raise a user-facing error.
    """
    ckpt = _FakeCkpt(root=tmp_path / "ckpt_run", resume_enabled=False)
    _patch_minimal_pipeline(monkeypatch, tmp_path, ckpt=ckpt, total_frames=100)

    args = _base_args(tmp_path, start_frame=20, end_frame=19)

    with pytest.raises(ResumeSafetyError):
        cli.run_pipeline(args)

def test_end_frame_equal_to_total_frames_exits_fast(monkeypatch, tmp_path: Path):
    """
    New run contract:
      end_frame is inclusive, so end_frame >= total_frames is out of bounds.
    """
    ckpt = _FakeCkpt(root=tmp_path / "ckpt_run", resume_enabled=False)
    _patch_minimal_pipeline(monkeypatch, tmp_path, ckpt=ckpt, total_frames=100)

    args = _base_args(tmp_path, start_frame=0, end_frame=100)

    with pytest.raises(ResumeSafetyError):
        cli.run_pipeline(args)

def test_no_resume_with_start_frame_cold_starts_at_start_frame(monkeypatch, tmp_path: Path, caplog):
    """
    If --no-resume is set, the pipeline must cold-start at start_frame
    and must not attempt checkpoint rehydration.
    """
    ckpt = _FakeCkpt(root=tmp_path / "ckpt_run", resume_enabled=True)

    seen = {}

    def fake_track_across_segments(**kwargs):
        seen["kwargs"] = kwargs
        return []

    _patch_minimal_pipeline(
        monkeypatch,
        tmp_path,
        ckpt=ckpt,
        total_frames=100,
        track_impl=fake_track_across_segments,
    )

    args = _base_args(tmp_path, no_resume=True, start_frame=25, end_frame=50)

    cli.run_pipeline(args)

    assert ckpt.rehydrate_called == 0
    assert seen["kwargs"]["resume_enabled"] is False

    # This is a forward-looking assertion for the new API contract.
    # It should fail until start_frame is plumbed into the tracking layer.
    assert seen["kwargs"]["start_frame"] == 25
    assert seen["kwargs"]["end_frame"] == 50

def test_incompatible_checkpoint_with_start_frame_exits_fast(monkeypatch, tmp_path: Path):
    """
    If normal resume logic finds a checkpoint but it is incompatible with the
    requested run, the CLI must exit fast rather than falling back to cold start.
    """
    ckpt = _FakeCkpt(
        root=tmp_path / "ckpt_run",
        resume_enabled=True,
        validate_exc=ResumeSafetyError("checkpoint incompatible with requested run"),
    )

    called = {"track": 0}

    def fake_track_across_segments(**kwargs):
        called["track"] += 1
        return []

    _patch_minimal_pipeline(
        monkeypatch,
        tmp_path,
        ckpt=ckpt,
        total_frames=100,
        track_impl=fake_track_across_segments,
    )

    args = _base_args(tmp_path, no_resume=False, start_frame=10, end_frame=20)

    with pytest.raises(ResumeSafetyError, match="incompatible"):
        cli.run_pipeline(args)

    assert ckpt.validate_called == 1
    assert ckpt.rehydrate_called == 0
    assert called["track"] == 0

def test_real_track_across_segments_accepts_frame_range_kwargs():
    """
    Guard against interface drift:
    if the CLI passes start_frame/end_frame into the real tracking entry point,
    the real function signature must accept them.
    """
    sig = inspect.signature(cli.track_across_segments.track_across_segments)

    assert "start_frame" in sig.parameters
    assert "end_frame" in sig.parameters

    assert sig.parameters["start_frame"].default == 0
    assert sig.parameters["end_frame"].default is None

def test_resume_with_start_frame_passes_frame_range_to_tracking_layer(
    monkeypatch,
    tmp_path: Path,
):
    """
    History-preserving frame-range execution:

    When resume is enabled and the user explicitly requests a start frame,
    the CLI must pass start_frame/end_frame through to the tracking layer
    so resume anchor selection can choose the latest embedding-safe frame
    strictly before the requested start.
    """
    ckpt = _FakeCkpt(
        root=tmp_path / "ckpt_run",
        resume_enabled=True,
        anchor_frame=19,
    )

    seen = {}

    def fake_track_across_segments(**kwargs):
        seen["kwargs"] = kwargs
        return []

    _patch_minimal_pipeline(
        monkeypatch,
        tmp_path,
        ckpt=ckpt,
        total_frames=100,
        track_impl=fake_track_across_segments,
    )

    args = _base_args(
        tmp_path,
        no_resume=False,
        resume_latest=True,
        start_frame=25,
        end_frame=50,
    )

    cli.run_pipeline(args)

    assert ckpt.validate_called == 1

    assert seen["kwargs"]["resume_enabled"] is True

    assert seen["kwargs"]["start_frame"] == 25
    assert seen["kwargs"]["end_frame"] == 50