import types
from pathlib import Path
import pytest

import facekit.cli.resolve_face_ids_v2_cli as cli


class _FakeReader:
    def __init__(self, path):
        self._path = path
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def fps(self): return 30.0
    def size(self): return (1920, 1080)
    def total_frames(self): return 60


def test_min_face_is_passed_to_load_yolo5face_model(monkeypatch, tmp_path: Path):
    # --- Arrange minimal filesystem inputs ---
    vid = tmp_path / "toy.mp4"
    vid.write_bytes(b"not-a-real-video")  # existence check only

    shots = tmp_path / "shots.json"
    shots.write_text('{"shots":[{"shot_number":1,"first_frame":0,"last_frame":59}]}')

    captured = {}

    def fake_load(detector_model_path, config_path, device="cuda", min_face=10):
        captured["min_face"] = min_face
        return object()

    # Stub pipeline pieces so run_pipeline exits early and safely.
    monkeypatch.setattr(cli, "ReaderCoordinator", lambda p: _FakeReader(p))
    monkeypatch.setattr(cli, "load_yolo5face_model", fake_load)
    monkeypatch.setattr(cli, "FaceDetector", lambda yolo: object())
    monkeypatch.setattr(cli, "FaceEmbedder", lambda *a, **k: object())

    # avoid checkpoint side effects
    class _FakeCkpt:
        root = tmp_path / "ckpt_run"
        resume_enabled = False
        def read_status(self): return {}
        def validate_resume_or_raise(self, *a, **k): return None
        def start(self, *a, **k): return None
        def rehydrate_runtime(self, *a, **k): return {}
        def finalize(self): return None

    monkeypatch.setattr(cli.CheckpointManager, "open", lambda *a, **k: _FakeCkpt())
    monkeypatch.setattr(cli.CheckpointManager, "compute_parent_dir", lambda *a, **k: tmp_path / "ckpt_parent")

    # Track across segments: return no tracks, and do nothing else
    monkeypatch.setattr(cli.track_across_segments, "track_across_segments", lambda **k: [])

    # Writer path not executed because output_global_json is None in args below
    # --- Build args object ---
    args = types.SimpleNamespace(
        input=str(vid),
        shot_segmentation=str(shots),
        schema_version="2.0",
        schema_dir=None,
        detector_model="models/detector/yolov5n_state_dict.pt",
        embedding_model="models/embedding/glintr100_dynamic.onnx",
        config="models/detector/yolov5n.yaml",
        min_face=37,  # <-- what we want to see plumbed
        detect_interval=30,
        embedding_batch_size_max=32,
        device="auto",
        emb_store="none",
        emb_sidecar_path=None,
        obs_sidecar_path=None,
        checkpoint_dir=str(tmp_path / "ckpt_parent"),
        no_resume=True,
        new_run=False,
        resume_latest=False,
        checkpoint_run_id=None,
        force=False,
        no_checkpoint_write=True,
        log="INFO",
        log_file=None,
        output_segment_json=None,
        output_global_json=None,
        output_video=None,
        post_min_gap_len=210,
        post_min_track_len=70,
        post_iou_threshold=0.2,
    )

    # --- Act ---
    cli.run_pipeline(args)

    # --- Assert ---
    # This should FAIL until you change the call to:
    # load_yolo5face_model(..., min_face=args.min_face)
    assert captured["min_face"] == 37
