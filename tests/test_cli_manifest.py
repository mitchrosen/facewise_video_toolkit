# tests/integration/test_resolve_cli_trackless_shots_v21.py

from types import SimpleNamespace
from pathlib import Path
import json

import pytest

from facekit.common.obs_consts import Source
from facekit.tracking.face_structures import FaceTrack, FaceObservation
import facekit.cli.resolve_face_ids_v2_cli as cli


def test_cli_v2_1_manifest_includes_trackless_shots_with_full_coverage(tmp_path: Path, monkeypatch):
    """
    Integration-style test for resolve_face_ids_v2_cli.run_pipeline().

    Shot segmentation defines 4 shots:
      shot 1: faces present
      shot 2: graphics-only, no faces
      shot 3: faces present
      shot 4: graphics-only, no faces (tail coverage)

    We stub track_across_segments.track_across_segments() to return tracks
    only for shots 1 and 3.

    Desired behavior:
      - manifest["shots"] contains shot_numbers [1, 2, 3, 4]
      - each shot has correct (first_frame, last_frame)
      - coverage is continuous from 0 to total_frames - 1
    """


    # ------------------------------
    # 1) Shot segmentation with 4 shots
    # ------------------------------
    # total_frames = 300 => last frame index = 299
    total_frames = 300
    shot_defs = [
        {"shot_number": 1, "first_frame": 0,   "last_frame": 99},
        {"shot_number": 2, "first_frame": 100, "last_frame": 199},  # graphics-only
        {"shot_number": 3, "first_frame": 200, "last_frame": 249},
        {"shot_number": 4, "first_frame": 250, "last_frame": 299},  # graphics-only tail
    ]
    shot_json_path = tmp_path / "shots.json"
    shot_json_path.write_text(json.dumps({"shots": shot_defs}), encoding="utf-8")

    # ------------------------------
    # 2) Fake video file
    # ------------------------------
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"FAKE")

    # ------------------------------
    # 3) Fake ReaderCoordinator -> FrameProvider
    # ------------------------------
    class FakeReader:
        def __init__(self, path: str):
            self._fps = 30.0
            self._size = (1920, 1080)
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

        # Interface required by track_across_segments
        def reset_to_frame(self, idx: int):
            pass

        def next(self):
            return None

    monkeypatch.setattr(cli, "ReaderCoordinator", FakeReader)

    # ------------------------------
    # 4) Detector / embedder stubs
    # ------------------------------
    def fake_load_yolo5face_model(detector_model, config, *, device=None, min_face=10, **kwargs):

        class DummyYOLO:
            pass
        return DummyYOLO()

    class FakeEmbedder:
        def __init__(self, model_path, device="cpu"):
            self.model_name = "fake-embedder"

        def set_max_batch_size(self, n: int):
            pass

    monkeypatch.setattr(cli, "load_yolo5face_model", fake_load_yolo5face_model)
    monkeypatch.setattr(cli, "FaceEmbedder", FakeEmbedder)

    # ------------------------------
    # 5) CheckpointManager stub
    # ------------------------------
    class FakeCheckpoint:
        def __init__(self, root: Path):
            self.root = root
            self.resume_enabled = False  # treat as cold start

        def read_status(self):
            return {}

        def validate_resume_or_raise(self, *a, **k):
            pass

        def start(self, *a, **k):
            pass

        def rehydrate_runtime(self, *a, **k):
            return {
                "anchor_frame": 0,
                "anchor_shot": None,
                "anchor_shot_first_frame": None,
                "track_order_entries": 0,
            }

        def finalize(self):
            pass

        def copy_ckpt_sidecars_to_final(self, **kwargs):
            pass

        def mark_completed(self):
            pass

    class FakeCheckpointManager:
        @staticmethod
        def open(
            parent_dir,
            video_path,
            options_snapshot,
            no_resume,
            force_new_run,
            run_id,
            resume_latest,
            **kwargs,
        ):
            return FakeCheckpoint(parent_dir / "fake-run")

    monkeypatch.setattr(cli, "CheckpointManager", FakeCheckpointManager)

    # ------------------------------
    # 6) track_across_segments stub
    #    -> tracks only for shots 1 and 3
    # ------------------------------
    def fake_track_across_segments(
        frame_source,
        shot_json_path: str,
        detector,
        embedder,
        iou_thresh=0.5,
        embedding_thresh=0.7,
        detect_interval=10,
        embedding_batch_size_max=32,
        embedding_queue_max_pending=1024,
        checkpoint=None,
        resume_enabled=True,
    ):
        tracks: list[FaceTrack] = []

        # Shot 1 track
        obs1 = [
            FaceObservation(
                frame_idx=0,
                source=Source.DETECTED,
                bbox=(10, 10, 50, 50),
                confidence=0.9,
            )
        ]
        tracks.append(
            FaceTrack(
                shot_id=1,
                track_id=0,
                observations=obs1,
                segment_id=0,
                global_id=0,
            )
        )

        # Shot 3 track
        obs3 = [
            FaceObservation(
                frame_idx=220,  # inside shot 3 range [200,249]
                source=Source.DETECTED,
                bbox=(20, 20, 60, 60),
                confidence=0.8,
            )
        ]
        tracks.append(
            FaceTrack(
                shot_id=3,
                track_id=0,
                observations=obs3,
                segment_id=0,
                global_id=1,
            )
        )

        return tracks

    monkeypatch.setattr(cli.track_across_segments, "track_across_segments", fake_track_across_segments)

    # ------------------------------
    # 7) validate_manifest stub: always OK
    # ------------------------------
    def fake_validate_manifest(json_path, schema_version, total_frame_count, schema_dir=None):
        return []

    monkeypatch.setattr(cli, "validate_manifest", fake_validate_manifest)

    # ------------------------------
    # 8) Capture manifest passed to write_v2_json
    # ------------------------------
    captured: dict = {}

    def fake_write_v2_json(path: str, manifest: dict):
        captured["path"] = path
        captured["manifest"] = manifest

    monkeypatch.setattr(cli, "write_v2_json", fake_write_v2_json)

    # ------------------------------
    # 9) Build args and run pipeline
    # ------------------------------
    args = SimpleNamespace(
        input=str(video_path),
        detector_model="dummy-detector.pt",
        embedding_model="dummy-emb.onnx",
        config="dummy-config.yaml",
        min_face=10,
        shot_segmentation=str(shot_json_path),
        schema_version="2.1",
        schema_dir=None,
        output_segment_json=None,
        output_global_json=True,             # trigger JSON manifest write
        output_video=None,
        detect_interval=30,
        post_min_gap_len=210,
        post_min_track_len=70,
        post_iou_threshold=0.2,
        embedding_batch_size_max=32,
        embedding_queue_max_pending=1024,
        device="cpu",
        emb_store="sidecar",
        emb_sidecar_path=None,
        obs_sidecar_path=None,
        checkpoint_dir=str(tmp_path / "ckpt"),
        no_resume=True,
        force=False,
        resume_latest=False,
        checkpoint_run_id=None,
        new_run=False,
        log="INFO",
        log_file=None,
    )

    cli.run_pipeline(args)

    # ------------------------------
    # 10) Assertions on manifest.shots
    # ------------------------------
    assert "manifest" in captured, "write_v2_json was never called"
    manifest = captured["manifest"]

    shots = manifest.get("shots", [])
    shot_nums = [s["shot_number"] for s in shots]

    # Desired shot list: all 4 shots (including 2 & 4 with no faces)
    assert shot_nums == [1, 2, 3, 4]

    # Expected coverage per shot
    expected_ranges = {
        1: (0, 99),
        2: (100, 199),
        3: (200, 249),
        4: (250, 299),
    }
    actual_ranges = {
        s["shot_number"]: (int(s["first_frame"]), int(s["last_frame"]))
        for s in shots
    }

    assert actual_ranges == expected_ranges

