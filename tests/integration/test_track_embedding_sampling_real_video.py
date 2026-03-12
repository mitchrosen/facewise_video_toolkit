import json
import os
from pathlib import Path

import numpy as np
import pytest

from facekit.common.obs_consts import Source
from facekit.detection.face_detector import FaceDetector
from facekit.detection.yolo5face_model import load_yolo5face_model
from facekit.pipeline.track_across_segments import track_across_segments


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _require_file(path: Path, label: str) -> Path:
    if not path.exists():
        pytest.skip(f"Missing {label}: {path}")
    return path


def _write_single_shot_json(video_path: Path, shot_json_path: Path) -> None:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    try:
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()

    if nframes <= 0:
        pytest.skip(f"Could not read frame count from {video_path}")

    payload = {
        "shots": [
            {
                "shot_number": 1,
                "first_frame": 0,
                "last_frame": nframes - 1,
            }
        ]
    }
    shot_json_path.write_text(json.dumps(payload), encoding="utf-8")


class RecordingEmbedder:
    def __init__(self):
        self.calls = []

    def get_embedding_batch(self, aligned_faces, batch_size=32):
        self.calls.append(len(aligned_faces))
        n = len(aligned_faces)
        return np.zeros((n, 512), dtype=np.float32)


@pytest.mark.integration
def test_tracked_frame_embedding_counts_on_real_video(tmp_path, monkeypatch):
    """
    Phase 3 pinned integration test on a real clip.

    With sparse detection cadence and track_sample_interval=2:
    - DETECTED observations are embedded on every detection frame
    - TRACKED observations receive propagated landmarks
    - sampled TRACKED observations are aligned and embedded
    """
    repo = _repo_root()

    video = _require_file(
        repo / "tests" / "assets" / "videos" / "OGsTest_10sec_snippet.mp4",
        "test video",
    )
    detector_weights = _require_file(
        repo / "models" / "detector" / "yolov5n_state_dict.pt",
        "detector weights",
    )
    detector_cfg = _require_file(
        repo / "models" / "detector" / "yolov5n.yaml",
        "detector config",
    )

    shot_json = tmp_path / "single_shot.json"
    _write_single_shot_json(video, shot_json)

    device = os.getenv("FACEKIT_DEVICE", "cpu")

    model = load_yolo5face_model(
        str(detector_weights),
        str(detector_cfg),
        device=device,
    )
    detector = FaceDetector(model)
    embedder = RecordingEmbedder()

    import facekit.pipeline.track_across_segments as tas

    align_call_count = {"n": 0}

    def fake_align_face_for_arcface(frame, landmarks, *args, **kwargs):
        align_call_count["n"] += 1
        return np.zeros((112, 112, 3), dtype=np.uint8)

    monkeypatch.setattr(tas, "align_face_for_arcface", fake_align_face_for_arcface)

    tracks = track_across_segments(
        video,
        str(shot_json),
        detector=detector,
        embedder=embedder,
        detect_interval=8,
        track_sample_interval=2,
        checkpoint=None,
        resume_enabled=False,
    )

    detected_total = 0
    tracked_total = 0
    detected_with_embedding = 0
    tracked_with_landmarks = 0
    tracked_with_embedding = 0

    per_track_summary = []

    for track in tracks:
        track_detected = 0
        track_tracked = 0
        track_tracked_with_landmarks = 0
        track_tracked_with_embedding = 0
        track_detected_with_embedding = 0

        for obs in getattr(track, "observations", []) or []:
            if obs.source == Source.DETECTED:
                detected_total += 1
                track_detected += 1
                if getattr(obs, "embedding", None) is not None:
                    detected_with_embedding += 1
                    track_detected_with_embedding += 1
            elif obs.source == Source.TRACKED:
                tracked_total += 1
                track_tracked += 1
                if getattr(obs, "landmarks", None) is not None:
                    tracked_with_landmarks += 1
                    track_tracked_with_landmarks += 1
                if getattr(obs, "embedding", None) is not None:
                    tracked_with_embedding += 1
                    track_tracked_with_embedding += 1

        per_track_summary.append(
            {
                "track_id": getattr(track, "track_id", None),
                "detected": track_detected,
                "tracked": track_tracked,
                "tracked_with_landmarks": track_tracked_with_landmarks,
                "tracked_with_embedding": track_tracked_with_embedding,
                "detected_with_embedding": track_detected_with_embedding,
            }
        )

    assert len(tracks) == 8
    assert detected_total == 86
    assert tracked_total == 568
    assert detected_with_embedding == 86
    assert tracked_with_landmarks == 568
    assert tracked_with_embedding == 284
    assert align_call_count["n"] == 370
    assert embedder.calls == [370]

    assert per_track_summary == [
        {
            "track_id": 0,
            "detected": 13,
            "tracked": 90,
            "tracked_with_landmarks": 90,
            "tracked_with_embedding": 39,
            "detected_with_embedding": 13,
        },
        {
            "track_id": 1,
            "detected": 2,
            "tracked": 7,
            "tracked_with_landmarks": 7,
            "tracked_with_embedding": 4,
            "detected_with_embedding": 2,
        },
        {
            "track_id": 2,
            "detected": 26,
            "tracked": 171,
            "tracked_with_landmarks": 171,
            "tracked_with_embedding": 98,
            "detected_with_embedding": 26,
        },
        {
            "track_id": 3,
            "detected": 16,
            "tracked": 105,
            "tracked_with_landmarks": 105,
            "tracked_with_embedding": 60,
            "detected_with_embedding": 16,
        },
        {
            "track_id": 4,
            "detected": 9,
            "tracked": 63,
            "tracked_with_landmarks": 63,
            "tracked_with_embedding": 27,
            "detected_with_embedding": 9,
        },
        {
            "track_id": 5,
            "detected": 1,
            "tracked": 7,
            "tracked_with_landmarks": 7,
            "tracked_with_embedding": 3,
            "detected_with_embedding": 1,
        },
        {
            "track_id": 6,
            "detected": 10,
            "tracked": 66,
            "tracked_with_landmarks": 66,
            "tracked_with_embedding": 28,
            "detected_with_embedding": 10,
        },
        {
            "track_id": 7,
            "detected": 9,
            "tracked": 59,
            "tracked_with_landmarks": 59,
            "tracked_with_embedding": 25,
            "detected_with_embedding": 9,
        },
    ]