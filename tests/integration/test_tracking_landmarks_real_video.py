import json
from pathlib import Path

import cv2
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


def _write_single_short_shot_json(
    *,
    video_path: Path,
    shot_json_path: Path,
    max_frames: int = 48,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    try:
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()

    if nframes <= 0:
        pytest.skip(f"Could not read frame count from {video_path}")

    last_frame = min(nframes, max_frames) - 1
    if last_frame < 1:
        pytest.skip(f"Video too short for test: {video_path}")

    payload = {
        "shots": [
            {
                "shot_number": 1,
                "first_frame": 0,
                "last_frame": last_frame,
            }
        ]
    }
    shot_json_path.write_text(json.dumps(payload), encoding="utf-8")


class DummyEmbedder:
    def get_embedding_batch(self, aligned_faces, batch_size=32):
        import numpy as np

        return np.zeros((len(aligned_faces), 512), dtype=np.float32)


@pytest.mark.integration
def test_tracked_observations_gain_landmarks_on_real_video(tmp_path):
    """
    Lean Phase 3 integration test:

    On a real clip with sparse detection cadence, tracking should produce at least
    one TRACKED observation with propagated landmarks.

    This test is intentionally narrower and faster than the full characterization test.
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

    shot_json = tmp_path / "short_single_shot.json"
    _write_single_short_shot_json(
        video_path=video,
        shot_json_path=shot_json,
        max_frames=48,
    )

    model = load_yolo5face_model(
        str(detector_weights),
        str(detector_cfg),
        device="cpu",
    )
    detector = FaceDetector(model)
    embedder = DummyEmbedder()

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
    tracked_with_landmarks = 0

    for track in tracks:
        for obs in getattr(track, "observations", []) or []:
            if obs.source == Source.DETECTED:
                detected_total += 1
            elif obs.source == Source.TRACKED:
                tracked_total += 1
                if getattr(obs, "landmarks", None) is not None:
                    tracked_with_landmarks += 1

    assert detected_total > 0, "Expected at least one DETECTED observation"
    assert tracked_total > 0, "Expected at least one TRACKED observation"
    assert tracked_with_landmarks > 0, (
        "Expected at least one TRACKED observation with propagated landmarks"
    )