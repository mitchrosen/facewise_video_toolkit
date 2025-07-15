import pytest
import numpy as np
import cv2
import tempfile
import json
from pathlib import Path
from facekit.pipeline.track_across_shots import track_across_shots
from facekit.tracking.face_structures import FaceObservation

class FakeEmbedder:
    counter = 0

    def get_embedding(self, frame, box):
        emb = np.zeros(512, dtype=np.float32)
        emb[FakeEmbedder.counter % 512] = 1.0  # Unique unit vector per call
        nonzero_index = np.argmax(emb)
        FakeEmbedder.counter += 1
        return emb
    
@pytest.fixture(autouse=True)
def reset_embedder_counter_each_test():
    FakeEmbedder.counter = 0

@pytest.fixture
def dummy_video(tmp_path):
    path = tmp_path / "dummy.mp4"
    height, width = 100, 100
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(path), fourcc, 5.0, (width, height))

    for _ in range(10):
        frame = np.full((height, width, 3), 127, dtype=np.uint8)
        out.write(frame)
    out.release()
    return path

@pytest.fixture
def dummy_shot_json(tmp_path):
    shot_data = {
        "shots": [
            {"shot_number": 1, "first_frame": 0, "last_frame": 4},
            {"shot_number": 2, "first_frame": 5, "last_frame": 9}
        ]
    }
    json_path = tmp_path / "shot_features.json"
    json_path.write_text(json.dumps(shot_data, indent=2))
    return json_path


def fake_load_yolo5face_model(model_path=None, config_path=None, device=None):
    """
    Returns a fake detector function that simulates a non-overlapping face position
    each time it's called, for testing track ID differentiation.
    """
    count = 0

    def fake_detector(frame, target_size=640):
        nonlocal count
        val = 10 * count
        result = [[val, val, val + 5, val + 5]]  # bbox
        boxes = [[val, val, val + 5, val + 5]]
        landmarks = [[[val + 2, val + 2]] * 5]  # dummy 5 landmarks
        confidences = [0.99]
        count += 1
        return boxes, landmarks, confidences

    return fake_detector

import json
import numpy as np
from pathlib import Path
from facekit.tracking.face_structures import FaceObservation

def test_track_across_shots_with_mock(monkeypatch, dummy_video, tmp_path):
    # Build custom shot segmentation file
    dummy_shot_json = tmp_path / "shot_features.json"
    shots = [
        {"shot_number": 1, "first_frame": 0, "last_frame": 2},  # 3 frames
        {"shot_number": 2, "first_frame": 3, "last_frame": 7},  # 5 frames
    ]
    dummy_shot_json.write_text(json.dumps({"shots": shots}, indent=2))

    # Patch model loader to return fake model
    import facekit.pipeline.track_across_shots as track_mod
    monkeypatch.setattr(track_mod, "load_yolo5face_model", lambda *a, **k: "fake_model")

    # Patch face detection to return one box per frame
    frame_counter = {"idx": 0}

    def fake_detect_faces_in_frame(model, frame):
        idx = frame_counter["idx"]
        frame_counter["idx"] += 1
        # Simulate detection for first 3 frames of shot 1 and all 5 frames of shot 2
        if idx in [0, 1, 2, 3, 4, 5, 6, 7]:
            val = idx * 10
            return [(val, val, val + 5, val + 5)], None, [0.99]
        return None

    monkeypatch.setattr(track_mod, "detect_faces_in_frame", fake_detect_faces_in_frame)

    # Patch embedder
    class FakeEmbedder:
        def get_embedding(self, frame, bbox):
            # Generate deterministic embedding based on bbox for uniqueness
            val = bbox[0]  # Use x1 as a seed for consistency
            embedding = np.zeros(512, dtype=np.float32)
            embedding[val % 512] = 1.0
            return embedding

    tracks = track_mod.track_across_shots(
        video_path=str(dummy_video),
        shot_json_path=str(dummy_shot_json),
        model_path="fake.pt",
        config_path="fake.yaml",
        embedder=FakeEmbedder(),
    )

    # Validate number of tracks: 3 in first shot, 5 in second → total = 8
    assert len(tracks) == 8

    # Group tracks by shot_id
    shot1_tracks = [t for t in tracks if t.shot_id == 1]
    shot2_tracks = [t for t in tracks if t.shot_id == 2]

    assert len(shot1_tracks) == 3
    assert len(shot2_tracks) == 5

    # Track IDs reset per shot
    assert {t.track_id for t in shot1_tracks} == {0, 1, 2}
    assert {t.track_id for t in shot2_tracks} == {0, 1, 2, 3, 4}

    # Ensure vchunk IDs reset within each shot and no duplicates per shot
    shot1_vchunk_ids = [t.vchunk_id for t in shot1_tracks]
    shot2_vchunk_ids = [t.vchunk_id for t in shot2_tracks]

    assert set(shot1_vchunk_ids) == {0, 1, 2}
    assert set(shot2_vchunk_ids) == {0, 1, 2, 3, 4}

    # Validate observation structure
    for track in tracks:
        assert len(track.observations) == 1
        obs = track.observations[0]
        assert isinstance(obs, FaceObservation)
        assert obs.confidence == 0.99


def test_all_tracks_have_valid_vchunk_ids(dummy_video, dummy_shot_json, monkeypatch):
    import facekit.pipeline.track_across_shots as track_mod
    monkeypatch.setattr(track_mod, "load_yolo5face_model", fake_load_yolo5face_model)

    tracks = track_across_shots(
        video_path=str(dummy_video),
        shot_json_path=str(dummy_shot_json),
        model_path="fake.pt",
        config_path="fake.yaml",
        embedder=FakeEmbedder(),
    )

    for track in tracks:
        assert hasattr(track, "vchunk_id"), "Track missing vchunk_id"
        assert isinstance(track.vchunk_id, int), f"vchunk_id should be int, got {type(track.vchunk_id)}"
        assert track.vchunk_id >= 0, "vchunk_id should be non-negative"

