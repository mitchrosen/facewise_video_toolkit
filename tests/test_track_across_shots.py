import pytest
import numpy as np
import cv2
import tempfile
import json
from pathlib import Path
from facekit.pipeline.track_across_shots import track_across_shots
from facekit.tracking.face_tracks import FaceObservation

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

def test_track_across_shots_with_mock(monkeypatch, dummy_video, dummy_shot_json):
    # Patch model loader to return fake model
    import facekit.pipeline.track_across_shots as track_mod
    monkeypatch.setattr(track_mod, "load_yolo5face_model", fake_load_yolo5face_model)

    tracks = track_across_shots(
        video_path=str(dummy_video),
        shot_json_path=str(dummy_shot_json),
        model_path="fake.pt",
        config_path="fake.yaml",
        embedder=FakeEmbedder(),
    )

    # Basic checks
    assert len(tracks) == 10
    assert sum(len(t.observations) for t in tracks) == 10
    track_ids = {t.track_id for t in tracks}

    assert len(tracks) == 10

    # Track IDs are reused per shot, so max unique IDs per shot = 5
    # Expect only 0–4 repeated twice
    track_ids = [t.track_id for t in tracks]
    assert set(track_ids).issubset(set(range(5)))
    assert track_ids.count(0) == 2  # Each ID appears twice (once per shot)
    assert track_ids.count(1) == 2
    assert track_ids.count(2) == 2
    assert track_ids.count(3) == 2
    assert track_ids.count(4) == 2

    # Global IDs should be unique for each face (since embeddings differ)
    global_ids = [t.global_id for t in tracks]
    assert set(global_ids) == set(range(10))

    for idx, track in enumerate(tracks):
        assert len(track.observations) == 1
        obs = track.observations[0]
        assert isinstance(obs, FaceObservation)
        val = idx * 10
        assert obs.bbox == (val, val, val+5, val+5)

        expected = np.zeros(512, dtype=np.float32)
        expected[idx % 512] = 1.0
        np.testing.assert_array_equal(obs.embedding, expected)
        
        assert obs.confidence == 0.99

    # test frame indices
    frame_idxs = [track.observations[0].frame_idx for track in tracks]
    assert frame_idxs == list(range(10))
