import pytest
import numpy as np
import cv2
import json
from pathlib import Path
from facekit.pipeline.track_across_shots import track_across_shots
from facekit.tracking.face_structures import FaceObservation

class FakeEmbedder:
    counter = 0

    def get_embedding(self, frame):
        emb = np.zeros(512, dtype=np.float32)
        emb[FakeEmbedder.counter % 512] = 1.0
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


def test_track_across_shots_with_mock(monkeypatch, dummy_video, tmp_path):
    # Build custom shot segmentation file
    dummy_shot_json = tmp_path / "shot_features.json"
    shots = [
        {"shot_number": 1, "first_frame": 0, "last_frame": 2},  # 3 frames
        {"shot_number": 2, "first_frame": 3, "last_frame": 7},  # 5 frames
    ]
    dummy_shot_json.write_text(json.dumps({"shots": shots}, indent=2))

    # ✅ Patch detection + embedding
    def fake_detect_faces_and_embeddings(frame, frame_idx, embedder=None):
        emb = embedder.get_embedding(frame) if embedder else None
        bbox = (frame_idx * 10, frame_idx * 10, frame_idx * 10 + 5, frame_idx * 10 + 5)
        return [FaceObservation(frame_idx=frame_idx, bbox=bbox, confidence=0.99, embedding=emb)]

    import facekit.pipeline.track_across_shots as track_mod
    monkeypatch.setattr(track_mod, "detect_faces_and_embeddings", fake_detect_faces_and_embeddings)

    # ✅ Run tracking
    tracks = track_mod.track_across_shots(
        video_path=str(dummy_video),
        shot_json_path=str(dummy_shot_json),
        embedder=FakeEmbedder(),
    )

    # ✅ Validate total number of tracks
    assert len(tracks) == 8  # 3 for shot 1 + 5 for shot 2

    # ✅ Validate per-shot track distribution
    shot1_tracks = [t for t in tracks if t.shot_id == 1]
    shot2_tracks = [t for t in tracks if t.shot_id == 2]
    assert len(shot1_tracks) == 3
    assert len(shot2_tracks) == 5

    # ✅ track_id resets per shot
    assert {t.track_id for t in shot1_tracks} == {0, 1, 2}
    assert {t.track_id for t in shot2_tracks} == {0, 1, 2, 3, 4}
    
    # ✅ vchunk_id reset per shot
    assert {t.vchunk_id for t in shot1_tracks} == {0, 1, 2}
    assert {t.vchunk_id for t in shot2_tracks} == {0, 1, 2, 3, 4}

    # ✅ Validate observations
    for track in tracks:
        assert len(track.observations) == 1
        obs = track.observations[0]
        assert isinstance(obs, FaceObservation)
        assert obs.confidence == 0.99
        assert obs.embedding is not None


def test_all_tracks_have_valid_vchunk_ids(monkeypatch, dummy_video, tmp_path):
    # Build minimal shot segmentation file
    dummy_shot_json = tmp_path / "shot_features.json"
    dummy_shot_json.write_text(json.dumps({"shots": [{"shot_number": 1, "first_frame": 0, "last_frame": 4}]}, indent=2))

    # ✅ Patch detection to return predictable embeddings
    def fake_detect_faces_and_embeddings(frame, frame_idx, embedder=None):
        emb = np.zeros(512, dtype=np.float32)
        emb[frame_idx % 512] = 1.0
        bbox = (frame_idx * 10, frame_idx * 10, frame_idx * 10 + 5, frame_idx * 10 + 5)
        return [FaceObservation(frame_idx=frame_idx, bbox=bbox, confidence=0.99, embedding=emb)]

    import facekit.pipeline.track_across_shots as track_mod
    monkeypatch.setattr(track_mod, "detect_faces_and_embeddings", fake_detect_faces_and_embeddings)

    tracks = track_mod.track_across_shots(
        video_path=str(dummy_video),
        shot_json_path=str(dummy_shot_json),
        embedder=FakeEmbedder(),
    )

    for track in tracks:
        assert hasattr(track, "vchunk_id")
        assert isinstance(track.vchunk_id, int)
        assert track.vchunk_id >= 0
