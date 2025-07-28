import json
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from facekit.pipeline.track_across_shots import track_across_shots


@pytest.fixture
def dummy_video(tmp_path):
    dummy_path = tmp_path / "dummy.mp4"
    dummy_path.write_bytes(b"not a real video")
    return str(dummy_path)


def make_mock_frames(num_frames):
    frames = []
    for i in range(num_frames):
        frame = MagicMock()
        frame.to_ndarray.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        frame.time = i * 1 / 30  # simulate 30fps timestamp
        frames.append(frame)
    return frames


def test_track_across_shots_with_mock_av(tmp_path):
    dummy_shot_json = tmp_path / "shot_features.json"
    shots = [
        {"shot_number": 1, "first_frame": 0, "last_frame": 2},
        {"shot_number": 2, "first_frame": 3, "last_frame": 7},
    ]
    dummy_shot_json.write_text(json.dumps({"shots": shots}, indent=2))

    with patch("facekit.utils.video_reader.av.open") as mock_av_open:
        mock_container = MagicMock()
        mock_stream = MagicMock()
        mock_stream.type = "video"
        mock_stream.frames = 8
        mock_container.streams.video = [mock_stream]
        mock_container.decode.return_value = make_mock_frames(8)
        mock_av_open.return_value = mock_container

        class FakeDetector:
            def detect_faces_in_frame(self, frame, target_size=640):
                boxes = [(10, 10, 50, 50)]
                landmarks = [[(38, 52), (73, 52), (56, 72), (42, 92), (71, 92)]]
                confidences = [0.99]
                return boxes, landmarks, confidences

        class FakeEmbedder:
            def get_embedding_batch(self, aligned_faces, batch_size=32):
                return [np.ones(512, dtype=np.float32) for _ in aligned_faces]

        tracks = track_across_shots(
            video_path="dummy.mp4",
            shot_json_path=str(dummy_shot_json),
            detector=FakeDetector(),
            embedder=FakeEmbedder(),
        )

        assert isinstance(tracks, list)
        assert all(hasattr(t, "track_id") for t in tracks)


def test_all_tracks_have_valid_vchunk_ids(monkeypatch, dummy_video, tmp_path):
    dummy_shot_json = tmp_path / "shot_features.json"
    dummy_shot_json.write_text(json.dumps({"shots": [{"shot_number": 1, "first_frame": 0, "last_frame": 4}]}))

    def fake_detect_faces_in_frame(frame, target_size=640):
        boxes = [(10, 10, 50, 50)]
        landmarks = [[(38, 52), (73, 52), (56, 72), (42, 92), (71, 92)]]
        confidences = [0.99]
        return boxes, landmarks, confidences

    class FakeDetector:
        def detect_faces_in_frame(self, frame, target_size=640):
            return fake_detect_faces_in_frame(frame, target_size)

    class FakeEmbedder:
        def get_embedding_batch(self, aligned_faces, batch_size=32):
            embeddings = []
            for idx, _ in enumerate(aligned_faces):
                emb = np.zeros(512, dtype=np.float32)
                emb[idx % 512] = 1.0
                embeddings.append(emb)
            return embeddings

    with patch("facekit.utils.video_reader.av.open") as mock_av_open:
        mock_container = MagicMock()
        mock_stream = MagicMock()
        mock_stream.type = "video"
        mock_stream.frames = 5
        mock_container.streams.video = [mock_stream]
        mock_container.decode.return_value = make_mock_frames(5)
        mock_av_open.return_value = mock_container

        tracks = track_across_shots(
            video_path=str(dummy_video),
            shot_json_path=str(dummy_shot_json),
            detector=FakeDetector(),
            embedder=FakeEmbedder(),
        )

        assert all(hasattr(track, "vchunk_id") for track in tracks)
