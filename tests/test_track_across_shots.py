import json
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import facekit.pipeline.track_across_shots as tacs
from facekit.pipeline.track_across_shots import track_across_shots
from tests.utils.video_mocks import make_pyav_like_frames
from tests.utils.video_mocks import make_frames_without_time

@pytest.fixture
def dummy_video(tmp_path):
    dummy_path = tmp_path / "dummy.mp4"
    dummy_path.write_bytes(b"not a real video")
    return str(dummy_path)

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
        mock_container.decode.return_value = make_pyav_like_frames(8)
        mock_av_open.return_value = mock_container

        class FakeDetector:
            def detect_faces_in_frame(self, frame, target_size=640):
                boxes = [(10, 10, 50, 50)]
                landmarks = [[(38, 52), (73, 52), (56, 72), (42, 92), (71, 92)]]
                confidences = [0.99]
                return boxes, landmarks, confidences

        class FakeEmbedder:
            def get_embedding_batch(self, aligned_faces, batch_size=32):
                # Shape: (K, 512), float32
                K = len(aligned_faces)
                return np.ones((K, 512), dtype=np.float32)

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
            K = len(aligned_faces)
            embs = np.zeros((K, 512), dtype=np.float32)
            for i in range(K):
                embs[i, i % 512] = 1.0
            return embs

    with patch("facekit.utils.video_reader.av.open") as mock_av_open:
        mock_container = MagicMock()
        mock_stream = MagicMock()
        mock_stream.type = "video"
        mock_stream.frames = 5
        mock_container.streams.video = [mock_stream]
        mock_container.decode.return_value = make_pyav_like_frames(5)
        mock_av_open.return_value = mock_container

        tracks = track_across_shots(
            video_path=str(dummy_video),
            shot_json_path=str(dummy_shot_json),
            detector=FakeDetector(),
            embedder=FakeEmbedder(),
        )

        assert all(hasattr(track, "vchunk_id") for track in tracks)

def test_detector_none_disables_tracker(tmp_path, monkeypatch):
    shot_json = tmp_path / "shots.json"
    shot_json.write_text(json.dumps({"shots":[{"shot_number":1,"first_frame":0,"last_frame":9}]}))

    with patch("facekit.utils.video_reader.av.open") as mock_open:
        c = MagicMock()
        s = MagicMock(); s.type="video"; s.frames=10
        c.streams.video=[s]
        c.decode.return_value = make_pyav_like_frames(10)
        mock_open.return_value = c

        class FakeDetector:
            def detect_faces_in_frame(self, frame, target_size=640):
                return None   # <- explicitly None

        class FakeEmbedder:
            def get_embedding_batch(self, aligned, batch_size=32):
                return np.zeros((len(aligned),512), dtype=np.float32)

        # Minimal tracker stub to ensure we don't call update when disabled
        class TrackerStub:
            def __init__(self, *a, **k): self.init_called=False; self.update_called=False
            def init_trackers(self, frame, boxes): self.init_called=True
            def update_trackers(self, frame): self.update_called=True; return []

        monkeypatch.setattr(tacs, "FaceTracker", TrackerStub)

        tracks = track_across_shots(
            video_path="dummy.mp4",
            shot_json_path=str(shot_json),
            detector=FakeDetector(),
            embedder=FakeEmbedder(),
            detect_interval=3,
        )
        # No detections → no tracks
        assert tracks == []

# def test_video_reader_fallback_without_time(monkeypatch, tmp_path):
#     # Build a shot json for, say, 5 frames
#     shots_json = tmp_path / "shots.json"
#     shots_json.write_text('{"shots":[{"shot_number":1,"first_frame":0,"last_frame":4}]}')

#     with patch("facekit.utils.video_reader.av.open") as mock_open:
#         container = MagicMock()
#         stream = MagicMock()
#         stream.type = "video"
#         stream.frames = 5
#         container.streams.video = [stream]
#         container.decode.return_value = make_frames_without_time(5)
#         mock_open.return_value = container

#         class FakeDetector:
#             def detect_faces_in_frame(self, frame, target_size=640):
#                 return ([(10, 10, 50, 50)], [[(38, 52),(73, 52),(56, 72),(42, 92),(71, 92)]], [0.99])

#         class FakeEmbedder:
#             def get_embedding_batch(self, faces, batch_size=32):
#                 import numpy as np
#                 return np.zeros((len(faces), 512), dtype=np.float32)

#         tracks = track_across_shots(
#             video_path="dummy.mp4",
#             shot_json_path=str(shots_json),
#             detector=FakeDetector(),
#             embedder=FakeEmbedder(),
#         )

#         assert len(tracks) >= 1
#         # Ensure we got frames despite missing .time
#         total_obs = sum(len(t.observations) for t in tracks)
#         assert total_obs > 0

def test_align_face_returns_none_is_skipped(tmp_path, monkeypatch):
    # 0..4 frames; align_face returns None half the time
    # Expect: no exception; embedding batch sees only non‑None crops
    import json, numpy as np
    from facekit.pipeline.track_across_shots import track_across_shots

    shot_json = tmp_path / "shots.json"
    shot_json.write_text(json.dumps({"shots":[{"shot_number":1,"first_frame":0,"last_frame":4}]}))

    with patch("facekit.utils.video_reader.av.open") as mock_open:
        c = MagicMock()
        s = MagicMock(); s.type="video"; s.frames=5
        c.streams.video=[s]; c.decode.return_value = make_pyav_like_frames(5)
        mock_open.return_value = c

        class FakeDetector:
            def detect_faces_in_frame(self, frame, target_size=640):
                return [(10,10,50,50)], [[(20,20)]*5], [0.9]

        # Force align_face_for_arcface to return None on odd calls
        calls = {"n":0}
        def fake_align(frame, lm):
            calls["n"] += 1
            return None if calls["n"] % 2 else np.zeros((112,112,3), dtype=np.uint8)

        monkeypatch.setattr(tacs, "align_face_for_arcface", fake_align)

        class FakeEmbedder:
            def get_embedding_batch(self, aligned_faces, batch_size=32):
                # Should only get crops for even calls
                return np.ones((len(aligned_faces),512), dtype=np.float32)

        tracks = track_across_shots(
            video_path="dummy.mp4",
            shot_json_path=str(shot_json),
            detector=FakeDetector(),
            embedder=FakeEmbedder(),
            detect_interval=1,
        )
        # Should produce one track with some observations; no crash
        assert tracks
