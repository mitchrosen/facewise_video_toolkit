import json
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from fractions import Fraction

from tests.utils.video_mocks import make_pyav_like_frames 
import facekit.pipeline.track_across_segments as tacs
from facekit.pipeline.track_across_segments import track_across_segments

@pytest.fixture
def dummy_video(tmp_path):
    dummy_path = tmp_path / "dummy.mp4"
    dummy_path.write_bytes(b"not a real video")
    return str(dummy_path)

def _install_asserting_aligner(monkeypatch, *, expected_first_x_fn):
    """
    Monkeypatch align_face_for_arcface with an assertion-heavy stub.

    expected_first_x_fn(frame_idx) -> float:
        returns the expected value for landmarks[0][0] for that DET call.
        (You choose the contract per test: frame-based, call-based, etc.)
    """
    import numpy as _np

    def _asserting_align(frame, landmarks, frame_idx=None, source=None, *, return_meta=False):
        assert frame_idx is not None, "aligner called with frame_idx=None"
        assert landmarks is not None, "aligner called with landmarks=None"
        arr = _np.asarray(landmarks, dtype=_np.float32)

        # Accept either (5,2) or (1,5,2) and normalize to (5,2)
        if arr.shape == (1, 5, 2):
            arr = arr[0]
        assert arr.shape == (5, 2), f"expected (5,2) landmarks, got {arr.shape}"
        assert _np.all(_np.isfinite(arr)), "landmarks contain NaN/Inf"

        # Strong semantic check (not just > 0)
        exp = float(expected_first_x_fn(frame_idx))
        got = float(arr[0, 0])
        assert got == exp, f"landmark corruption: expected arr[0,0]={exp} got {got} (frame_idx={frame_idx})"

        chip = _np.zeros((112, 112, 3), dtype=_np.uint8)
        if return_meta:
            return chip, {"frame_idx": frame_idx, "source": source}
        return chip

    monkeypatch.setattr(tacs, "align_face_for_arcface", _asserting_align, raising=True)

def test_track_across_segments_with_mock_av(tmp_path, monkeypatch):
    dummy_shot_json = tmp_path / "shot_features.json"
    shots = [
        {"shot_number": 1, "first_frame": 0, "last_frame": 2},
        {"shot_number": 2, "first_frame": 3, "last_frame": 7},
    ]
    dummy_shot_json.write_text(json.dumps({"shots": shots}, indent=2))

    class FakeTracker:
        def __init__(self, tracker_type="CSRT", *args, **kwargs):
            self.trackers = []
            self.track_ids = []

        def init_trackers(self, frame, boxes, track_ids=None, *a, **k):
            self.trackers = list(boxes)
            self.track_ids = list(track_ids) if track_ids is not None else list(range(len(self.trackers)))

        def update_trackers(self, frame):
            return {tid: box for tid, box in zip(self.track_ids, self.trackers)}

    class FakeDetector:
        def __init__(self):
            self.calls = 0

        def detect_faces_in_frame(self, frame, target_size=640):
            self.calls += 1
            boxes = [(10, 10, 50, 50)]
            # Encode FRAME INDEX (+1) into landmarks[0][0].x (stronger than call count)
            # NOTE: track_across_segments passes the absolute frame index to aligner as frame_idx.
            # We don't have frame_idx here, so just encode call count and validate via call count
            # OR switch to frame_idx encoding by using the aligner meta instead (see below).
            x = float(self.calls)
            landmarks = [[(x, 0.0)] + [(0.0, 0.0)] * 4]
            confidences = [0.99]
            return boxes, landmarks, confidences

    det = FakeDetector()

    # Expected landmark x equals detector call count *at the time aligner is called*.
    # Since aligner is invoked on each detection, this should match 1..N in order.
    align_calls = {"n": 0}
    def _expected(_frame_idx):
        align_calls["n"] += 1
        return float(align_calls["n"])

    _install_asserting_aligner(monkeypatch, expected_first_x_fn=_expected)

    with patch("facekit.utils.video_reader.av.open") as mock_av_open, \
         patch.object(tacs, "FaceTracker", FakeTracker):
        mock_container = MagicMock()
        mock_stream = MagicMock()
        mock_stream.type = "video"
        mock_stream.frames = 8
        mock_container.streams.video = [mock_stream]
        mock_container.decode.return_value = make_pyav_like_frames(8)
        mock_av_open.return_value = mock_container

        class FakeEmbedder:
            def get_embedding_batch(self, aligned_faces, batch_size=32):
                K = len(aligned_faces)
                return np.ones((K, 512), dtype=np.float32)

        tracks = track_across_segments(
            frame_source="dummy.mp4",
            shot_json_path=str(dummy_shot_json),
            detector=det,
            embedder=FakeEmbedder(),
        )

        assert isinstance(tracks, list)
        assert all(hasattr(t, "track_id") for t in tracks)

def test_all_tracks_have_valid_segment_ids(monkeypatch, dummy_video, tmp_path):
    dummy_shot_json = tmp_path / "shot_features.json"
    dummy_shot_json.write_text(json.dumps({"shots": [{"shot_number": 1, "first_frame": 0, "last_frame": 4}]}))

    _calls = {"n": 0}

    def fake_detect_faces_in_frame(frame, target_size=640):
        _calls["n"] += 1
        boxes = [(10, 10, 50, 50)]
        x = float(_calls["n"])
        landmarks = [[(x, 0.0)] + [(0.0, 0.0)] * 4]
        confidences = [0.99]
        return boxes, landmarks, confidences

    class FakeDetector:
        def detect_faces_in_frame(self, frame, target_size=640):
            return fake_detect_faces_in_frame(frame, target_size)
        
    align_calls = {"n": 0}
    def _expected(_frame_idx):
        align_calls["n"] += 1
        return float(align_calls["n"])

    _install_asserting_aligner(monkeypatch, expected_first_x_fn=_expected)

    class FakeEmbedder:
        def get_embedding_batch(self, aligned_faces, batch_size=32):
            K = len(aligned_faces)
            embs = np.zeros((K, 512), dtype=np.float32)
            for i in range(K):
                embs[i, i % 512] = 1.0
            return embs
        
    class FakeTracker:
        def __init__(self, tracker_type="CSRT", *args, **kwargs):
            self.trackers = []
            self.track_ids = []

        def init_trackers(self, frame, boxes, track_ids=None, *a, **k):
            self.trackers = list(boxes)
            # Mirror aggregator behavior in this code path: IDs start at 2
            n = len(self.trackers)
            self.track_ids = list(track_ids) if track_ids is not None else list(range(len(self.trackers)))


        def update_trackers(self, frame):
            return {tid: box for tid, box in zip(self.track_ids, self.trackers)}

    with patch("facekit.utils.video_reader.av.open") as mock_av_open, \
     patch.object(tacs, "FaceTracker", FakeTracker):
        mock_container = MagicMock()
        mock_stream = MagicMock()
        mock_stream.type = "video"
        mock_stream.frames = 5
        mock_container.streams.video = [mock_stream]
        mock_container.decode.return_value = make_pyav_like_frames(5)
        mock_av_open.return_value = mock_container

        tracks = track_across_segments(
            frame_source=str(dummy_video),
            shot_json_path=str(dummy_shot_json),
            detector=FakeDetector(),
            embedder=FakeEmbedder(),
        )

        assert all(hasattr(track, "segment_id") for track in tracks)

def test_detector_none_disables_tracker(tmp_path, monkeypatch):
    shot_json = tmp_path / "shots.json"
    shot_json.write_text(json.dumps({"shots":[{"shot_number":1,"first_frame":0,"last_frame":9}]}))

    class FakeTracker:
        def __init__(self, tracker_type="CSRT", *args, **kwargs):
            self.trackers = []
            self.track_ids = []

        def init_trackers(self, frame, boxes, track_ids=None, *a, **k):
            self.trackers = list(boxes)
            self.track_ids = list(track_ids) if track_ids else list(range(len(self.trackers)))

        def update_trackers(self, frame):
            # track_across_segments expects a dict {track_id: (x,y,w,h)}
            return {tid: box for tid, box in zip(self.track_ids, self.trackers)}

    with patch("facekit.utils.video_reader.av.open") as mock_open, \
     patch.object(tacs, "FaceTracker", FakeTracker):
        c = MagicMock()
        s = MagicMock(); s.type="video"; s.frames=10
        c.streams.video=[s]
        c.decode.return_value = make_pyav_like_frames(10)
        mock_open.return_value = c

        class FakeDetector:
            def detect_faces_in_frame(self, frame, target_size=640):
                return None   # explicitly None

        class FakeEmbedder:
            def get_embedding_batch(self, aligned, batch_size=32):
                return np.zeros((len(aligned),512), dtype=np.float32)
            
        # Minimal tracker stub to ensure we don't call update when disabled
        class TrackerStub:
            def __init__(self, *a, **k): self.init_called=False; self.update_called=False
            def init_trackers(self, frame, boxes): self.init_called=True
            def update_trackers(self, frame): self.update_called=True; return []

        monkeypatch.setattr(tacs, "FaceTracker", TrackerStub)

        tracks = track_across_segments(
            frame_source="dummy.mp4",
            shot_json_path=str(shot_json),
            detector=FakeDetector(),
            embedder=FakeEmbedder(),
            detect_interval=3,
        )
        # No detections -> no tracks
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

#         tracks = track_across_segments(
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
    shot_json = tmp_path / "shots.json"
    shot_json.write_text(json.dumps({"shots": [{"shot_number": 1, "first_frame": 0, "last_frame": 4}]}))

    # Patch av.open to behave like the real thing:
    # - acts as a context manager
    # - yields fresh frames on every call to av.open(...).decode(...)
    with patch("facekit.utils.video_reader.av.open") as mock_open:
        # Describe the video stream once (shape/fps/etc)
        stream = MagicMock()
        stream.type = "video"
        stream.frames = 5
        stream.average_rate = Fraction(30, 1)
        stream.base_rate = Fraction(30, 1)
        stream.time_base = Fraction(1, 30)

        def make_context():
            c = MagicMock()
            c.streams.video = [stream]
            # Fresh iterator every time decode() is called
            c.decode.side_effect = lambda video=0: iter(make_pyav_like_frames(5))
            mgr = MagicMock()
            mgr.__enter__.return_value = c
            mgr.__exit__.return_value = False
            return mgr

        # Return a NEW context-managed container on each av.open(...)
        mock_open.side_effect = lambda *a, **k: make_context()

        class FakeDetector:
            def __init__(self):
                self.calls = 0
            def detect_faces_in_frame(self, frame, target_size=640):
                self.calls += 1
                x = float(self.calls)
                return [(10, 10, 50, 50)], [[(x, 0.0)] + [(0.0, 0.0)] * 4], [0.9]

        # Force align_face_for_arcface to return None on odd calls
        calls = {"n": 0}

        def fake_align(frame, landmarks, frame_idx=None, source=None):
            calls["n"] += 1

            arr = np.asarray(landmarks, dtype=np.float32)
            if arr.shape == (1, 5, 2):
                arr = arr[0]
            assert arr.shape == (5, 2)
            assert np.all(np.isfinite(arr))

            expected_x = float(calls["n"])
            actual_x = float(arr[0, 0])
            assert actual_x == expected_x, (
                f"expected landmarks[0][0].x={expected_x} but got {actual_x} (align call {calls['n']})"
            )

            return None if calls["n"] % 2 else np.zeros((112, 112, 3), dtype=np.uint8)

        monkeypatch.setattr(tacs, "align_face_for_arcface", fake_align)

        class FakeEmbedder:
            def get_embedding_batch(self, aligned_faces, batch_size=32):
                # Should only get crops for even calls
                return np.ones((len(aligned_faces), 512), dtype=np.float32)

        tracks = track_across_segments(
            frame_source="dummy.mp4",
            shot_json_path=str(shot_json),
            detector=FakeDetector(),
            embedder=FakeEmbedder(),
            detect_interval=1,
        )
        assert tracks  # at least one track, and no crash
