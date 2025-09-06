import json
import numpy as np
import pytest
from fractions import Fraction
from unittest.mock import patch
from tests.utils.pyav_fakes import FakeFrame, FakeContainer

@pytest.mark.integration
def test_track_across_segments_two_abutting_shots_no_overlap(tmp_path, monkeypatch):
    from facekit.pipeline.track_across_segments import track_across_segments

    # Keep this test focused on shot boundaries; avoid OpenCV geometry.
    monkeypatch.setattr(
        "facekit.pipeline.track_across_segments.align_face_for_arcface",
        lambda frame, lm, frame_idx=None, source=None: np.zeros((112, 112, 3), dtype=np.uint8),
    )

    # Shots [0..2] and [3..7]
    shot_json = tmp_path / "shots.json"
    shot_json.write_text(json.dumps({
        "shots": [
            {"shot_number": 1, "first_frame": 0, "last_frame": 2},
            {"shot_number": 2, "first_frame": 3, "last_frame": 7},
        ]
    }))

    # Deterministic fakes
    class FakeDetector:
        def detect_faces_in_frame(self, frame, target_size=640):
            # Landmarks are ignored because we patched the aligner above.
            return [(10,10,50,50)], [[(12,15),(28,15),(20,22),(14,30),(26,30)]], [0.9]

    class FakeEmbedder:
        def get_embedding_batch(self, aligned_faces, batch_size=32):
            return np.ones((len(aligned_faces), 512), dtype=np.float32)

    class FakeFaceTracker:
        def __init__(self, tracker_type="CSRT", *args, **kwargs):
            self.active_boxes = []
            self.ids = []

        def init_trackers(self, frame, boxes_xywh, track_ids=None, *a, **k):
            self.active_boxes = list(boxes_xywh)
            self.ids = list(track_ids) if track_ids is not None else list(range(len(self.active_boxes)))


        def update_trackers(self, frame):
            # Return a dict keyed by track_id as expected by track_across_segments
            return {tid: box for tid, box in zip(self.ids, self.active_boxes)}

    monkeypatch.setattr("facekit.pipeline.track_across_segments.FaceTracker", FakeFaceTracker)

    # Build frames 0..7 with numeric pts/time
    n, fps, tb = 8, 30.0, Fraction(1, 30)
    frames = [FakeFrame(i, pts=int(round((i/fps)/tb)), time=i/fps) for i in range(n)]
    container = FakeContainer(frames, fps_num=30, fps_den=1, time_base=tb)

    with patch("facekit.utils.video_reader.av.open", return_value=container):
        tracks = track_across_segments(
            video_path="dummy.mp4",
            shot_json_path=str(shot_json),
            detector=FakeDetector(),
            embedder=FakeEmbedder(),
        )

    # Collect observed frame indices across all tracks
    all_obs = sorted({obs.frame_idx for tr in tracks for obs in tr.observations})
    assert all_obs[0] == 0 and all_obs[-1] == 7
    assert all_obs == list(range(8))  # no overlap/no gap
