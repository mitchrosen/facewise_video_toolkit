import numpy as np
from pathlib import Path

from facekit.common.obs_consts import Source
from facekit.pipeline.track_across_segments import track_across_segments


class TinyFrameProvider:
    """FrameProvider-like object: supports reset_to_frame() + next(), plus fps/size/total_frames attrs."""
    def __init__(self, frames):
        self._frames = list(frames)
        self.fps = 30.0
        self.size = (self._frames[0].shape[1], self._frames[0].shape[0])  # (w,h)
        self.total_frames = len(self._frames)
        self._i = 0
 
    def reset_to_frame(self, i: int):
        self._i = int(i)

    def next(self):
        if self._i < 0 or self._i >= len(self._frames):
            return None
        f = self._frames[self._i]
        self._i += 1
        return f


class FakeDetector:
    def detect_faces_in_frame(self, frame):
        # One face, landmarks non-None so it should be treated as detected
        boxes = [(10, 10, 50, 50)]
        landmarks = [[(1.0, 2.0)] * 5]
        confs = [0.9]
        return boxes, landmarks, confs


class FakeEmbedder:
    def get_embedding_batch(self, aligned_faces, batch_size=32):
        # not used in this test (we set detect_interval=1, but we only do 1 frame)
        n = len(aligned_faces)
        return np.zeros((n, 512), dtype=np.float32)


def test_track_across_segments_sets_aligned_face_on_detection(tmp_path, monkeypatch):
    """
    Contract test: on a detection frame with landmarks, track_across_segments()
    must compute obs.aligned_face immediately (no later get_frame rereads).
    """
    # Patch align_face_for_arcface used inside track_across_segments module
    import facekit.pipeline.track_across_segments as tas

    sentinel = np.zeros((112, 112, 3), dtype=np.uint8)
    align_calls = {"n": 0}

    def fake_align(frame, landmarks, *args, **kwargs):
        align_calls["n"] += 1
        return sentinel

    monkeypatch.setattr(tas, "align_face_for_arcface", fake_align)

    # Minimal shot json: single shot, single frame
    shot_json = tmp_path / "shots.json"
    shot_json.write_text(
        '{"shots":[{"shot_number":1,"first_frame":0,"last_frame":0}]}',
        encoding="utf-8",
    )

    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    fp = TinyFrameProvider([frame])

    tracks = track_across_segments(
        fp,
        str(shot_json),
        detector=FakeDetector(),
        embedder=FakeEmbedder(),
        detect_interval=1,  # ensure detection
        checkpoint=None,
        resume_enabled=False,
    )

    # Expect align_face_for_arcface to have been called (aligned_face computed at detection time),
    # but aligned_face may be cleared after embedding. So we assert:
    #  - align_face_for_arcface() was invoked
    #  - at least one DETECTED obs with landmarks ended with embedding set
    #  - aligned_face cleared (memory reclaimed; no later reread needed)
    assert align_calls["n"] >= 1, "Expected align_face_for_arcface to be called on a detection frame"

    detected_with_embedding = 0
    for t in tracks:
        for ob in getattr(t, "observations", []) or []:
            if ob.source == Source.DETECTED and getattr(ob, "landmarks", None) is not None:
                if getattr(ob, "embedding", None) is not None:
                    detected_with_embedding += 1
                    assert getattr(ob, "aligned_face", None) is None, (
                        "Expected aligned_face to be cleared after embedding"
                    )

    assert detected_with_embedding >= 1, "Expected at least one DETECTED obs to have embedding attached"