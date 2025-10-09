import json
import numpy as np
import pytest
from fractions import Fraction
from unittest.mock import patch
from pathlib import Path

from facekit.pipeline.track_across_segments import track_across_segments
from facekit.io.frame_provider import FrameProvider, ReaderCoordinator


# ---------- Minimal fake FrameProvider (sequential only) ----------
class FakeProvider(FrameProvider):
    def __init__(self, total_frames: int, shape=(64, 64, 3)):
        self._total = total_frames
        self._shape = shape
        self._cursor = None
        self._closed = False

    def reset_to_frame(self, start_idx: int) -> None:
        self._cursor = int(start_idx)

    def next(self):
        if self._cursor is None or self._cursor >= self._total:
            return None
        frame = np.zeros(self._shape, dtype=np.uint8)
        self._cursor += 1
        return frame

    def get_frame(self, frame_idx: int):
        if 0 <= frame_idx < self._total:
            return np.zeros(self._shape, dtype=np.uint8)
        return None

    def close(self) -> None:
        self._closed = True

    # helper for assertions
    @property
    def closed(self) -> bool:
        return self._closed


# ------------------------------ Tests ------------------------------

def test_track_across_segments_with_provider(tmp_path: Path, monkeypatch):
    """Ensure we can pass a FrameProvider instance, and the callee does NOT close it."""
    # Shots: a single shot [0..4]
    shot_json = tmp_path / "shots.json"
    shot_json.write_text(json.dumps({"shots": [{"shot_number": 1, "first_frame": 0, "last_frame": 4}]}))

    # Always-detect stub so we exercise detection→aggregate→bootstrap path every frame
    class FakeDetector:
        def detect_faces_in_frame(self, frame, target_size=640):
            # one box, trivial landmarks/conf
            return [(10, 10, 30, 30)], [[(12, 12)] * 5], [0.99]

    # Alignment: return a dummy crop
    monkeypatch.setattr(
        "facekit.pipeline.track_across_segments.align_face_for_arcface",
        lambda frame, lm, frame_idx=None, source=None: np.zeros((112, 112, 3), dtype=np.uint8),
    )

    # Embedder: return valid (K,512) float32
    class FakeEmbedder:
        def get_embedding_batch(self, aligned_faces, batch_size=32):
            K = len(aligned_faces)
            return np.ones((K, 512), dtype=np.float32)

    # Pre-instantiated provider (5 frames total)
    prov = FakeProvider(total_frames=5)

    tracks = track_across_segments(
        frame_source=prov,                      # Pass provider directly
        shot_json_path=str(shot_json),
        detector=FakeDetector(),
        embedder=FakeEmbedder(),
        detect_interval=1,                      # detect every frame
    )

    # Basic sanity: at least one finalized track with a segment_id
    assert len(tracks) >= 1
    assert all(t.segment_id is not None for t in tracks)

    # Critical: callee must NOT close caller-owned provider
    assert prov.closed is False


def test_track_across_segments_invalid_frame_source_raises(tmp_path: Path):
    """Passing an invalid frame_source (e.g., None) should raise TypeError."""
    shot_json = tmp_path / "shots.json"
    shot_json.write_text(json.dumps({"shots": [{"shot_number": 1, "first_frame": 0, "last_frame": 0}]}))

    class FakeDetector:
        def detect_faces_in_frame(self, frame, target_size=640):
            return None

    class FakeEmbedder:
        def get_embedding_batch(self, aligned_faces, batch_size=32):
            return np.zeros((len(aligned_faces), 512), dtype=np.float32)

    with pytest.raises(TypeError):
        # frame_source=None should hit TypeError branch
        track_across_segments(
            frame_source=None,                   # invalid by design
            shot_json_path=str(shot_json),
            detector=FakeDetector(),
            embedder=FakeEmbedder(),
        )


# ---- Minimal PyAV-like fakes for ReaderCoordinator path ----

class _FakeFrame:
    def __init__(self, pts, t, h=64, w=64):
        self.pts = pts
        self.time = t
        self._h = h
        self._w = w

    def to_ndarray(self, format="bgr24"):
        # tiny deterministic BGR frame (zeros are fine for this test)
        return np.zeros((self._h, self._w, 3), dtype=np.uint8)


def _make_container(total=5, fps_num=30, fps_den=1, w=64, h=64):
    """
    Return a fresh *context-managed* fake av.open() container each call with:
      - streams.video[0] exposing width/height/average_rate/base_rate/time_base/frames
      - seek(...): no-op, present so VideoReader can call it
      - decode(video=0): yields a fresh generator of frames with numeric pts/time
    """
    class _C:
        def __init__(self):
            # build a minimal "stream" object
            s = type("V", (), {})()
            s.type = "video"
            s.frames = total
            s.width = w
            s.height = h
            s.average_rate = Fraction(fps_num, fps_den)
            s.base_rate = Fraction(fps_num, fps_den)
            s.time_base = Fraction(1, fps_num)
            self.streams = type("S", (), {"video": [s]})()

        # context manager API
        def __enter__(self): return self
        def __exit__(self, exc_type, exc, tb): return False

        # allow seek signature used by VideoReader
        def seek(self, *a, **k):
            return None

        # IMPORTANT: return a *fresh* generator every call
        def decode(self, video=0):
            s = self.streams.video[0]
            fps = float(s.average_rate)
            tb = float(s.time_base)
            for i in range(s.frames):
                pts = int(round((i / fps) / tb))
                t = i / fps
                yield _FakeFrame(pts=pts, t=t, h=s.height, w=s.width)

    return _C()


def test_track_across_segments_with_readercoordinator_instance_not_closed(tmp_path: Path, monkeypatch):
    """When passing a ReaderCoordinator instance, the callee must NOT close it."""
    # Shots: single shot [0..4]
    shot_json = tmp_path / "shots.json"
    shot_json.write_text(json.dumps({"shots": [{"shot_number": 1, "first_frame": 0, "last_frame": 4}]}))

    # Detector: always returns one face
    class FakeDetector:
        def detect_faces_in_frame(self, frame, target_size=640):
            return [(10, 10, 30, 30)], [[(12, 12)] * 5], [0.99]

    # Aligner: deterministic dummy crop
    monkeypatch.setattr(
        "facekit.pipeline.track_across_segments.align_face_for_arcface",
        lambda frame, lm, frame_idx=None, source=None: np.zeros((112, 112, 3), dtype=np.uint8),
    )

    # Embedder: valid (K, 512) float32
    class FakeEmbedder:
        def get_embedding_batch(self, aligned_faces, batch_size=32):
            K = len(aligned_faces)
            return np.ones((K, 512), dtype=np.float32)

    # Patch av.open so ReaderCoordinator/VideoReader see our fresh fake container each call
    with patch("facekit.utils.video_reader.av.open",
               side_effect=lambda *a, **k: _make_container(total=5)):
        # Construct a real ReaderCoordinator (caller-owned)
        prov = ReaderCoordinator("dummy.mp4")

        # Spy on .close() to ensure the callee does not call it
        closed = {"called": False}
        orig_close = prov.close

        def spy_close():
            closed["called"] = True
            return orig_close()

        # Replace the method on this instance only
        import types
        prov.close = types.MethodType(lambda self: spy_close(), prov)

        # Run the pipeline using the provider directly
        tracks = track_across_segments(
            frame_source=prov,
            shot_json_path=str(shot_json),
            detector=FakeDetector(),
            embedder=FakeEmbedder(),
            detect_interval=1,  # detect every frame for simplicity
        )

        # Basic sanity: got tracks with segment_ids
        assert len(tracks) >= 1
        assert all(t.segment_id is not None for t in tracks)

        # Critical: callee must NOT close a caller-owned provider
        assert closed["called"] is False
