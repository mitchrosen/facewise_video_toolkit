import cv2
import pytest
from pathlib import Path
from unittest import mock
import numpy as np
from facekit.pipeline.draw_tracks import draw_tracks_on_video
from facekit.tracking.face_structures import FaceTrack, FaceObservation, FaceTrack

@mock.patch("cv2.putText")
@mock.patch("cv2.rectangle")
@mock.patch("cv2.VideoWriter")
@mock.patch("cv2.VideoCapture")
def test_draw_tracks_on_video_mocks(VideoCaptureMock, VideoWriterMock, rectangle_mock, puttext_mock, tmp_path):
    # Mock frame data
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # Mock VideoCapture
    video_capture_instance = mock.Mock()
    video_capture_instance.isOpened.return_value = True
    video_capture_instance.get.side_effect = lambda x: {
        3: 100,  # width
        4: 100,  # height
        5: 30.0,  # fps
        7: 5,     # total frames
    }[x]

    def read_side_effect():
        for _ in range(5):
            yield (True, dummy_frame.copy())
        while True:
            yield (False, None)

    read_gen = read_side_effect()
    video_capture_instance.read.side_effect = lambda: next(read_gen)

    VideoCaptureMock.return_value = video_capture_instance

    # Mock VideoWriter
    video_writer_instance = mock.Mock()
    VideoWriterMock.return_value = video_writer_instance

    # Define dummy tracks
    tracks = [
        FaceTrack(
            shot_id=1,
            track_id=1,
            observations=[
                FaceObservation(frame_idx=0, bbox=(10, 10, 20, 20), confidence=0.9),
                FaceObservation(frame_idx=2, bbox=(12, 12, 22, 22), confidence=0.95)
            ]
        )
    ]

    output_path = tmp_path / "output.mp4"
    draw_tracks_on_video("dummy.mp4", str(output_path), tracks)

    # Assertions
    assert VideoCaptureMock.called
    assert VideoWriterMock.called
    assert video_writer_instance.write.call_count == 5  # total frames
    assert rectangle_mock.call_args_list == [
        mock.call(mock.ANY, (10, 10), (20, 20), mock.ANY, 2),
        mock.call(mock.ANY, (12, 12), (22, 22), mock.ANY, 2),
    ]
    assert puttext_mock.call_count >= 2

    # Confirm correct arguments to drawing functions
    rectangle_mock.assert_any_call(mock.ANY, (10, 10), (20, 20), mock.ANY, 2)
    # Grab the expected label positions dynamically
    bbox = tracks[0].observations[1].bbox
    labels_drawn = [call[0][1] for call in puttext_mock.call_args_list]
    assert "1" in labels_drawn

def test_draw_tracks_forces_avi_suffix(monkeypatch, tmp_path):
    created = {}

    # ---- Mocks ----
    class DummyCapture:
        def __init__(self, *_):
            self._read_calls = 0
        def isOpened(self): return True
        def get(self, prop):
            if prop == cv2.CAP_PROP_FRAME_WIDTH:  return 640
            if prop == cv2.CAP_PROP_FRAME_HEIGHT: return 360
            if prop == cv2.CAP_PROP_FPS:          return 30
            if prop == cv2.CAP_PROP_FRAME_COUNT:  return 1
            return 0
        def read(self):
            # one black frame, then EOF
            if self._read_calls == 0:
                self._read_calls += 1
                return True, np.zeros((360, 640, 3), dtype=np.uint8)
            return False, None
        def release(self): pass

    class DummyWriter:
        def __init__(self, filename, fourcc, fps, size):
            created["filename"] = filename
            created["fourcc"] = fourcc
            created["fps"] = fps
            created["size"] = size
            self._writes = 0
        def write(self, frame): self._writes += 1
        def release(self): pass

    monkeypatch.setattr(cv2, "VideoCapture", DummyCapture)
    monkeypatch.setattr(cv2, "VideoWriter", DummyWriter)
    monkeypatch.setattr(cv2, "VideoWriter_fourcc", lambda *args: 1234)

    # minimal track content is optional for this test; empty list is fine
    tracks = []

    out_mp4 = tmp_path / "output.mp4"
    draw_tracks_on_video("input.mp4", str(out_mp4), tracks)

    # Assert that VideoWriter was called with an .avi path
    assert created["filename"].endswith(".avi")
    # Also sanity‑check the fourcc and geometry were passed through
    assert created["fourcc"] == 1234
    assert created["fps"] == 30
    assert created["size"] == (640, 360)


def test_draw_tracks_keeps_avi_suffix(monkeypatch, tmp_path):
    created = {}

    class DummyCapture:
        def __init__(self, *_): pass
        def isOpened(self): return True
        def get(self, prop):
            if prop == cv2.CAP_PROP_FRAME_WIDTH:  return 640
            if prop == cv2.CAP_PROP_FRAME_HEIGHT: return 360
            if prop == cv2.CAP_PROP_FPS:          return 30
            if prop == cv2.CAP_PROP_FRAME_COUNT:  return 1
        def read(self): return False, None
        def release(self): pass

    class DummyWriter:
        def __init__(self, filename, fourcc, fps, size):
            created["filename"] = filename
        def write(self, frame): pass
        def release(self): pass

    monkeypatch.setattr(cv2, "VideoCapture", DummyCapture)
    monkeypatch.setattr(cv2, "VideoWriter", DummyWriter)
    monkeypatch.setattr(cv2, "VideoWriter_fourcc", lambda *args: 1234)

    out_avi = tmp_path / "already.avi"
    draw_tracks_on_video("input.mp4", str(out_avi), [])

    assert created["filename"].endswith("already.avi")
