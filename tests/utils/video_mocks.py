import numpy as np
from unittest.mock import MagicMock

def make_pyav_like_frames(num_frames: int, fps: int = 30):
    """
    Return a list of MagicMock 'PyAV-like' frames.
    Each frame has:
      - .time   -> float timestamp in seconds
      - .to_ndarray(format="bgr24") -> np.ndarray image
    Use this when you want to simulate the normal code path in VideoReader.
    """
    frames = []
    for i in range(num_frames):
        frame = MagicMock()
        frame.to_ndarray.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        frame.time = i / float(fps)
        frames.append(frame)
    return frames


def make_frames_without_time(num_frames: int):
    """
    Return a list of MagicMock frames WITHOUT .time (and WITHOUT .pts).
    Each frame still supports .to_ndarray().
    Use this to test VideoReader's fallback behavior when timestamps are missing.
    """
    frames = []
    for _ in range(num_frames):
        frame = MagicMock()
        frame.to_ndarray.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        # no frame.time, no frame.pts
        frames.append(frame)
    return frames
