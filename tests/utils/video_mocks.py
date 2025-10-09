import numpy as np
from unittest.mock import MagicMock

def make_pyav_like_frames(num_frames: int, fps: int = 30):
    """
    Return a list of MagicMock 'PyAV-like' frames.
    Each frame has:
      - .time   -> float timestamp in seconds
      - .to_ndarray(format="bgr24") -> np.ndarray image
    Use for simulating the normal code path in VideoReader.
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


#--------- fake VideoReader -----------
def synthetic_frame(i: int, w=64, h=48) -> np.ndarray:
    """
    Deterministic BGR frame for index i.
    - Channel 0 (B): constant = i % 256
    - Channel 1 (G): horizontal gradient
    - Channel 2 (R): vertical gradient
    This makes every i visually and byte-wise unique/stable.
    """
    i8 = np.uint8(i & 0xFF)
    g = np.linspace(0, 255, w, dtype=np.uint8)[None, :].repeat(h, axis=0)
    r = np.linspace(0, 255, h, dtype=np.uint8)[:, None].repeat(w, axis=1)
    b = np.full((h, w), i8, dtype=np.uint8)
    return np.dstack([b, g, r])  # BGR

class FakeVideoReader:
    """
    Absolute 0-based, inclusive get_frames() with deterministic frames.
    Exposes attributes that ReaderCoordinator.total_frames() checks.
    """
    def __init__(self, *_a, total=200, w=64, h=48, fps=30.0, **_k):
        self.total_frames = int(total)
        self.width = int(w)
        self.height = int(h)
        self.fps = float(fps)

    def get_frames(self, start_frame: int, end_frame: int):
        start = max(0, int(start_frame)); 
        end = int(end_frame)
        if end < start: return []
        end = min(end, self.total_frames - 1)
        return [synthetic_frame(i) for i in range(start, end+1)]

    def close(self):  # parity
        pass