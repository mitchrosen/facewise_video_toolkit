from fractions import Fraction
import numpy as np
from unittest.mock import MagicMock

# ---- tiny image generator (or import from tests.utils.fakes) ----
def synthetic_frame(i: int, w=64, h=48):
    i8 = np.uint8(i & 0xFF)
    g = np.linspace(0, 255, w, dtype=np.uint8)[None, :].repeat(h, 0)
    r = np.linspace(0, 255, h, dtype=np.uint8)[:, None].repeat(w, 1)
    b = np.full((h, w), i8, dtype=np.uint8)
    return np.dstack([b, g, r])  # BGR

class FakeFrame:
    """Minimal PyAV-like frame: has .pts, .time, .to_ndarray()."""
    def __init__(self, i, pts=None, time=None, w=64, h=48):
        self.pts = pts
        self.time = time
        self._i = i; self._w = w; self._h = h
    def to_ndarray(self, format="bgr24"):
        return synthetic_frame(self._i, self._w, self._h)

class FakeStream:
    def __init__(self, width=640, height=360, fps_num=30, fps_den=1,
                 time_base=Fraction(1,30), frames=0):
        self.width = width
        self.height = height
        self.average_rate = Fraction(fps_num, fps_den)
        self.base_rate = self.average_rate
        self.time_base = time_base
        self.frames = int(frames)  # needed by some code paths

class FakeContainer:
    """
    Simulates av.open(..) object enough for VideoReader:
      - streams.video[0]
      - seek(pts, any_frame, backward)
      - decode(video=0)
      - context manager support
    """
    def __init__(self, frames, fps_num=30, fps_den=1, time_base=Fraction(1, 30),
                 width=640, height=360):
        self._frames = frames
        self.streams = type("S", (), {
            "video": [FakeStream(width, height, fps_num, fps_den, time_base, len(frames))]
        })
        self._seeked = False

    def __enter__(self):  # support: with av.open(...) as c:
        return self
    def __exit__(self, exc_type, exc, tb):
        return False

    def decode(self, video=0):
        # Simple behavior is fine; VideoReader filters by pts/time after seek
        for f in self._frames:
            yield f

    def seek(self, *a, **k):
        self._seeked = True

    def close(self):
        pass

def make_pts_time_sequence(n, fps=30.0, time_base=Fraction(1, 30),
                           jitter=None, pts_none=False, time_none=False,
                           pts_mock=False, time_mock=False):
    """
    Create n FakeFrame objects. One can:
      - add small time jitter near edges,
      - force pts/time to None or MagicMock to exercise fallbacks.
    """
    frames = []
    for i in range(n):
        t = i / fps
        if jitter:
            t = t + jitter(i)
        if pts_none:
            pts = None
        elif pts_mock:
            pts = MagicMock()
        else:
            pts = int(round(t / time_base))
        if time_none:
            tm = None
        elif time_mock:
            tm = MagicMock()
        else:
            tm = t
        frames.append(FakeFrame(i, pts=pts, time=tm))
    return frames
