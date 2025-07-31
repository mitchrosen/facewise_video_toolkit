from fractions import Fraction
import types
import numpy as np
from unittest.mock import MagicMock

class FakeFrame:
    """Minimal PyAV-like frame: has .pts, .time, .to_ndarray()."""
    def __init__(self, idx, img_side=4, pts=None, time=None):
        self.idx = idx
        self.pts = pts
        self.time = time
        self._img = np.full((img_side, img_side, 3), idx, dtype=np.uint8)

    def to_ndarray(self, format="bgr24"):
        return self._img

class FakeStream:
    def __init__(self, average_rate, time_base):
        self.type = "video"
        self.average_rate = average_rate   # Fraction
        self.time_base = time_base         # Fraction
        self.base_rate = average_rate

class FakeContainer:
    """
    Simulates av.open(..) object enough for VideoReader:
    - streams.video[0]
    - seek(pts, any_frame, backward)
    - decode(video=0)
    """
    def __init__(self, frames, fps_num=30, fps_den=1, time_base=Fraction(1, 30)):
        self._frames = frames
        self.streams = types.SimpleNamespace(video=[FakeStream(Fraction(fps_num, fps_den), time_base)])
        self._decode_start_idx = 0
        self._closed = False

    def seek(self, pts, any_frame=False, backward=True):
        tb = self.streams.video[0].time_base
        fps = float(self.streams.video[0].average_rate)
        t_sec = float(pts * tb)
        idx = int(round(t_sec * fps))
        self._decode_start_idx = max(0, min(idx, len(self._frames) - 1))

    def decode(self, video=0):
        for i in range(self._decode_start_idx, len(self._frames)):
            yield self._frames[i]

    def close(self):
        self._closed = True

def make_pts_time_sequence(n, fps=30.0, time_base=Fraction(1, 30),
                           jitter=None, pts_none=False, time_none=False,
                           pts_mock=False, time_mock=False):
    """
    Create n FakeFrame objects. You can:
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
