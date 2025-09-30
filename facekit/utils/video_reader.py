# facekit/utils/video_reader.py

import av
import warnings
from numbers import Number
import numpy as np

def _is_num(x): return isinstance(x, Number)

class VideoReader:
    """
    Exact 0-based random access using PTS.
    get_frames(start,end) is inclusive and frame-accurate when PTS exist.
    """

    def __init__(self, video_path: str):
        self.path = video_path
        self.width = None
        self.height = None
        self.fps = None
        self.time_base = None
        self._pts_index = None
        self.total_frames = None
        self._build_meta_and_pts_index()

    def _build_meta_and_pts_index(self):
        # 1) read stream meta
        with av.open(self.path) as c:
            s = c.streams.video[0]
            self.width  = getattr(s, "width",  None)
            self.height = getattr(s, "height", None)
            rate = getattr(s, "average_rate", None) or getattr(s, "base_rate", None)
            self.fps = float(rate) if rate else None
            self.time_base = s.time_base
        # 2) build frame_index → pts map
        pts = []
        with av.open(self.path) as c:
            for fr in c.decode(video=0):
                pts.append(fr.pts)
        self.total_frames = len(pts)
        # Require numeric PTS for exact mapping
        if pts and all(isinstance(p, (int, np.integer)) for p in pts):
            self._pts_index = np.asarray(pts, dtype=np.int64)
        else:
            self._pts_index = None  # mark unusable; will fall back

    def get_frames(self, start_frame: int, end_frame: int):
        start = max(0, int(start_frame))
        end   = max(start, int(end_frame))
        need  = end - start + 1
        if need <= 0:
            return []

        with av.open(self.path) as c:
            s = c.streams.video[0]

            # Lazy meta if index wasn’t built
            if self.width is None:
                self.width  = getattr(s, "width",  None)
                self.height = getattr(s, "height", None)
                rate = getattr(s, "average_rate", None) or getattr(s, "base_rate", None)
                self.fps = float(rate) if rate else None
                self.time_base = s.time_base

            fps = self.fps
            tb  = self.time_base

            use_pts = (self._pts_index is not None) and (end < len(self._pts_index))

            # ---------- Exact PTS seek (preferred) ----------
            if use_pts:
                target_pts = int(self._pts_index[start])
                try:
                    c.seek(target_pts, any_frame=False, backward=True, stream=s)
                except Exception:
                    warnings.warn("Seek not supported", RuntimeWarning, stacklevel=2)
                    return self._count_fallback(c, start, need)

                out = []
                for fr in c.decode(video=0):
                    p = getattr(fr, "pts", None)
                    if not isinstance(p, (int, np.integer)):
                        # PTS disappeared midstream → fall back to count
                        return self._count_fallback(c, start, need)
                    if p < target_pts:
                        continue
                    out.append(fr.to_ndarray(format="bgr24"))
                    if len(out) >= need:
                        return out
                # EOF before enough frames → count fallback
                return self._count_fallback(c, start, need)

            # ---------- Time-based seek (when PTS unusable) ----------
            if fps and tb:
                target_t = start / fps
                try:
                    est_pts = int(round(target_t / float(tb)))
                    c.seek(est_pts, any_frame=False, backward=True, stream=s)
                except Exception:
                    warnings.warn("Seek not supported", RuntimeWarning, stacklevel=2)
                    return self._count_fallback(c, start, need)

                out = []
                for fr in c.decode(video=0):
                    fpts  = getattr(fr, "pts", None)
                    ftime = getattr(fr, "time", None)
                    if _is_num(ftime):
                        if ftime + 1e-9 < target_t:
                            continue
                    elif _is_num(fpts):
                        if float(fpts) * float(tb) + 1e-9 < target_t:
                            continue
                    else:
                        # No filterable metadata → count fallback
                        return self._count_fallback(c, start, need)
                    out.append(fr.to_ndarray(format="bgr24"))
                    if len(out) >= need:
                        return out
                return self._count_fallback(c, start, need)

            # ---------- No usable meta → pure count ----------
            return self._count_fallback(c, start, need)

    def _count_fallback(self, container, start, need):
        # rewind best-effort, then count
        try:
            s = container.streams.video[0]
            container.seek(0, any_frame=True, backward=True, stream=s)
        except Exception:
            pass
        out, idx = [], 0
        for fr in container.decode(video=0):
            if idx < start:
                idx += 1; continue
            out.append(fr.to_ndarray(format="bgr24"))
            idx += 1
            if len(out) >= need:
                break
        return out

    def close(self):
        pass
