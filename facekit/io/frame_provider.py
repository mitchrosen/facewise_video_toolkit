from __future__ import annotations
from typing import Protocol, Optional, List, Dict, runtime_checkable
import numpy as np

from facekit.utils.video_reader import VideoReader

@runtime_checkable
class FrameProvider(Protocol):
    # Sequential
    def reset_to_frame(self, start_idx: int) -> None: ...
    def next(self) -> Optional[np.ndarray]: ...
    # Random access
    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]: ...
     # Lifecycle
    def close(self) -> None: ...
    # Metadata
    def fps(self) -> Optional[float]: ...
    def size(self) -> Optional[tuple[int, int]]: ...
    def total_frames(self) -> Optional[int]: ...

class ReaderCoordinator(FrameProvider):
    """
    Pure PyAV coordinator:
      - sequential reads: chunked via VideoReader.get_frames(start, end)
      - random access:    single-frame via VideoReader.get_frames(i, i)
    """

    def __init__(self, video_path: str, seq_chunk: int = 64, lru_size: int = 256):
        self._path = video_path
        self._vid_reader = VideoReader(video_path)  # PyAV-backed, frame-accurate
        self._seq_chunk = max(1, int(seq_chunk))

        # sequential cursor & buffer
        self._cur_idx: Optional[int] = None
        self._buf: List[np.ndarray] = []
        self._buf_head_idx: Optional[int] = None  # frame_idx of _buf[0]

        # small LRU for random single-frame lookups
        self._lru_size = max(0, int(lru_size))
        self._lru: Dict[int, np.ndarray] = {}
        self._order: List[int] = []
        self._total_frames_cache: Optional[int] = None

    # ---------- sequential path ----------

    def reset_to_frame(self, start_idx: int) -> None:
        if start_idx < 0:
            raise ValueError(f"start_idx must be >= 0, got {start_idx}")

        # Reset sequential buffer to start at the exact frame
        self._cur_idx = int(start_idx)
        self._buf = []
        self._buf_head_idx = None
        # lazily filled on first next()

    def _fill_seq_buffer(self) -> None:
        assert self._cur_idx is not None
        start = self._cur_idx
        end = start + self._seq_chunk - 1
        frames = self._vid_reader.get_frames(start, end)
        self._buf = frames or []
        self._buf_head_idx = start if self._buf else None

    def next(self) -> Optional[np.ndarray]:
        # Serve from buffer; when empty, fetch next chunk
        if self._cur_idx is None:
            return None
        if not self._buf:
            self._fill_seq_buffer()
            if not self._buf:
                return None  # EOF
        # pop first
        frame = self._buf.pop(0)
        # advance cursor/head
        if self._buf_head_idx is not None:
            self._buf_head_idx += 1
        self._cur_idx += 1
        return frame

    # ---------- random access (sparse) ----------

    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        i = int(frame_idx)
        if i < 0:
            raise ValueError(f"frame_idx must be >= 0, got {i}")

        # if self._lru_size > 0 and i in self._lru:
        #     # refresh LRU
        #     self._order.remove(i)
        #     self._order.append(i)
        #     return self._lru[i]

        # frames = self._vid_reader.get_frames(i, i)


        tf = self.total_frames()
        if tf is not None and i >= tf:
            return None  # or raise
        # (cache logic unchanged)
        frames = self._vid_reader.get_frames(i, i)

        frame = frames[0] if frames else None

        if frame is not None and self._lru_size > 0:
            self._lru[i] = frame
            self._order.append(i)
            if len(self._order) > self._lru_size:
                evict = self._order.pop(0)
                self._lru.pop(evict, None)

        return frame

    # ---------- helpers ----------

    def fps(self) -> Optional[float]:
        return getattr(self._vid_reader, "fps", None)

    def size(self) -> Optional[tuple[int, int]]:
        w = getattr(self._vid_reader, "width", None)
        h = getattr(self._vid_reader, "height", None)
        return (w, h) if isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0 else None
    
    def total_frames(self) -> Optional[int]:
        """Best-effort total frame count with layered fallbacks:
        1) cached
        2) PyAV stream.frames
        3) duration * average_rate
        4) OpenCV CAP_PROP_FRAME_COUNT
        5) full decode scan (slow, but correct)
        """
        if self._total_frames_cache is not None:
            return self._total_frames_cache

        # 1) If VideoReader exposes a total_frames, trust it.
        tf = getattr(self._vid_reader, "total_frames", None)
        if isinstance(tf, (int, np.integer)) and tf > 0:
            self._total_frames_cache = int(tf)
            return self._total_frames_cache

        # 2) PyAV stream.frames (may be 0 or None on some containers)
        try:
            stream = getattr(self._vid_reader, "stream", None)
            if stream is not None:
                n = getattr(stream, "frames", None)
                if isinstance(n, (int, np.integer)) and n > 0:
                    self._total_frames_cache = int(n)
                    return self._total_frames_cache
        except Exception:
            pass

        # 3) duration * average_rate (estimate; often accurate)
        try:
            container = getattr(self._vid_reader, "container", None)
            stream = getattr(self._vid_reader, "stream", None)
            if container is not None and stream is not None:
                dur = getattr(container, "duration", None)  # in time_base units
                tb = getattr(stream, "time_base", None)
                avg = getattr(stream, "average_rate", None) or getattr(stream, "base_rate", None)
                if dur is not None and tb is not None and avg:
                    # convert duration to seconds, multiply by fps
                    seconds = float(dur * tb)
                    est = int(round(seconds * float(avg)))
                    if est > 0:
                        self._total_frames_cache = est
                        return self._total_frames_cache
        except Exception:
            pass

        # 4) OpenCV fallback (quick metadata)
        try:
            import cv2
            cap = cv2.VideoCapture(self._path)
            try:
                if cap.isOpened():
                    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if n > 0:
                        self._total_frames_cache = n
                        return n
            finally:
                cap.release()
        except Exception:
            pass

        # 5) Full scan (slow but definitive). Use a fresh container so we don't
        #    disturb the main reader's decode state.
        try:
            import av
            with av.open(self._path) as c:
                # Some codecs/containers decode faster with explicit thread count;
                # leave defaults
                cnt = 0
                for _frame in c.decode(video=0):
                    cnt += 1
            self._total_frames_cache = cnt  # accept any count EVEN 0
            return cnt
        except Exception as e:
            raise RuntimeError( f"Could not determine total frame count for video {self._path!r}. "
                               " Error during full decode scan. " ) from e
            
    def close(self) -> None:
        # VideoReader likely closes via GC, but be explicit if it has a close()
        close = getattr(self._vid_reader, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
        self._buf.clear()
        self._lru.clear()
        self._order.clear()

    # --------- support for context manager protocol ----------
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        self.close()
