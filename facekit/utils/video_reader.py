# facekit/utils/video_reader.py
import warnings
import av
import numpy as np

class VideoReader:
    """
    A PyAV-based video reader for efficient shot/chunk-based frame access.

    Features:
    - Efficient random access using FFmpeg seek.
    - Falls back to sequential decoding if seek is not supported.
    - Returns frames as OpenCV-compatible NumPy arrays (BGR format).
    """

    def __init__(self, video_path: str):
        self._source_path = video_path
        self.container = av.open(self._source_path)
        self.stream = self.container.streams.video[0]
        # Defensive: average_rate can be None for some streams
        self.fps = float(self.stream.average_rate) if self.stream.average_rate else float(self.stream.base_rate)
        self.time_base = self.stream.time_base

    def get_frames(self, start_frame: int, end_frame: int):
        """
        Retrieve frames from start_frame to end_frame (inclusive).
        Returns: List[np.ndarray] in BGR format.
        """
        frames = []
        target_count = end_frame - start_frame + 1
        if target_count <= 0:
            return frames

        # Convert frame indices to time/PTS
        start_time = start_frame / self.fps
        start_pts = int(round(start_time / self.time_base))

        # Seek slightly BEFORE the shot start, then decode forward
        preroll_frames = 5  # small safety margin
        seek_time = max(0.0, start_time - preroll_frames / self.fps)
        seek_pts = int(round(seek_time / self.time_base))

        try:
            # Backward seek → keyframe at/before seek_pts
            self.container.seek(seek_pts, any_frame=False, backward=True)
        except (OSError, PermissionError) as e:
            warnings.warn(f"Seek not supported ({e}); falling back to sequential read.", RuntimeWarning)
            return self._sequential_fallback(start_frame, end_frame)

        # Fallback counters for mocks / edge streams
        to_skip_by_count = max(0, int(round((start_time - seek_time) * self.fps)))
        eps = 1e-9

        emitted = 0
        for frame in self.container.decode(video=0):
            # Skip until we reach the requested start
            usable_pts = isinstance(getattr(frame, "pts", None), (int, float))
            usable_time = isinstance(getattr(frame, "time", None), (int, float))

            if usable_pts:
                if frame.pts < start_pts:
                    continue
            elif usable_time:
                if frame.time + eps < start_time:
                    continue
            else:
                if to_skip_by_count > 0:
                    to_skip_by_count -= 1
                    continue

            # Emit exactly target_count frames
            frames.append(frame.to_ndarray(format="bgr24"))
            emitted += 1
            if emitted >= target_count:
                break

        return frames

    def _sequential_fallback(self, start_frame: int, end_frame: int):
        """Fallback: reopen container and decode sequentially to reach start_frame."""
        # Reopen container fresh from the stored source path
        try:
            self.container.close()
        except Exception:
            pass
        self.container = av.open(self._source_path)
        self.stream = self.container.streams.video[0]

        frames = []
        target_count = end_frame - start_frame + 1
        if target_count <= 0:
            return frames

        frame_index = 0
        for frame in self.container.decode(video=0):
            if frame_index < start_frame:
                frame_index += 1
                continue
            frames.append(frame.to_ndarray(format="bgr24"))
            frame_index += 1
            if len(frames) >= target_count:
                break
        return frames

    def close(self):
        """Close the video container."""
        self.container.close()
