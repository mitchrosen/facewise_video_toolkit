# facekit/utils/video_reader.py
import numpy as np
import warnings
import av

class VideoReader:
    """
    A PyAV-based video reader for efficient shot/chunk-based frame access.

    Features:
    - Efficient random access using FFmpeg seek.
    - Falls back to sequential decoding if seek is not supported.
    - Returns frames as OpenCV-compatible NumPy arrays (BGR format).
    """

    def __init__(self, video_path: str):
        self.container = av.open(video_path)
        self.stream = self.container.streams.video[0]
        self.fps = float(self.stream.average_rate)
        self.time_base = self.stream.time_base

    def get_frames(self, start_frame: int, end_frame: int):
        """
        Retrieve frames from start_frame to end_frame (inclusive).
        
        Args:
            start_frame (int): First frame index to read.
            end_frame (int): Last frame index to read.

        Returns:
            List[np.ndarray]: A list of frames in BGR format.
        """
        frames = []

        # Compute time range in seconds
        start_time = start_frame / self.fps
        end_time = end_frame / self.fps

        # Try efficient seek if not starting from 0
        if start_frame > 0:
            try:
                seek_pts = int(start_time / self.time_base)
                self.container.seek(seek_pts, any_frame=False, backward=False)
            except (OSError, PermissionError) as e:
                warnings.warn(
                    f"Seek not supported ({e}); falling back to sequential read. "
                    "Performance may be slow for large skips.",
                    RuntimeWarning
                )
                return self._sequential_fallback(start_frame, end_frame)
        
        # Decode frames and filter by timestamp
        for frame in self.container.decode(video=0):
            if frame.time is None:
                continue
            if frame.time < start_time:
                continue
            if frame.time > end_time:
                break
            img = frame.to_ndarray(format="bgr24")
            frames.append(img)

        return frames

    def _sequential_fallback(self, start_frame: int, end_frame: int):
        """Fallback: reopen container and decode sequentially to reach start_frame."""
        # Reopen container fresh
        self.container.close()
        self.container = av.open(self.container.name)
        self.stream = self.container.streams.video[0]

        frames = []
        frame_index = 0
        for frame in self.container.decode(video=0):
            if frame_index < start_frame:
                frame_index += 1
                continue
            if frame_index > end_frame:
                break
            img = frame.to_ndarray(format="bgr24")
            frames.append(img)
            frame_index += 1
        return frames

    def close(self):
        """Close the video container."""
        self.container.close()
