# facekit/detection/face_detector.py
import numpy as np
from typing import List, Tuple, Optional
from numpy.typing import NDArray

class FaceDetector:
    """
    Wrapper class for YOLO-based face detector.
    Provides a consistent interface for face detection across the codebase.
    """

    def __init__(self, detector_model):
        """
        Args:
            detector_model: The YOLO-based detection model (already loaded).
        """
        self.detector_model = detector_model

    def detect_faces_in_frame(
        self,
        frame: NDArray,
        target_size: int = 640
    ) -> Tuple[List[List[int]], List[List[List[int]]], List[float]]:
        """
        Run face detection on a single video frame using model.

        Returns:
            A tuple of:
                - List of bounding boxes
                - List of landmarks
                - List of confidence scores
            Returns empty lists for all three if no faces are detected or an error occurs.
        """
        try:
            results = self.detector_model(frame, target_size=target_size)
            if (
                isinstance(results, tuple)
                and len(results) == 3
                and all(isinstance(r, list) for r in results)
            ):
                return results
        except Exception:
            pass

        return [], [], []

