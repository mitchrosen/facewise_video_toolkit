import cv2
import numpy as np
from typing import List, Tuple

# ArcFace reference template
ARC_FACE_TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

def align_face_for_arcface(image: np.ndarray, landmarks: List[Tuple[float, float]]) -> np.ndarray:
    """
    Align a face to ArcFace's input format (112x112) using similarity transform.

    Args:
        image: Original BGR image (numpy array).
        landmarks: List of 5 (x, y) landmark points for the face.

    Returns:
        Aligned 112x112 RGB image ready for ArcFace embedding.
    """
    if len(landmarks) != 5:
        raise ValueError("Expected 5 facial landmarks for ArcFace alignment.")

    src = np.array(landmarks, dtype=np.float32)
    dst = ARC_FACE_TEMPLATE.copy()
    tform = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)[0]
    aligned = cv2.warpAffine(image, tform, (112, 112), borderValue=0.0)
    aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    return aligned_rgb
