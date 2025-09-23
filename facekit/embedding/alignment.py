import cv2
import numpy as np
from typing import List, Tuple, Optional

# ArcFace reference template
ARC_FACE_TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

def align_face_for_arcface(
    image: np.ndarray,
    landmarks: List[Tuple[float, float]],
    frame_idx: Optional[int] = None,
    source: Optional[str] = None
) -> Optional[np.ndarray]:
    """
    Align a face to ArcFace's input format (112x112) using similarity transform.

    Returns:
        Aligned 112x112 RGB image, or None if alignment fails.
    """

    context = f"Frame {frame_idx}" if frame_idx is not None else "Unknown frame"
    context += f" (source={source})" if source else ""

    if landmarks is None or len(landmarks) != 5:
        return None

    src = np.array(landmarks, dtype=np.float32)
    if src.shape != (5, 2):
        return None
    if not np.isfinite(src).all():
        return None

    dst = ARC_FACE_TEMPLATE.copy()
    tform_result = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)

    if tform_result is None or tform_result[0] is None:
        return None

    tform = tform_result[0]
    aligned = cv2.warpAffine(image, tform, (112, 112), borderValue=0.0)

    aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)

    return aligned_rgb
