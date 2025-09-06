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

first_time = True

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

    # DEBUG
    if not hasattr(align_face_for_arcface, "first_time"):
        print(f"[DEBUG] align_face_for_arcface() CALLED — frame {frame_idx}, source={source}")
        align_face_for_arcface.first_time = False

    if frame_idx in {14863, 14864}:
        print(f"\n[DEBUG] --- DETAILED DEBUG FOR FRAME {frame_idx} ---")
        print(f"[DEBUG] Source: {source}")
        print(f"[DEBUG] Raw landmarks: {landmarks}")

        import os
        os.makedirs("debug_output", exist_ok=True)
        print(f"[DEBUG] Frame {frame_idx}: Debug images to be saved to debug_output")

        # Save raw frame
        cv2.imwrite(f"debug_output/frame_{frame_idx}_raw.jpg", image)

        # Copy image to draw on
        img_with_landmarks = image.copy()
        for (x, y) in landmarks:
            if np.isfinite(x) and np.isfinite(y):
                cv2.circle(img_with_landmarks, (int(x), int(y)), 2, (0, 255, 0), -1)

        cv2.imwrite("debug_output/frame_{frame_idx}_landmarks.jpg", img_with_landmarks)


    context = f"Frame {frame_idx}" if frame_idx is not None else "Unknown frame"
    context += f" (source={source})" if source else ""

    if landmarks is None or len(landmarks) != 5:
        print(f"[DEBUG] {context}: Invalid landmarks: None or wrong length")
        return None

    src = np.array(landmarks, dtype=np.float32)
    if src.shape != (5, 2):
        print(f"[DEBUG] {context}: Invalid landmark shape: {src.shape}")
        return None
    if not np.isfinite(src).all():
        print(f"[DEBUG] {context}: Invalid landmark values (non-finite): {src}")
        return None

    dst = ARC_FACE_TEMPLATE.copy()
    tform_result = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)

    # DEBUG
    if frame_idx in {14864, 14864}:
        print(f"[DEBUG] Frame {frame_idx}: tform_result = {tform_result}")

    if tform_result is None or tform_result[0] is None:
        print(f"[DEBUG] {context}: estimateAffinePartial2D failed for landmarks: {landmarks}")
        return None

    tform = tform_result[0]
    aligned = cv2.warpAffine(image, tform, (112, 112), borderValue=0.0)

    # DEBUG
    if frame_idx in {14864, 14864}:
        cv2.imwrite("debug_output/frame_{frame_idx}_aligned_bgr.png", aligned)

    aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)

    # DEBUG
    if frame_idx in {14864, 14864}:
        cv2.imwrite("debug_output/frame_{frame_idx}_aligned_rgb.png", aligned_rgb)

    return aligned_rgb
