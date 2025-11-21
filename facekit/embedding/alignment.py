import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

# ArcFace 112×112 reference (left eye, right eye, nose, left mouth, right mouth)
ARC_FACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

def align_face_for_arcface(
    image: np.ndarray,
    landmarks: List[Tuple[float, float]],
    frame_idx: Optional[int] = None,
    source: Optional[str] = None,
    *,
    return_meta: bool = False,
) -> Optional[np.ndarray] | Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Align a face to ArcFace's input format (112x112) using a similarity transform.

    Assumes input `image` is BGR (OpenCV convention). Returns an RGB uint8 crop.

    Landmarks order must be: left-eye, right-eye, nose, left-mouth, right-mouth.
    """
    meta: Dict[str, Any] = {"ok": False, "reason": None, "inliers": None, "M": None}

    # ---- Basic input validation ------------------------------------------------
    if image is None or not isinstance(image, np.ndarray):
        meta["reason"] = "image_none_or_not_ndarray"
        return (None, meta) if return_meta else None

    if image.ndim == 2:  # grayscale → make 3-channel
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3 or image.shape[2] != 3:
        meta["reason"] = f"bad_shape_{getattr(image, 'shape', None)}"
        return (None, meta) if return_meta else None

    if image.dtype != np.uint8:
        # Coerce gently to uint8 (handles float [0,1] or [0,255])
        img = image.astype(np.float32, copy=False)
        if img.max() <= 1.01:
            img = img * 255.0
        image = np.clip(img, 0, 255).astype(np.uint8, copy=False)

    if landmarks is None or len(landmarks) != 5:
        meta["reason"] = "need_5_landmarks"
        return (None, meta) if return_meta else None

    src = np.asarray(landmarks, dtype=np.float32).reshape(5, 2)
    if not np.isfinite(src).all():
        meta["reason"] = "non_finite_landmarks"
        return (None, meta) if return_meta else None

    # ---- Estimate similarity (affine partial) ---------------------------------
    # RANSAC is slightly more forgiving than LMEDS for a single bad point.
    # Note: RANSAC params only apply when method=cv2.RANSAC.
    dst = ARC_FACE_TEMPLATE
    M, inliers = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=2000,
        confidence=0.99,
        refineIters=10,
    )
    meta["M"] = M
    meta["inliers"] = inliers

    if M is None:
        meta["reason"] = "estimateAffine_failed"
        return (None, meta) if return_meta else None

    # ---- Warp to 112x112, BGR → RGB ------------------------------------------
    aligned = cv2.warpAffine(
        image,
        M,
        (112, 112),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),  # black padding
    )

    aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    meta["ok"] = True

    return (aligned_rgb, meta) if return_meta else aligned_rgb
