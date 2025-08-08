import cv2
import numpy as np

def propagate_landmarks_via_optical_flow(prev_gray_roi, curr_gray_roi, prev_landmarks):
    """
    Use Lucas-Kanade optical flow to propagate facial landmarks from one frame to the next.

    Args:
        prev_gray_roi (np.ndarray): Grayscale image of previous ROI (usually a face crop).
        curr_gray_roi (np.ndarray): Grayscale image of current ROI (same dimensions as prev).
        prev_landmarks (np.ndarray): Landmark coordinates in previous ROI, shape (5, 2), dtype float32.

    Returns:
        new_landmarks (np.ndarray): Propagated landmarks in current ROI, shape (5, 2), dtype float32.
        status (np.ndarray): Status flags for each point (1 if found, 0 otherwise).
    """
    if prev_gray_roi is None or curr_gray_roi is None or prev_landmarks is None:
        raise ValueError("Missing input for optical flow propagation.")

    if not isinstance(prev_landmarks, np.ndarray):
        prev_landmarks = np.array(prev_landmarks, dtype=np.float32)
    else:
        prev_landmarks = prev_landmarks.astype(np.float32)

    if prev_landmarks.ndim != 2 or prev_landmarks.shape[1] != 2:
        raise ValueError(f"prev_landmarks must be shape (N, 2); got {prev_landmarks.shape}")

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray_roi, curr_gray_roi,
        prev_landmarks.reshape(-1, 1, 2), None,
        winSize=(15, 15), maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    new_landmarks = next_pts.reshape(-1, 2)
    return new_landmarks, status.reshape(-1)
