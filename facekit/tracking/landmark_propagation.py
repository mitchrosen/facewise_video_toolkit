import numpy as np


def propagate_landmarks_by_bbox_transform(
    *,
    prev_landmarks,
    prev_bbox,
    curr_bbox,
):
    if prev_landmarks is None:
        return None

    prev_landmarks = np.asarray(prev_landmarks, dtype=np.float32)
    if prev_landmarks.shape != (5, 2):
        return None

    if prev_bbox is None or curr_bbox is None:
        return None

    px1, py1, px2, py2 = prev_bbox
    cx1, cy1, cx2, cy2 = curr_bbox

    prev_w = float(px2 - px1)
    prev_h = float(py2 - py1)
    curr_w = float(cx2 - cx1)
    curr_h = float(cy2 - cy1)

    if prev_w <= 0.0 or prev_h <= 0.0:
        return None
    if curr_w <= 0.0 or curr_h <= 0.0:
        return None

    x_norm = (prev_landmarks[:, 0] - float(px1)) / prev_w
    y_norm = (prev_landmarks[:, 1] - float(py1)) / prev_h

    out = np.empty((5, 2), dtype=np.float32)
    out[:, 0] = float(cx1) + x_norm * curr_w
    out[:, 1] = float(cy1) + y_norm * curr_h
    return out