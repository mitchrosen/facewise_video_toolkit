from __future__ import annotations
import os
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import pytest
import cv2

from facekit.detection.yolo5face_model import load_yolo5face_model
from facekit.detection.face_detector import FaceDetector
from facekit.tracking.face_tracker import FaceTracker
from facekit.tracking.tracker_validator import TrackerValidator, ValidatorParams


BBox = Tuple[float, float, float, float]  # x, y, w, h


# -----------------------
# helpers
# -----------------------
DBG_DIR = Path("validator_dbg")
DBG_DIR.mkdir(parents=True, exist_ok=True)

def _draw_box(img, box_xywh, color=(0,255,0), label=""):
    x, y, w, h = [int(round(v)) for v in box_xywh]
    out = img.copy()
    cv2.rectangle(out, (x, y), (x+w, y+h), color, 2)
    if label:
        cv2.putText(out, label, (x, max(20, y-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return out

def _save_dbg(name, img):
    p = DBG_DIR / name
    cv2.imwrite(str(p), img)
    return p

def _need_cv2():
    if cv2 is None:
        pytest.skip("OpenCV not available.")


def _video_path() -> str:
    """
    Use TEST_VIDEO if set, else tests/assets/sample.mp4.
    """
    env_p = os.environ.get("TEST_VIDEO")
    if env_p and Path(env_p).exists():
        return env_p
    candidate = Path(__file__).parents[2] / "tests" / "data" / "interview-sam-altman_5sec_snippet.mp4"
    if candidate.exists():
        return str(candidate)

    pytest.skip("No test video found. Set TEST_VIDEO or add tests/assets/sample.mp4.")


def _open_cap(path: str):
    cap = cv2.VideoCapture(path)
    assert cap.isOpened(), f"Failed to open video: {path}"
    return cap


def _read_frame(cap, idx: int) -> Optional[np.ndarray]:
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if idx < 0 or idx >= total:
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    return frame if ok else None


def _pick_pair(cap):
    """
    Choose a consecutive frame pair in the middle of the clip to reduce preroll weirdness.
    """
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total < 2:
        pytest.skip("Video has fewer than 2 frames.")
    start = max(0, total // 2 - 1)
    f0 = _read_frame(cap, start)
    f1 = _read_frame(cap, start + 1)
    if f0 is None or f1 is None:
        pytest.skip("Could not decode two consecutive frames.")
    return start, f0, f1


def _first_face_bbox(det: FaceDetector, frame: np.ndarray) -> Optional[BBox]:
    out = det.detect_faces_in_frame(frame)
    if not out:
        return None
    boxes, _, _ = out
    if not boxes:
        return None
    # detector returns xyxy; convert to xywh
    x1, y1, x2, y2 = [float(v) for v in boxes[0][:4]]
    return (x1, y1, x2 - x1, y2 - y1)


def _iou_xywh(a: BBox, b: BBox) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter + 1e-6
    return inter / union


def _shift(b: BBox, dx: float, dy: float) -> BBox:
    x, y, w, h = b
    return (x + dx, y + dy, w, h)


def _scale(b: BBox, sx: float, sy: float) -> BBox:
    x, y, w, h = b
    cx, cy = x + 0.5 * w, y + 0.5 * h
    nw, nh = max(1.0, w * sx), max(1.0, h * sy)
    return (cx - 0.5 * nw, cy - 0.5 * nh, nw, nh)


def _aspect(b: BBox, factor: float) -> BBox:
    # increase w, decrease h (or vice-versa) to change w/h
    x, y, w, h = b
    cx, cy = x + 0.5 * w, y + 0.5 * h
    nw, nh = max(1.0, w * factor), max(1.0, h / factor)
    return (cx - 0.5 * nw, cy - 0.5 * nh, nw, nh)


def _paint_hsv(frame: np.ndarray, box: BBox, color_bgr=(0, 255, 255)) -> np.ndarray:
    x, y, w, h = [int(v) for v in box]
    H, W = frame.shape[:2]
    x1 = max(0, min(x, W - 1)); y1 = max(0, min(y, H - 1))
    x2 = max(0, min(W, x + w));    y2 = max(0, min(H, y + h))
    out = frame.copy()
    if x2 > x1 and y2 > y1:
        cv2.rectangle(out, (x1, y1), (x2, y2), color_bgr, thickness=-1)
    return out

def to_xywh(box, frame_shape):
    """Return a positive (x,y,w,h) regardless of whether `box` is XYXY, XYWH, or normalized."""
    x1, y1, a, b = map(float, box)
    H, W = frame_shape[:2]

    # If looks normalized (<=1), scale up
    if 0.0 <= a <= 1.0 and 0.0 <= b <= 1.0 and 0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0:
        x1 *= W; y1 *= H; a *= W; b *= H

    # If a/b look like bottom-right coords within the image and > top-left, treat as XYXY
    if a > x1 and b > y1 and a <= W and b <= H:
        w = a - x1
        h = b - y1
        return (x1, y1, w, h)

    # Otherwise assume XYWH
    w, h = a, b
    return (x1, y1, w, h)

def _sanitize_for_init(box, shape):
    H, W = shape[:2]
    x, y, w, h = [float(v) for v in box]
    x = max(0.0, min(x, W - 1.0))
    y = max(0.0, min(y, H - 1.0))
    w = max(1.0, min(w, W - x))
    h = max(1.0, min(h, H - y))
    return (int(round(x)), int(round(y)), int(round(w)), int(round(h)))

# -----------------------
# the test
# -----------------------
# @pytest.mark.slow
def test_detect_track_and_validate_then_break_each_rule():
    video = _video_path()

    # Load YOLOv5Face detector
    det_path = "models/detector/yolov5n_state_dict.pt"
    det_cfg  = "models/detector/yolov5n.yaml"
    device = "cuda" if (hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0) else "cpu"

    try:
        model = load_yolo5face_model(detector_model_path=det_path, config_path=det_cfg, device=device)
    except Exception as e:
        pytest.skip(f"Face detector model not available: {e}")

    det = FaceDetector(model)

    cap = _open_cap(video)
    try:
        first_idx, f0, f1 = _pick_pair(cap)

        # 1) DETECT on frame t
        b0 = _first_face_bbox(det, f0)
        if b0 is None:
            pytest.skip("No face detected in the chosen frame; pick a different clip.")

        # 2) INIT TRACKER at t from detection
        box_xywh = to_xywh(b0, f0.shape)
        box_xywh = _sanitize_for_init(box_xywh, f0.shape)

        # # Save visual debug
        # _save_dbg("f0_det.png", _draw_box(f0, box_xywh, (0,255,0), "det(t)"))
        # _save_dbg("f1_expect.png", _draw_box(f1, box_xywh, (255,255,0), "expected(t->t+1)"))

        tracker = FaceTracker(tracker_type="CSRT")
        tracker.init_trackers(np.ascontiguousarray(f0), [box_xywh], [1])

        # 3) TRACK to frame t+1 (normal case)
        tracked = tracker.update_trackers(f1)
        assert tracked and tracked.get(1) is not None, "Tracker failed on the very next frame."
        x, y, w, h = tracked[1]
        b1_track = (float(x), float(y), float(w), float(h))

        # 4) VALIDATE (should PASS on normal step)
        pass_params = ValidatorParams(
            iou_thresh=0.4,       # modest
            area_delta_max=0.6,   # modest
            asp_ratio_delta_max=0.6,
            v_max=0.9,
            hsv_thresh=1.0,       # lenient appearance for natural changes
        )
        v_ok = TrackerValidator([f0, f1], first_frame_idx=first_idx, params=pass_params)
        v_ok.set_baseline({1: b0}, first_idx)
        assert v_ok.validate({1: b1_track}, first_idx + 1, verbose=True), "Normal track should pass validation."

        # 5) BREAK EACH RULE (one-by-one)

        # -- IoU too low: shift bbox a lot (keep frame same)
        bad_iou_params = ValidatorParams(iou_thresh=0.7, hsv_thresh=1.0)
        v_iou = TrackerValidator([f0, f1], first_frame_idx=first_idx, params=bad_iou_params)
        v_iou.set_baseline({1: b0}, first_idx)
        b1_iou = _shift(b0, dx=0.7 * b0[2], dy=0.0)  # shift ~70% of width
        assert _iou_xywh(b0, b1_iou) < 0.7
        assert v_iou.validate({1: b1_iou}, first_idx + 1, verbose=True) is False

        # -- Area change too large: zoom by +60% area
        bad_area_params = ValidatorParams(area_delta_max=0.2, hsv_thresh=1.0)
        v_area = TrackerValidator([f0, f1], first_frame_idx=first_idx, params=bad_area_params)
        v_area.set_baseline({1: b0}, first_idx)
        b1_area = _scale(b0, sx=1.6, sy=1.6)  # ~+156% area; |Δ|/A0 ~ 0.56
        assert v_area.validate({1: b1_area}, first_idx + 1, verbose=True) is False

        # -- Aspect ratio change too large
        bad_ar_params = ValidatorParams(asp_ratio_delta_max=0.2, hsv_thresh=1.0)
        v_ar = TrackerValidator([f0, f1], first_frame_idx=first_idx, params=bad_ar_params)
        v_ar.set_baseline({1: b0}, first_idx)
        b1_ar = _aspect(b0, factor=1.8)
        assert v_ar.validate({1: b1_ar}, first_idx + 1, verbose=True) is False

        # -- Velocity too high: big center jump vs previous diagonal
        bad_vel_params = ValidatorParams(v_max=0.3, hsv_thresh=1.0)
        v_vel = TrackerValidator([f0, f1], first_frame_idx=first_idx, params=bad_vel_params)
        v_vel.set_baseline({1: b0}, first_idx)
        diag = (b0[2] ** 2 + b0[3] ** 2) ** 0.5
        b1_vel = _shift(b0, dx=0.9 * diag, dy=0.0)  # center shift ~0.9*diag
        assert v_vel.validate({1: b1_vel}, first_idx + 1, verbose=True) is False

        # -- Appearance (HSV) change: paint inside bbox on frame t+1 but keep geometry
        bad_app_params = ValidatorParams(iou_thresh=0.4, hsv_thresh=0.05)  # strict appearance
        f1_painted = _paint_hsv(f1, b0, color_bgr=(0, 255, 255))
        v_app = TrackerValidator([f0, f1_painted], first_frame_idx=first_idx, params=bad_app_params)
        v_app.set_baseline({1: b0}, first_idx)
        assert v_app.validate({1: b1_track}, first_idx + 1, verbose=True) is False

    finally:
        cap.release()
