from __future__ import annotations
import numpy as np
import cv2
import pytest

from facekit.tracking.tracker_validator import TrackerValidator, ValidatorParams

def _solid_frame(h: int = 240, w: int = 320, bgr=(40, 40, 40)) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = bgr
    return img

def _paint_box(frame: np.ndarray, box_xywh, bgr) -> np.ndarray:
    x, y, w, h = [int(round(v)) for v in box_xywh]
    out = frame.copy()
    cv2.rectangle(out, (x, y), (x + w, y + h), bgr, thickness=-1)
    return out

@pytest.fixture
def frames_gray():
    # 60 deterministic frames, identical content
    return [_solid_frame() for _ in range(60)]

@pytest.fixture
def frames_with_patch():
    # 60 deterministic frames with a patch whose color varies slightly
    frames = []
    for i in range(60):
        img = _solid_frame()
        x, y, s = 100, 80, 40
        color = (40 + (i % 5), 180, 220)  # BGR
        cv2.rectangle(img, (x, y), (x + s, y + s), color, thickness=-1)
        frames.append(img)
    return frames

BBox = tuple[float, float, float, float]

def _box(x, y, w, h) -> BBox:
    return (float(x), float(y), float(w), float(h))

def test_accepts_consecutive_small_changes(frames_with_patch):
    p = ValidatorParams(
        iou_thresh=0.4,
        area_delta_max=0.5,
        asp_ratio_delta_max=0.5,
        v_max=0.8,
        hsv_thresh=0.5,
    )
    v = TrackerValidator(params=p)

    f10 = 10
    boxes10 = {1: _box(100, 80, 40, 40)}
    assert v.validate(boxes10, current_frame=frames_with_patch[f10], frame_idx=f10) is True

    f11 = 11
    boxes11 = {1: _box(102, 82, 41, 39)}
    assert v.validate(boxes11, current_frame=frames_with_patch[f11], frame_idx=f11) is True

    f12 = 12
    boxes12 = {1: _box(104, 84, 41, 40)}
    assert v.validate(boxes12, current_frame=frames_with_patch[f12], frame_idx=f12) is True

def test_rejects_large_motion(frames_gray):
    v = TrackerValidator(params=ValidatorParams())

    # seed at frame 0
    assert v.validate({5: _box(50, 50, 40, 40)}, current_frame=frames_gray[0], frame_idx=0) is True

    # jump far away next frame -> should fail on velocity and/or IoU
    assert v.validate({5: _box(200, 160, 40, 40)}, current_frame=frames_gray[1], frame_idx=1) is False

def test_rejects_low_iou(frames_gray):
    p = ValidatorParams(iou_thresh=0.7, hsv_thresh=2.0)  # disable HSV to isolate IoU
    v = TrackerValidator(params=p)

    f = 5
    assert v.validate({3: _box(100, 80, 40, 40)}, current_frame=frames_gray[f], frame_idx=f) is True

    # Shift enough to drop IoU below 0.7
    assert v.validate({3: _box(130, 80, 40, 40)}, current_frame=frames_gray[f + 1], frame_idx=f + 1) is False

def test_reseeds_on_non_consecutive(frames_with_patch):
    v = TrackerValidator(params=ValidatorParams())

    first = 20
    assert v.validate({1: _box(100, 80, 40, 40)}, current_frame=frames_with_patch[first], frame_idx=first) is True

    # skip a frame => non-consecutive => reseed and accept
    assert v.validate({1: _box(101, 81, 40, 40)}, current_frame=frames_with_patch[first + 2], frame_idx=first + 2) is True

def test_hsv_appearance_threshold(frames_gray):
    """
    Keep geometry identical, but change the pixels inside the box a lot between frames
    so failure is driven by HSV (not IoU/velocity).
    """
    box = _box(100, 80, 40, 40)

    # frame 0: box painted one color; frame 1: box painted very different color
    f0 = _paint_box(frames_gray[0], box, bgr=(0, 0, 255))     # red
    f1 = _paint_box(frames_gray[1], box, bgr=(0, 255, 255))   # yellow/cyan-ish

    p = ValidatorParams(
        iou_thresh=0.4,
        area_delta_max=0.8,
        asp_ratio_delta_max=0.8,
        v_max=2.0,         # generous, so geom can’t fail
        hsv_thresh=0.05,   # strict appearance
    )
    v = TrackerValidator(params=p)

    assert v.validate({9: box}, current_frame=f0, frame_idx=0) is True
    assert v.validate({9: box}, current_frame=f1, frame_idx=1) is False
