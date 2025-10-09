from __future__ import annotations
import numpy as np
import cv2
import pytest

from facekit.tracking.tracker_validator import TrackerValidator, ValidatorParams

def _solid_frame(h: int = 240, w: int = 320, bgr=(40, 40, 40)) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = bgr
    return img

@pytest.fixture
def frames_gray():
    """
    Simple deterministic frames list for validator (no I/O).
    60 frames, same content by default.
    """
    return [_solid_frame() for _ in range(60)]

@pytest.fixture
def frames_with_patch():
    """
    Frames where we draw a colored patch (so HSV signatures are non-degenerate).
    """
    frames = []
    for i in range(60):
        img = _solid_frame()
        # small patch varies slightly over time (but stays very similar)
        x, y, s = 100, 80, 40
        color = (40 + (i % 5), 180, 220)  # BGR
        cv2.rectangle(img, (x, y), (x + s, y + s), color, thickness=-1)
        frames.append(img)
    return frames

BBox = tuple[float, float, float, float]

def _box(x, y, w, h) -> BBox:
    return (float(x), float(y), float(w), float(h))


def test_accepts_consecutive_small_changes(frames_with_patch):
    first = 10
    p = ValidatorParams(
        iou_thresh=0.4,           # allow moderate overlap drop
        area_delta_max=0.5,
        asp_ratio_delta_max=0.5,
        v_max=0.8,
        hsv_thresh=0.5
    )
    v = TrackerValidator(frames_with_patch, first_frame_idx=first, params=p)

    # Seed at frame 10
    f10 = first
    boxes10 = {1: _box(100, 80, 40, 40)}
    v.set_baseline(boxes10, f10)

    # Small, realistic movement/scale change at frame 11
    f11 = first + 1
    boxes11 = {1: _box(102, 82, 41, 39)}
    assert v.validate(boxes11, f11, verbose=True) is True

    # Another tiny change at frame 12
    f12 = first + 2
    boxes12 = {1: _box(104, 84, 41, 40)}
    assert v.validate(boxes12, f12) is True


def test_rejects_large_motion(frames_gray):
    first = 0
    v = TrackerValidator(frames_gray, first_frame_idx=first, params=ValidatorParams())
    # seed at frame 0
    boxes0 = {5: _box(50, 50, 40, 40)}
    v.set_baseline(boxes0, 0)

    # jump far away next frame -> should fail on velocity/IoU
    boxes1 = {5: _box(200, 160, 40, 40)}
    assert v.validate(boxes1, 1, verbose=True) is False


def test_rejects_low_iou(frames_gray):
    first = 5
    p = ValidatorParams(iou_thresh=0.7)  # make IoU stricter
    v = TrackerValidator(frames_gray, first_frame_idx=first, params=p)
    v.set_baseline({3: _box(100, 80, 40, 40)}, first)

    # Shift enough to drop IoU below 0.7 (but not too far to trigger velocity alone)
    assert v.validate({3: _box(130, 80, 40, 40)}, first + 1, verbose=True) is False


def test_reseeds_on_non_consecutive(frames_with_patch):
    first = 20
    v = TrackerValidator(frames_with_patch, first_frame_idx=first, params=ValidatorParams())

    # First call with no baseline -> seeds and returns True
    assert v.validate({1: _box(100, 80, 40, 40)}, first) is True

    # Skip a frame (non-consecutive); validator should reseed and accept
    assert v.validate({1: _box(101, 81, 40, 40)}, first + 2) is True


def test_hsv_appearance_threshold(frames_with_patch):
    first = 0
    # Tighten hsv_thresh so appearance difference triggers failure
    p = ValidatorParams(hsv_thresh=0.05, iou_thresh=0.4)  # lenient geom, strict appearance
    v = TrackerValidator(frames_with_patch, first_frame_idx=first, params=p)

    # Seed at 0
    v.set_baseline({9: _box(100, 80, 40, 40)}, 0)

    # Make a "different appearance" by shifting the box outside the colored patch
    # BUT keep geometry okay to ensure failure is from appearance, not IoU/velocity.
    # At frame 1, the patch is still near (100,80). Move to a region without patch.
    new_box = _box(10, 10, 40, 40)
    ok = v.validate({9: new_box}, 1, verbose=True)

    # Expect failure due to appearance mismatch (HSV distance > hsv_thresh)
    assert ok is False