import numpy as np
import pytest
import cv2

import sys
sys.modules.pop("mouthtracker.pipeline.mouthtrack_frame_by_frame", None)
sys.modules.pop("mouthtracker.pipeline", None)

from mouthtracker.tracking.mouth_tracker import MouthTracker, get_mouth_box_from_landmarks

# --- Fixtures ---

@pytest.fixture
def dummy_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)  # black frame

@pytest.fixture
def dummy_landmarks():
    """
    Generate one face's 5-point landmarks.
    Index 2 = center mouth, 3 = left, 4 = right
    """
    return [[
        (30, 30),  # point 0
        (60, 30),  # point 1
        (45, 50),  # center mouth
        (35, 55),  # left mouth
        (55, 55),  # right mouth
    ]]

# --- Tests ---

def test_get_mouth_box_from_landmarks(dummy_landmarks):
    box = get_mouth_box_from_landmarks(dummy_landmarks[0])
    assert box == (30, 45, 30, 15)

def test_get_mouth_box_from_landmarks_wide_mouth():
    """
    Uses a wider mouth layout to verify bounding box expansion logic.
    """
    # Wider spread in x, higher placement in y
    landmarks = [
        (0, 0),               # filler
        (0, 0),               # filler
        (80, 40),             # center mouth
        (60, 42),             # left mouth
        (100, 42),            # right mouth
    ]

    box = get_mouth_box_from_landmarks(landmarks)
    # x_min = 60 → 55, x_max = 100 → 105, w = 50
    # y_min = 40 → 35, y_max = 42 → 47, h = 12
    assert box == (55, 35, 50, 12)

def test_tracker_initialization_creates_expected_number(dummy_frame, dummy_landmarks):
    tracker = MouthTracker()
    tracker.init_trackers(dummy_frame, dummy_landmarks)

    assert len(tracker.trackers) == 1
    assert len(tracker.tracker_boxes) == 1
    assert len(tracker.track_fail_counts) == 1

def test_tracker_update_returns_boxes(dummy_frame, dummy_landmarks):
    tracker = MouthTracker()
    tracker.init_trackers(dummy_frame, dummy_landmarks)

    result = tracker.update_trackers(dummy_frame)
    assert isinstance(result, list)
    assert len(result) == 1
    box = result[0]
    assert box is None or (len(box) == 4 and all(isinstance(v, (int, float)) for v in box))

def test_multiple_face_tracking(dummy_frame):
    landmarks_set = []
    for offset in range(0, 100, 50):  # two fake faces with 5 landmarks each
        landmarks = [
            (30 + offset, 30 + offset),
            (60 + offset, 30 + offset),
            (45 + offset, 50 + offset),
            (35 + offset, 55 + offset),
            (55 + offset, 55 + offset),
        ]
        landmarks_set.append(landmarks)

    tracker = MouthTracker()
    tracker.init_trackers(dummy_frame, landmarks_set)
    assert len(tracker.trackers) == 2

    boxes = tracker.update_trackers(dummy_frame)
    assert len(boxes) == 2
