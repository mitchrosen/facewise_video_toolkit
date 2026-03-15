import numpy as np

from facekit.tracking.landmark_propagation import (
    propagate_landmarks_by_bbox_transform,
)


def test_landmarks_translate_when_bbox_translates_without_scale_change():
    prev_bbox = (100, 50, 200, 150)
    curr_bbox = (112, 68, 212, 168)  # +12 x, +18 y

    prev_landmarks = np.array(
        [
            [120.0, 80.0],
            [180.0, 80.0],
            [150.0, 105.0],
            [130.0, 130.0],
            [170.0, 130.0],
        ],
        dtype=np.float32,
    )

    propagated = propagate_landmarks_by_bbox_transform(
        prev_landmarks=prev_landmarks,
        prev_bbox=prev_bbox,
        curr_bbox=curr_bbox,
    )

    expected = prev_landmarks + np.array([12.0, 18.0], dtype=np.float32)

    assert propagated is not None
    assert propagated.dtype == np.float32
    assert propagated.shape == (5, 2)
    assert np.allclose(propagated, expected)


def test_landmarks_scale_with_bbox_resize():
    prev_bbox = (100, 50, 200, 150)   # width=100, height=100
    curr_bbox = (100, 50, 250, 200)   # width=150, height=150

    prev_landmarks = np.array(
        [
            [120.0, 80.0],
            [180.0, 80.0],
            [150.0, 105.0],
            [130.0, 130.0],
            [170.0, 130.0],
        ],
        dtype=np.float32,
    )

    propagated = propagate_landmarks_by_bbox_transform(
        prev_landmarks=prev_landmarks,
        prev_bbox=prev_bbox,
        curr_bbox=curr_bbox,
    )

    expected = np.array(
        [
            [130.0, 95.0],
            [220.0, 95.0],
            [175.0, 132.5],
            [145.0, 170.0],
            [205.0, 170.0],
        ],
        dtype=np.float32,
    )

    assert propagated is not None
    assert np.allclose(propagated, expected)


def test_missing_previous_landmarks_returns_none():
    propagated = propagate_landmarks_by_bbox_transform(
        prev_landmarks=None,
        prev_bbox=(100, 50, 200, 150),
        curr_bbox=(110, 60, 210, 160),
    )

    assert propagated is None


def test_invalid_previous_bbox_returns_none():
    prev_landmarks = np.array(
        [
            [120.0, 80.0],
            [180.0, 80.0],
            [150.0, 105.0],
            [130.0, 130.0],
            [170.0, 130.0],
        ],
        dtype=np.float32,
    )

    propagated = propagate_landmarks_by_bbox_transform(
        prev_landmarks=prev_landmarks,
        prev_bbox=(100, 50, 100, 150),  # zero width
        curr_bbox=(110, 60, 210, 160),
    )

    assert propagated is None


def test_invalid_current_bbox_returns_none():
    prev_landmarks = np.array(
        [
            [120.0, 80.0],
            [180.0, 80.0],
            [150.0, 105.0],
            [130.0, 130.0],
            [170.0, 130.0],
        ],
        dtype=np.float32,
    )

    propagated = propagate_landmarks_by_bbox_transform(
        prev_landmarks=prev_landmarks,
        prev_bbox=(100, 50, 200, 150),
        curr_bbox=(110, 60, 110, 160),  # zero width
    )

    assert propagated is None


def test_returns_none_for_wrong_landmark_shape():
    bad_landmarks = np.array(
        [
            [120.0, 80.0],
            [180.0, 80.0],
            [150.0, 105.0],
        ],
        dtype=np.float32,
    )

    propagated = propagate_landmarks_by_bbox_transform(
        prev_landmarks=bad_landmarks,
        prev_bbox=(100, 50, 200, 150),
        curr_bbox=(110, 60, 210, 160),
    )

    assert propagated is None