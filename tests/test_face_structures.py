import pytest
import numpy as np
from facekit.tracking.face_structures import FaceObservation

def dummy_landmarks():
    return [(1, 2)] * 68

def dummy_aligned_face():
    return np.zeros((112, 112, 3), dtype=np.uint8)

def test_face_observation_accepts_tuple_bbox():
    obs = FaceObservation(
        0,
        (10, 20, 30, 40),
        embedding=None,
        confidence=0.99
    )
    assert obs.bbox == (10, 20, 30, 40)

def test_face_observation_accepts_list_bbox():
    obs = FaceObservation(frame_idx=0, bbox=[10, 20, 30, 40])
    assert isinstance(obs.bbox, tuple)
    assert obs.bbox == (10, 20, 30, 40)

def test_face_observation_rejects_invalid_bbox():
    # Too few elements
    with pytest.raises(ValueError, match=r"Invalid bbox: could not convert to 4-tuple of ints — received"):
        FaceObservation(frame_idx=0, bbox=(10, 20, 30))

    # Not all ints
    with pytest.raises(ValueError, match=r"Invalid bbox: could not convert to 4-tuple of ints — received"):
        FaceObservation(frame_idx=0, bbox=(10, "20", 30, 40))

    # Wrong type entirely
    with pytest.raises(ValueError, match=r"Invalid bbox: could not convert to 4-tuple of ints — received"):
        FaceObservation(frame_idx=0, bbox="not a tuple")


