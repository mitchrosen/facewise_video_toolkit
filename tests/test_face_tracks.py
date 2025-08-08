import pytest
import numpy as np
from facekit.tracking.face_structures import FaceTrack, FaceObservation


def make_obs(frame_idx, bbox, embedding=None):
    return FaceObservation(
        frame_idx=frame_idx,
        bbox=bbox,
        embedding=np.array(embedding) if embedding is not None else None,
    )


def test_duration_calculation():
    track = FaceTrack(shot_id=0, track_id=3)
    assert track.duration() == 0
    track.add_observation(make_obs(5, (0, 0, 5, 5)))
    track.add_observation(make_obs(9, (5, 5, 10, 10)))
    assert track.duration() == 5


def test_frame_indices():
    track = FaceTrack(shot_id=0, track_id=1)
    track.add_observation(make_obs(3, (10, 10, 50, 50)))
    track.add_observation(make_obs(4, (12, 11, 51, 52)))
    track.add_observation(make_obs(5, (13, 12, 52, 54)))

    assert track.duration() == 3
    assert track.get_frame_indices() == [3, 4, 5]


def test_add_and_retrieve_bboxes():
    track = FaceTrack(shot_id=0, track_id=1)
    assert track.last_frame() == -1
    assert track.get_first_bbox() is None
    assert track.get_last_bbox() is None

    track.add_observation(make_obs(2, (5, 5, 15, 15)))
    track.add_observation(make_obs(3, (10, 10, 20, 20)))

    assert track.last_frame() == 3
    assert track.get_first_bbox() == (5, 5, 15, 15)
    assert track.get_last_bbox() == (10, 10, 20, 20)


def test_invalid_bbox_raises_error():
    with pytest.raises(ValueError):
        FaceObservation(frame_idx=0, bbox=(1, 2, 3))  # Too short

    with pytest.raises(ValueError):
        FaceObservation(frame_idx=0, bbox=("a", 2, 3, 4))  # Wrong type


def test_post_init_duplicate_frame_idx_raises():
    obs1 = make_obs(0, (0, 0, 10, 10))
    obs2 = make_obs(0, (1, 1, 11, 11))  # Duplicate frame_idx

    with pytest.raises(ValueError):
        FaceTrack(shot_id=0, track_id=99, observations=[obs1, obs2])


def test_add_observation_duplicate_frame_idx_raises():
    track = FaceTrack(shot_id=0, track_id=1)
    track.add_observation(make_obs(10, (1, 1, 5, 5)))

    with pytest.raises(ValueError):
        track.add_observation(make_obs(10, (2, 2, 6, 6)))


def test_add_observation_force_overwrites():
    track = FaceTrack(shot_id=0, track_id=1)
    track.add_observation(make_obs(10, (1, 1, 5, 5)))
    track.add_observation(make_obs(10, (2, 2, 6, 6)), force=True)

    # Confirm that bbox was updated
    assert track.get_bbox_by_frame(10) == (2, 2, 6, 6)


def test_get_bbox_by_index():
    track = FaceTrack(shot_id=0, track_id=2)
    track.add_observation(make_obs(0, (1, 1, 10, 10)))
    track.add_observation(make_obs(1, (2, 2, 11, 11)))

    assert track.get_bbox_by_observation_index(0) == (1, 1, 10, 10)
    assert track.get_bbox_by_observation_index(1) == (2, 2, 11, 11)
    assert track.get_bbox_by_observation_index(999) is None  # out-of-bounds check


def test_get_bbox_by_frame():
    track = FaceTrack(shot_id=0, track_id=2)
    track.add_observation(make_obs(10, (1, 1, 10, 10)))
    track.add_observation(make_obs(401, (2, 2, 11, 11)))

    assert track.get_bbox_by_frame(10) == (1, 1, 10, 10)
    assert track.get_bbox_by_frame(401) == (2, 2, 11, 11)
    assert track.get_bbox_by_frame(999) is None  # non-existent frame check


def test_get_average_bbox():
    track = FaceTrack(shot_id=0, track_id=4)
    assert track.get_average_bbox() is None

    track.add_observation(make_obs(0, (0, 0, 10, 10)))
    track.add_observation(make_obs(1, (10, 10, 20, 20)))

    avg_bbox = track.get_average_bbox()
    assert avg_bbox == (5.0, 5.0, 15.0, 15.0)


def test_compute_average_embedding():
    e1 = np.array([1.0, 0.0])
    e2 = np.array([0.0, 1.0])
    track = FaceTrack(shot_id=0, track_id=5)
    track.add_observation(make_obs(0, (0, 0, 5, 5), embedding=e1))
    track.add_observation(make_obs(1, (1, 1, 6, 6), embedding=e2))

    avg = track.compute_average_embedding()
    expected = np.array([0.5, 0.5])
    assert np.allclose(avg, expected)


def test_shares_frames_with():
    t1 = FaceTrack(shot_id=0, track_id=1)
    t2 = FaceTrack(shot_id=0, track_id=2)

    t1.add_observation(make_obs(5, (10, 10, 20, 20)))
    t1.add_observation(make_obs(6, (12, 12, 22, 22)))

    t2.add_observation(make_obs(6, (30, 30, 40, 40)))
    t2.add_observation(make_obs(7, (32, 32, 42, 42)))

    assert t1.shares_frames_with(t2) is True


def test_shares_frames_with_false_case():
    track1 = FaceTrack(shot_id=0, track_id=6)
    track2 = FaceTrack(shot_id=0, track_id=7)

    track1.add_observation(make_obs(1, (0, 0, 10, 10)))
    track2.add_observation(make_obs(2, (0, 0, 10, 10)))
    assert not track1.shares_frames_with(track2)

    track2.add_observation(make_obs(1, (5, 5, 15, 15)))
    assert track1.shares_frames_with(track2)
