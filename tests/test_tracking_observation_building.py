import numpy as np

from facekit.common.obs_consts import Source
from facekit.pipeline.track_across_segments import append_tracking_observation
from facekit.tracking.face_structures import FaceObservation


class DummyTrack:
    def __init__(self, track_id, observations):
        self.track_id = track_id
        self.observations = observations


class DummyAggregator:
    def __init__(self, tracks):
        self.tracks = tracks


def make_previous_observation(*, frame_idx=41, track_id=7, bbox, landmarks):
    return FaceObservation(
        frame_idx=frame_idx,
        track_id=track_id,
        bbox=bbox,
        embedding=None,
        confidence=None,
        aligned_face=None,
        landmarks=landmarks,
        source=Source.TRACKED,
    )


def test_append_tracking_observation_converts_xywh_to_xyxy_bbox():
    observations = []
    aggregator = DummyAggregator(tracks=[])

    result = append_tracking_observation(
        observations,
        frame_idx=42,
        track_id=7,
        tracked_box=(100.0, 50.0, 80.0, 120.0),
        aggregator=aggregator,
    )

    assert result is None
    assert len(observations) == 1

    obs = observations[0]
    assert obs.frame_idx == 42
    assert obs.track_id == 7
    assert obs.bbox == (100, 50, 180, 170)
    assert obs.source == Source.TRACKED
    assert obs.embedding is None
    assert obs.confidence is None
    assert obs.aligned_face is None
    assert obs.landmarks is None


def test_append_tracking_observation_propagates_from_latest_prior_observation_on_matching_track():
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

    prior_obs = make_previous_observation(
        bbox=(100, 50, 200, 150),
        landmarks=prev_landmarks,
    )

    aggregator = DummyAggregator(
        tracks=[DummyTrack(track_id=7, observations=[prior_obs])]
    )
    observations = []

    append_tracking_observation(
        observations,
        frame_idx=42,
        track_id=7,
        tracked_box=(112.0, 68.0, 100.0, 100.0),
        aggregator=aggregator,
    )

    assert len(observations) == 1
    obs = observations[0]

    assert obs.frame_idx == 42
    assert obs.track_id == 7
    assert obs.bbox == (112, 68, 212, 168)
    assert obs.source == Source.TRACKED
    assert obs.landmarks is not None
    assert obs.landmarks.shape == (5, 2)


def test_append_tracking_observation_leaves_landmarks_none_without_prior_landmarks():
    prior_obs = make_previous_observation(
        bbox=(100, 50, 200, 150),
        landmarks=None,
    )

    aggregator = DummyAggregator(
        tracks=[DummyTrack(track_id=7, observations=[prior_obs])]
    )
    observations = []

    append_tracking_observation(
        observations,
        frame_idx=42,
        track_id=7,
        tracked_box=(112.0, 68.0, 100.0, 100.0),
        aggregator=aggregator,
    )

    assert len(observations) == 1
    assert observations[0].landmarks is None


def test_append_tracking_observation_uses_latest_matching_track_observation():
    old_landmarks = np.array(
        [
            [10.0, 10.0],
            [20.0, 10.0],
            [15.0, 15.0],
            [12.0, 20.0],
            [18.0, 20.0],
        ],
        dtype=np.float32,
    )
    new_landmarks = np.array(
        [
            [120.0, 80.0],
            [180.0, 80.0],
            [150.0, 105.0],
            [130.0, 130.0],
            [170.0, 130.0],
        ],
        dtype=np.float32,
    )

    old_obs = make_previous_observation(
        frame_idx=10,
        track_id=7,
        bbox=(0, 0, 100, 100),
        landmarks=old_landmarks,
    )
    new_obs = make_previous_observation(
        frame_idx=41,
        track_id=7,
        bbox=(100, 50, 200, 150),
        landmarks=new_landmarks,
    )

    aggregator = DummyAggregator(
        tracks=[DummyTrack(track_id=7, observations=[old_obs, new_obs])]
    )
    observations = []

    append_tracking_observation(
        observations,
        frame_idx=42,
        track_id=7,
        tracked_box=(112.0, 68.0, 100.0, 100.0),
        aggregator=aggregator,
    )

    assert len(observations) == 1
    obs = observations[0]

    assert obs.track_id == 7
    assert obs.bbox == (112, 68, 212, 168)
    assert obs.landmarks is not None
    assert obs.landmarks.shape == (5, 2)


def test_append_tracking_observation_ignores_history_from_other_tracks():
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

    other_track_obs = make_previous_observation(
        track_id=99,
        bbox=(100, 50, 200, 150),
        landmarks=prev_landmarks,
    )

    aggregator = DummyAggregator(
        tracks=[DummyTrack(track_id=99, observations=[other_track_obs])]
    )
    observations = []

    append_tracking_observation(
        observations,
        frame_idx=42,
        track_id=7,
        tracked_box=(112.0, 68.0, 100.0, 100.0),
        aggregator=aggregator,
    )

    assert len(observations) == 1
    assert observations[0].track_id == 7
    assert observations[0].landmarks is None