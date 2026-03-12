import pytest

from facekit.embedding.embedding_sampling import (
    should_sample_track_observation,
)


def test_first_frame_of_track_is_always_sampled():
    assert should_sample_track_observation(
        track_local_index=0,
        is_detection_frame=True,
        track_sample_interval=5,
    )


def test_detection_frame_is_always_sampled_regardless_of_track_sample_interval():
    assert should_sample_track_observation(
        track_local_index=7,
        is_detection_frame=True,
        track_sample_interval=100,
    )


def test_track_sample_interval_uses_track_local_frame_index():
    interval = 5

    assert should_sample_track_observation(
        track_local_index=5,
        is_detection_frame=False,
        track_sample_interval=interval,
    )

    assert should_sample_track_observation(
        track_local_index=10,
        is_detection_frame=False,
        track_sample_interval=interval,
    )


def test_non_detection_non_interval_frame_is_not_sampled():
    assert not should_sample_track_observation(
        track_local_index=3,
        is_detection_frame=False,
        track_sample_interval=5,
    )


def test_detection_frame_sampling_is_not_blocked_by_interval_logic():
    assert should_sample_track_observation(
        track_local_index=3,
        is_detection_frame=True,
        track_sample_interval=5,
    )