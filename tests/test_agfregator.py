import pytest
import numpy as np
from facekit.tracking.face_tracks import FaceObservation, FaceTrack
from facekit.tracking.aggregator import ShotFaceTrackAggregator


def make_obs(frame_idx, bbox, embedding=None):
    return FaceObservation(
        frame_idx=frame_idx,
        bbox=bbox,
        embedding=np.array(embedding) if embedding is not None else None,
    )


def test_single_observation_creates_one_track():
    aggregator = ShotFaceTrackAggregator(shot_number=1)
    obs = make_obs(0, (10, 10, 50, 50))
    aggregator.add_frame_observations(0, [obs])
    tracks = aggregator.finalize_tracks()
    assert len(tracks) == 1
    assert tracks[0].get_frame_indices() == [0]


def test_same_face_across_multiple_frames():
    aggregator = ShotFaceTrackAggregator(shot_number=1)
    obs1 = make_obs(0, (10, 10, 50, 50))
    obs2 = make_obs(1, (11, 11, 51, 51))
    aggregator.add_frame_observations(0, [obs1])
    aggregator.add_frame_observations(1, [obs2])
    tracks = aggregator.finalize_tracks()
    assert len(tracks) == 1
    assert tracks[0].get_frame_indices() == [0, 1]


def test_different_faces_create_different_tracks():
    aggregator = ShotFaceTrackAggregator(shot_number=1)
    obs1 = make_obs(0, (10, 10, 50, 50))
    obs2 = make_obs(1, (200, 200, 250, 250))  # Far apart
    aggregator.add_frame_observations(0, [obs1])
    aggregator.add_frame_observations(1, [obs2])
    tracks = aggregator.finalize_tracks()
    assert len(tracks) == 2
    all_indices = [track.get_frame_indices() for track in tracks]
    assert sorted(all_indices) == [[0], [1]]


def test_multiple_faces_same_frame():
    aggregator = ShotFaceTrackAggregator(shot_number=1)
    obs1 = make_obs(0, (10, 10, 50, 50))
    obs2 = make_obs(0, (100, 100, 140, 140))
    aggregator.add_frame_observations(0, [obs1, obs2])
    tracks = aggregator.finalize_tracks()
    assert len(tracks) == 2
    assert all([track.get_frame_indices() == [0] for track in tracks])


def test_track_extension_and_creation():
    aggregator = ShotFaceTrackAggregator(shot_number=1)
    obs1 = make_obs(0, (10, 10, 50, 50))
    obs2 = make_obs(1, (11, 11, 51, 51))  # Should match obs1
    obs3 = make_obs(1, (100, 100, 140, 140))  # New face

    aggregator.add_frame_observations(0, [obs1])
    aggregator.add_frame_observations(1, [obs2, obs3])

    tracks = aggregator.finalize_tracks()
    assert len(tracks) == 2
    lengths = sorted(len(track.observations) for track in tracks)
    assert lengths == [1, 2]


def test_get_tracks_in_frame():
    aggregator = ShotFaceTrackAggregator(shot_number=1)
    aggregator.add_frame_observations(0, [make_obs(0, (0, 0, 10, 10))])
    aggregator.add_frame_observations(1, [make_obs(1, (0, 0, 10, 10))])
    aggregator.add_frame_observations(2, [make_obs(2, (100, 100, 110, 110))])
    aggregator.add_frame_observations(3, [make_obs(3, (0, 0, 10, 10)), make_obs(3, (100, 100, 110, 110))])

    tracks = aggregator.finalize_tracks()

    result = aggregator.get_tracks_in_frame(3)
    assert len(result) == 2
    track_ids = [t.track_id for t in result]
    assert sorted(track_ids) == [0, 1]


def test_iou_matching_threshold():
    aggregator = ShotFaceTrackAggregator(shot_number=1)
    obs1 = make_obs(0, (10, 10, 50, 50))
    obs2 = make_obs(1, (40, 40, 80, 80))  # Partial overlap, but small IoU
    aggregator.add_frame_observations(0, [obs1])
    aggregator.add_frame_observations(1, [obs2])
    tracks = aggregator.finalize_tracks()
    assert len(tracks) == 2  # Should be too far to merge


def test_patch_can_merge_with(monkeypatch):
    calls = []

    def fake_can_merge_with(self, other, iou_thresh=0.5, embedding_thresh=0.6):
        frame_idx = other.observations[0].frame_idx if other.observations else None
        calls.append((self.track_id, frame_idx, iou_thresh, embedding_thresh))
        return True

    monkeypatch.setattr(FaceTrack, "can_merge_with", fake_can_merge_with)

    aggregator = ShotFaceTrackAggregator(shot_number=1)
    obs1 = make_obs(0, (0, 0, 10, 10))
    obs2 = make_obs(1, (1, 1, 11, 11))  # Should match due to patched method
    aggregator.add_frame_observations(0, [obs1])
    aggregator.add_frame_observations(1, [obs2])

    tracks = aggregator.finalize_tracks()
    assert len(tracks) == 1
    assert len(calls) == 1
    track_id, frame_idx, iou_thresh, embedding_thresh = calls[0]
    assert track_id == 0
    assert frame_idx == 1
    assert iou_thresh == 0.5
    assert embedding_thresh == 0.6

