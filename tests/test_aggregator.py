import pytest
import numpy as np
from facekit.tracking.face_structures import FaceObservation, FaceTrack
from facekit.tracking.aggregator import ShotFaceTrackAggregator
from facekit.common.obs_consts import Source


def make_obs(frame_idx, bbox, embedding=None):
    return FaceObservation(
        frame_idx=frame_idx,
        bbox=bbox,
        embedding=np.array(embedding) if embedding is not None else None,
        source=Source.DETECTED
    )

def test_single_observation_creates_one_track():
    aggregator = ShotFaceTrackAggregator(shot_number=1)
    obs = make_obs(0, (10, 10, 50, 50))
    aggregator.update_tracks_with_frame(0, [obs])
    tracks = aggregator.finalize_tracks()
    assert len(tracks) == 1
    assert tracks[0].get_frame_indices() == [0]

def test_preassigned_track_id_extends_existing_track():
    aggregator = ShotFaceTrackAggregator(shot_number=1)
    # frame 0 creates track 0
    aggregator.update_tracks_with_frame(0, [make_obs(0, (10,10,50,50))])
    # frame 1: caller explicitly assigns to track 0
    obs = make_obs(1, (11,11,51,51))
    obs.track_id = 0
    aggregator.update_tracks_with_frame(1, [obs])
    tracks = aggregator.finalize_tracks()
    assert len(tracks) == 1
    assert tracks[0].track_id == 0
    assert tracks[0].get_frame_indices() == [0, 1]

def test_different_faces_create_different_tracks():
    aggregator = ShotFaceTrackAggregator(shot_number=1)
    obs1 = make_obs(0, (10, 10, 50, 50))
    obs2 = make_obs(1, (200, 200, 250, 250))  # Far apart
    aggregator.update_tracks_with_frame(0, [obs1])
    aggregator.update_tracks_with_frame(1, [obs2])
    tracks = aggregator.finalize_tracks()
    assert len(tracks) == 2
    all_indices = [track.get_frame_indices() for track in tracks]
    assert sorted(all_indices) == [[0], [1]]

def test_multiple_faces_same_frame():
    aggregator = ShotFaceTrackAggregator(shot_number=1)
    obs1 = make_obs(0, (10, 10, 50, 50))
    obs2 = make_obs(0, (100, 100, 140, 140))
    aggregator.update_tracks_with_frame(0, [obs1, obs2])
    tracks = aggregator.finalize_tracks()
    assert len(tracks) == 2
    assert all([track.get_frame_indices() == [0] for track in tracks])

def test_preassigned_and_unassigned_create_and_extend():
    aggregator = ShotFaceTrackAggregator(shot_number=1)
    aggregator.update_tracks_with_frame(0, [make_obs(0, (10,10,50,50))])  # creates tid=0
    obs_extend = make_obs(1, (11,11,51,51))
    obs_extend.track_id = 0  # caller assigns
    obs_new = make_obs(1, (100,100,140,140))  # caller leaves unassigned => new track
    aggregator.update_tracks_with_frame(1, [obs_extend, obs_new])
    tracks = aggregator.finalize_tracks()
    assert len(tracks) == 2
    lens = sorted(len(t.observations) for t in tracks)
    assert lens == [1, 2]

def test_get_tracks_in_frame():
    aggregator = ShotFaceTrackAggregator(shot_number=1)
    aggregator.update_tracks_with_frame(0, [make_obs(0, (0, 0, 10, 10))])
    aggregator.update_tracks_with_frame(1, [make_obs(1, (0, 0, 10, 10))])
    aggregator.update_tracks_with_frame(2, [make_obs(2, (100, 100, 110, 110))])
    aggregator.update_tracks_with_frame(3, [make_obs(3, (0, 0, 10, 10)), make_obs(3, (100, 100, 110, 110))])

    aggregator.finalize_tracks()

    result = aggregator.get_tracks_in_frame(3)
    assert len(result) == 2  # Two tracks visible in frame 3
    track_ids = [t.track_id for t in result]
    assert len(set(track_ids)) == 2  # Ensure two unique track IDs in that frame

def test_iou_matching_threshold():
    aggregator = ShotFaceTrackAggregator(shot_number=1)
    obs1 = make_obs(0, (10, 10, 50, 50))
    obs2 = make_obs(1, (40, 40, 80, 80))  # Partial overlap, but small IoU
    aggregator.update_tracks_with_frame(0, [obs1])
    aggregator.update_tracks_with_frame(1, [obs2])
    tracks = aggregator.finalize_tracks()
    assert len(tracks) == 2  # Should be too far to merge

def test_embedding_based_merge():
    aggregator = ShotFaceTrackAggregator(shot_number=1, embedding_threshold=0.6)

    emb = np.ones(512, dtype=np.float32)
    obs1 = make_obs(0, (0, 0, 10, 10), embedding=emb)
    obs2 = make_obs(1, (100, 100, 110, 110), embedding=emb)  # Far apart, IoU fails

    aggregator.update_tracks_with_frame(0, [obs1])
    aggregator.update_tracks_with_frame(1, [obs2])
    aggregator.finalize_tracks()

    # Both tracks should exist
    assert len(aggregator.tracks) == 2

    segment_id_counter = 0
    aggregator.resolve_segment_ids(segment_id_counter, embedding_threshold=0.6)

    ids = {t.segment_id for t in aggregator.tracks}
    assert len(ids) == 1, f"Expected both tracks to share the same ID, got {ids}"
