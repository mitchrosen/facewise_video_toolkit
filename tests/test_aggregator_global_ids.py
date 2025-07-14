import numpy as np
from facekit.tracking.aggregator import ShotFaceTrackAggregator
from facekit.tracking.face_structures import FaceTrack, FaceObservation


def dummy_observation(frame_idx, bbox, embedding=None):
    return FaceObservation(frame_idx=frame_idx, bbox=bbox, embedding=embedding)


def dummy_track(track_id, shot_id, embedding=None):
    obs = dummy_observation(frame_idx=0, bbox=(0, 0, 10, 10), embedding=embedding)
    return FaceTrack(track_id=track_id, shot_id=shot_id, observations=[obs])


def test_assigns_unique_ids_when_no_embeddings():
    aggregator = ShotFaceTrackAggregator(shot_number=0)
    obs1 = dummy_observation(0, (10, 10, 50, 50))
    obs2 = dummy_observation(0, (100, 100, 150, 150))
    
    aggregator.add_frame_observations(0, [obs1, obs2])
    aggregator.finalize_tracks()
    
    vchunk_id_counter = 0
    updated_counter = aggregator.resolve_vchunk_ids([], vchunk_id_counter)

    vchunk_ids = [track.vchunk_id for track in aggregator.tracks]
    
    assert len(set(vchunk_ids)) == 2
    assert sorted(vchunk_ids) == [0, 1]
    assert updated_counter == 2


def test_reuses_vchunk_id_on_same_embedding_within_chunk():
    shared_embedding = np.ones(512, dtype=np.float32)

    aggregator = ShotFaceTrackAggregator(shot_number=0)
    obs1 = dummy_observation(0, (10, 10, 50, 50), embedding=shared_embedding)
    obs2 = dummy_observation(1, (110, 110, 150, 150), embedding=shared_embedding)  # Far apart, will not merge


    aggregator.add_frame_observations(0, [obs1])
    aggregator.add_frame_observations(1, [obs2])
    aggregator.finalize_tracks()

    # Manually assign vchunk_id to first track to simulate one resolved earlier
    aggregator.tracks[0].vchunk_id = 42
    aggregator.tracks[0].is_active = False 

    vchunk_id_counter = 43
    updated_counter = aggregator.resolve_vchunk_ids([], vchunk_id_counter)

    assert aggregator.tracks[1].vchunk_id == 42
    assert updated_counter == 43  # No new ID consumed

def test_does_not_merge_dissimilar_embeddings():
    emb1 = np.ones(512, dtype=np.float32)
    emb2 = np.zeros(512, dtype=np.float32)
    emb1 /= np.linalg.norm(emb1)
    emb2[0] = 1.0  # Very different

    aggregator = ShotFaceTrackAggregator(shot_number=0)
    obs1 = dummy_observation(0, (10, 10, 50, 50), embedding=emb1)
    obs2 = dummy_observation(1, (100, 100, 150, 150), embedding=emb2)

    aggregator.add_frame_observations(0, [obs1])
    aggregator.add_frame_observations(1, [obs2])
    aggregator.finalize_tracks()

    aggregator.tracks[0].vchunk_id = 42
    aggregator.tracks[0].is_active = False

    updated_counter = aggregator.resolve_vchunk_ids([], vchunk_id_counter=43)

    assert aggregator.tracks[1].vchunk_id == 43
    assert updated_counter == 44

def test_reuses_id_from_prior_tracks():
    emb = np.ones(512, dtype=np.float32)
    emb /= np.linalg.norm(emb)

    # Create a prior track from a previous chunk
    prior_track = dummy_track(track_id=0, shot_id=99, embedding=emb)
    prior_track.vchunk_id = 101  # Simulated vchunk_id assignment

    # New shot
    aggregator = ShotFaceTrackAggregator(shot_number=0)
    obs = dummy_observation(0, (10, 10, 50, 50), embedding=emb)
    aggregator.add_frame_observations(0, [obs])
    aggregator.finalize_tracks()

    updated_counter = aggregator.resolve_vchunk_ids([prior_track], vchunk_id_counter=102)

    assert aggregator.tracks[0].vchunk_id == 101
    assert updated_counter == 102  # No new ID consumed

def test_skips_tracks_without_embedding():
    obs = dummy_observation(0, (10, 10, 50, 50), embedding=None)
    aggregator = ShotFaceTrackAggregator(shot_number=0)
    aggregator.add_frame_observations(0, [obs])
    aggregator.finalize_tracks()

    updated_counter = aggregator.resolve_vchunk_ids([], vchunk_id_counter=5)

    assert aggregator.tracks[0].vchunk_id == 5
    assert updated_counter == 6


def test_preserves_preassigned_vchunk_id():
    emb = np.ones(512, dtype=np.float32)
    emb /= np.linalg.norm(emb)

    aggregator = ShotFaceTrackAggregator(shot_number=0)
    obs = dummy_observation(0, (0, 0, 10, 10), embedding=emb)
    aggregator.add_frame_observations(0, [obs])
    aggregator.finalize_tracks()

    aggregator.tracks[0].vchunk_id = 77  # Preassigned
    updated_counter = aggregator.resolve_vchunk_ids([], vchunk_id_counter=78)

    assert aggregator.tracks[0].vchunk_id == 77
    assert updated_counter == 78

