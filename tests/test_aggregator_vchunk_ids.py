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
    
    aggregator.update_tracks_with_frame(0, [obs1, obs2])
    aggregator.finalize_tracks()
    
    vchunk_id_counter = 0
    updated_counter = aggregator.resolve_vchunk_ids(vchunk_id_counter)

    vchunk_ids = [track.vchunk_id for track in aggregator.tracks]
    
    assert len(set(vchunk_ids)) == 2
    assert sorted(vchunk_ids) == [0, 1]
    assert updated_counter == 2


def test_reuses_vchunk_id_on_same_embedding_within_chunk():
    shared_embedding = np.ones(512, dtype=np.float32)

    # Set a very high threshold for frame-level merging to prevent merging
    aggregator = ShotFaceTrackAggregator(shot_number=0, embedding_threshold=1.1)
    obs1 = dummy_observation(0, (10, 10, 50, 50), embedding=shared_embedding)
    obs2 = dummy_observation(10, (1000, 1000, 1050, 1050), embedding=shared_embedding)

    aggregator.update_tracks_with_frame(0, [obs1])
    aggregator.update_tracks_with_frame(10, [obs2])
    aggregator.finalize_tracks()

    # Ensure two distinct tracks were created
    assert len(aggregator.tracks) == 2

    # Assign vchunk_id to the first track
    aggregator.tracks[0].vchunk_id = 42

    vchunk_id_counter = 43
    updated_counter = aggregator.resolve_vchunk_ids(vchunk_id_counter, embedding_threshold=0.8)

    # Now the second track should reuse vchunk_id from the first track
    assert aggregator.tracks[1].vchunk_id == 42


def test_does_not_merge_dissimilar_embeddings():
    emb1 = np.ones(512, dtype=np.float32)
    emb2 = np.zeros(512, dtype=np.float32)
    emb1 /= np.linalg.norm(emb1)
    emb2[0] = 1.0  # Very different

    aggregator = ShotFaceTrackAggregator(shot_number=0)
    obs1 = dummy_observation(0, (10, 10, 50, 50), embedding=emb1)
    obs2 = dummy_observation(1, (100, 100, 150, 150), embedding=emb2)

    aggregator.update_tracks_with_frame(0, [obs1])
    aggregator.update_tracks_with_frame(1, [obs2])
    aggregator.finalize_tracks()

    aggregator.tracks[0].vchunk_id = 42
    aggregator.tracks[0].is_active = False

    updated_counter = aggregator.resolve_vchunk_ids(vchunk_id_counter=43)

    assert aggregator.tracks[1].vchunk_id == 43
    assert updated_counter == 44

def test_skips_tracks_without_embedding():
    obs = dummy_observation(0, (10, 10, 50, 50), embedding=None)
    aggregator = ShotFaceTrackAggregator(shot_number=0)
    aggregator.update_tracks_with_frame(0, [obs])
    aggregator.finalize_tracks()

    updated_counter = aggregator.resolve_vchunk_ids(vchunk_id_counter=5)

    assert aggregator.tracks[0].vchunk_id == 5
    assert updated_counter == 6


def test_preserves_preassigned_vchunk_id():
    emb = np.ones(512, dtype=np.float32)
    emb /= np.linalg.norm(emb)

    aggregator = ShotFaceTrackAggregator(shot_number=0)
    obs = dummy_observation(0, (0, 0, 10, 10), embedding=emb)
    aggregator.update_tracks_with_frame(0, [obs])
    aggregator.finalize_tracks()

    aggregator.tracks[0].vchunk_id = 77  # Preassigned
    updated_counter = aggregator.resolve_vchunk_ids(vchunk_id_counter=78)

    assert aggregator.tracks[0].vchunk_id == 77
    assert updated_counter == 78

def test_vchunk_id_reuse_on_embedding_similarity():
    shared_emb = np.ones(512, dtype=np.float32)
    aggregator = ShotFaceTrackAggregator(shot_number=0)

    # Two far-apart observations → separate tracks
    obs1 = dummy_observation(0, (0, 0, 50, 50), embedding=shared_emb)
    obs2 = dummy_observation(100, (500, 500, 550, 550), embedding=shared_emb)

    aggregator.update_tracks_with_frame(0, [obs1])
    aggregator.update_tracks_with_frame(100, [obs2])
    aggregator.finalize_tracks()
    assert len(aggregator.tracks) == 2

    # Assign ID to first track
    aggregator.tracks[0].vchunk_id = 10

    # Resolve IDs with high similarity threshold
    counter = aggregator.resolve_vchunk_ids(11, embedding_threshold=0.9)
    assert aggregator.tracks[1].vchunk_id == 10
    assert counter == 11

def test_no_vchunk_id_reuse_when_similarity_below_threshold():
    emb1 = np.ones(512, dtype=np.float32)
    emb2 = -np.ones(512, dtype=np.float32)  # Opposite direction
    aggregator = ShotFaceTrackAggregator(shot_number=0)

    obs1 = dummy_observation(0, (0, 0, 50, 50), embedding=emb1)
    obs2 = dummy_observation(10, (500, 500, 550, 550), embedding=emb2)

    aggregator.update_tracks_with_frame(0, [obs1])
    aggregator.update_tracks_with_frame(10, [obs2])
    aggregator.finalize_tracks()
    aggregator.tracks[0].vchunk_id = 10

    counter = aggregator.resolve_vchunk_ids(11, embedding_threshold=0.9)
    assert aggregator.tracks[1].vchunk_id != 10
    assert aggregator.tracks[1].vchunk_id == 11
    assert counter == 12

def test_conflict_resolution_reuses_highest_similarity_first():
    emb_ref = np.ones(512, dtype=np.float32)
    emb_close = np.ones(512, dtype=np.float32) * 0.99
    emb_far = np.ones(512, dtype=np.float32) * 0.5

    aggregator = ShotFaceTrackAggregator(shot_number=0)

    # Existing track with vchunk_id
    ref_track = FaceTrack(shot_id=0, track_id=0)
    ref_track.vchunk_id = 42
    ref_track.embeddings = [emb_ref]
    aggregator.tracks.append(ref_track)

    # Two unassigned tracks: one close, one far
    t1 = FaceTrack(shot_id=0, track_id=1)
    t1.embeddings = [emb_close]
    t2 = FaceTrack(shot_id=0, track_id=2)
    t2.embeddings = [emb_far]
    aggregator.tracks.extend([t1, t2])

    counter = aggregator.resolve_vchunk_ids(43, embedding_threshold=0.6)

    # ✅ Both should share ID 42 because of propagation logic
    assert t1.vchunk_id == 42, "Closest match should reuse vchunk_id 42"
    assert t2.vchunk_id == 42, "Propagation: far track joins same cluster via similarity chain"
    assert counter == 43, "Counter should remain unchanged since no new ID assigned"


def test_assigns_unique_vchunk_ids_when_no_embeddings():
    aggregator = ShotFaceTrackAggregator(shot_number=0)

    # Create tracks with no embeddings
    for i in range(3):
        track = FaceTrack(shot_id=0, track_id=i)
        aggregator.tracks.append(track)

    counter = aggregator.resolve_vchunk_ids(100)
    assigned_ids = [t.vchunk_id for t in aggregator.tracks]

    # All IDs should be unique and start from 100
    assert sorted(assigned_ids) == [100, 101, 102]
    assert counter == 103

def test_mixed_tracks_reuse_and_assign_new_ids():
    aggregator = ShotFaceTrackAggregator(shot_number=0)

    # Existing track with ID
    existing_track = FaceTrack(shot_id=0, track_id=0)
    existing_track.vchunk_id = 50
    existing_track.embeddings = [np.ones(512)]
    aggregator.tracks.append(existing_track)

    # Two unassigned tracks: one similar to existing, one dissimilar
    similar_track = FaceTrack(shot_id=0, track_id=1)
    similar_track.embeddings = [np.ones(512)]
    far_track = FaceTrack(shot_id=0, track_id=2)
    far_track.embeddings = [-np.ones(512)]
    aggregator.tracks.extend([similar_track, far_track])

    counter = aggregator.resolve_vchunk_ids(51, embedding_threshold=0.9)

    # Similar track reuses ID; far track gets new one
    assert similar_track.vchunk_id == 50
    assert far_track.vchunk_id == 51
    assert counter == 52

def test_tracks_without_embeddings_get_new_ids():
    aggregator = ShotFaceTrackAggregator(shot_number=0)

    # Existing track with embedding
    ref_track = FaceTrack(shot_id=0, track_id=0)
    ref_track.vchunk_id = 10
    ref_track.embeddings = [np.ones(512)]
    aggregator.tracks.append(ref_track)

    # Unassigned track with no embedding
    no_emb_track = FaceTrack(shot_id=0, track_id=1)
    aggregator.tracks.append(no_emb_track)

    counter = aggregator.resolve_vchunk_ids(11, embedding_threshold=0.9)

    assert no_emb_track.vchunk_id == 11  # Cannot reuse without embedding
    assert counter == 12

def test_counter_increments_correctly_for_multiple_new_assignments():
    aggregator = ShotFaceTrackAggregator(shot_number=0)

    for i in range(3):
        t = FaceTrack(shot_id=0, track_id=i)
        aggregator.tracks.append(t)

    counter = aggregator.resolve_vchunk_ids(200)
    assert counter == 203  # 3 tracks → increment by 3

