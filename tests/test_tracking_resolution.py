import numpy as np
from facekit.tracking.face_structures import FaceObservation, FaceTrack
from facekit.tracking.tracking_resolution import GlobalIdentityResolver

def dummy_observation(frame_idx, bbox, embedding=None):
    return FaceObservation(frame_idx=frame_idx, bbox=bbox, embedding=embedding, source='detection')

def dummy_track(segment_id, track_id, emb_val):
    obs = dummy_observation(0, (0, 0, 10, 10), embedding=emb_val)
    track = FaceTrack(track_id=track_id, shot_id=0, segment_id=segment_id, observations=[obs])
    track.embeddings = [emb_val]  # Ensure it's explicitly populated
    return track

def test_global_id_resolution_unique():
    """
    Each segment_id represents a unique face. Expect global_id to be 0,1,2...
    """
    def normalize(vec):
        return vec / np.linalg.norm(vec)

    emb0 = normalize(np.eye(512)[0].astype(np.float32))
    emb1 = normalize(np.eye(512)[1].astype(np.float32))
    emb2 = normalize(np.eye(512)[2].astype(np.float32))

    tracks = [
        dummy_track(0, 0, emb_val=emb0),
        dummy_track(1, 0, emb_val=emb1),
        dummy_track(2, 0, emb_val=emb2),
    ]

    resolver = GlobalIdentityResolver()
    resolver.resolve_global_ids(tracks)

    global_ids = [t.global_id for t in tracks]
    assert sorted(global_ids) == [0, 1, 2]

def test_global_id_resolution_merges_similar_embeddings():
    shared_embedding = np.ones(512, dtype=np.float32)

    tracks = []
    for i in range(3):
        t = FaceTrack(shot_id=0, track_id=i)
        obs = FaceObservation(frame_idx=i, bbox=(0, 0, 10, 10), embedding=shared_embedding, source='detection')
        t.add_observation(obs)
        tracks.append(t)

    resolver = GlobalIdentityResolver()
    resolver.resolve_global_ids(tracks)

    global_ids = {track.global_id for track in tracks}
    assert global_ids == {0}, "All tracks should share global_id since they share embedding"

# def test_global_id_resolution_skips_tracks_without_embedding():
#     """
#     Tracks without embeddings should be assigned unique global_ids.
#     """
#     obs1 = dummy_observation(frame_idx=0, bbox=(0, 0, 10, 10), embedding=None)
#     obs2 = dummy_observation(frame_idx=1, bbox=(10, 10, 20, 20), embedding=None)

#     track1 = FaceTrack(track_id=0, shot_id=0, observations=[obs1], segment_id=0)
#     track2 = FaceTrack(track_id=1, shot_id=0, observations=[obs2], segment_id=1)

#     tracks = [
#         track1,
#         track2
#     ]

#     resolver = GlobalIdentityResolver()
#     resolver.resolve_global_ids(tracks)

#     global_ids = {track.global_id for track in tracks}
#     assert global_ids == {0, 1}

def test_global_id_resolution_does_not_merge_dissimilar_embeddings():
    emb1 = np.zeros(512, dtype=np.float32)
    emb1[0] = 1.0
    emb2 = np.zeros(512, dtype=np.float32)
    emb2[511] = 1.0

    track1 = FaceTrack(track_id=0, shot_id=0, observations=[dummy_observation(0, (0, 0, 10, 10), emb1)], segment_id=0)
    track1.embeddings = [emb1]
    track2 = FaceTrack(track_id=1, shot_id=0, observations=[dummy_observation(1, (10, 10, 20, 20), emb2)], segment_id=1)
    track2.embeddings = [emb2]

    tracks = [
        track1,
        track2
    ]

    resolver = GlobalIdentityResolver(embedding_threshold=0.7)
    resolver.resolve_global_ids(tracks)

    assert track1.global_id != track2.global_id
