import numpy as np
from facekit.tracking.face_structures import FaceObservation, FaceTrack
from facekit.tracking.tracking_resolution import GlobalIdentityResolver


def dummy_observation(frame_idx, bbox, embedding=None):
    return FaceObservation(frame_idx=frame_idx, bbox=bbox, embedding=embedding)


def dummy_track(vchunk_id, track_id, emb_val):
    emb = np.zeros(512, dtype=np.float32)
    emb[emb_val % 512] = 1.0
    obs = dummy_observation(frame_idx=0, bbox=(0, 0, 10, 10), embedding=emb)
    return FaceTrack(track_id=track_id, shot_id=0, observations=[obs], vchunk_id=vchunk_id)


def test_global_id_resolution_unique():
    """
    Each vchunk_id represents a unique face. Expect global_id to be 0,1,2...
    """
    tracks = [
        dummy_track(0, 0, emb_val=0),
        dummy_track(1, 0, emb_val=1),
        dummy_track(2, 0, emb_val=2),
    ]

    resolver = GlobalIdentityResolver()
    resolver.resolve_global_ids(tracks)

    global_ids = {track.global_id for track in tracks}
    assert global_ids == {0, 1, 2}


def test_global_id_resolution_merges_similar_embeddings():
    shared_embedding = np.ones(512, dtype=np.float32)

    tracks = []
    for i in range(3):
        t = FaceTrack(shot_id=0, track_id=i)
        obs = FaceObservation(frame_idx=i, bbox=(0, 0, 10, 10), embedding=shared_embedding)
        t.add_observation(obs)
        tracks.append(t)

    resolver = GlobalIdentityResolver()
    resolver.resolve_global_ids(tracks)

    global_ids = {track.global_id for track in tracks}
    assert global_ids == {0}, "All tracks should share global_id since they share embedding"



def test_global_id_resolution_skips_tracks_without_embedding():
    """
    Tracks without embeddings should be assigned unique global_ids.
    """
    obs1 = dummy_observation(frame_idx=0, bbox=(0, 0, 10, 10), embedding=None)
    obs2 = dummy_observation(frame_idx=1, bbox=(10, 10, 20, 20), embedding=None)

    track1 = FaceTrack(track_id=0, shot_id=0, observations=[obs1], vchunk_id=0)
    track2 = FaceTrack(track_id=1, shot_id=0, observations=[obs2], vchunk_id=1)

    tracks = [
        track1,
        track2
    ]

    resolver = GlobalIdentityResolver()
    resolver.resolve_global_ids(tracks)

    global_ids = {track.global_id for track in tracks}
    assert global_ids == {0, 1}


def test_global_id_resolution_does_not_merge_dissimilar_embeddings():
    emb1 = np.zeros(512, dtype=np.float32)
    emb1[0] = 1.0
    emb2 = np.zeros(512, dtype=np.float32)
    emb2[511] = 1.0

    track1 = FaceTrack(track_id=0, shot_id=0, observations=[dummy_observation(0, (0, 0, 10, 10), emb1)], vchunk_id=0)
    track2 = FaceTrack(track_id=1, shot_id=0, observations=[dummy_observation(1, (10, 10, 20, 20), emb2)], vchunk_id=1)

    tracks = [
        track1,
        track2
    ]

    resolver = GlobalIdentityResolver(embedding_threshold=0.7)
    resolver.resolve_global_ids(tracks)

    assert track1.global_id != track2.global_id
