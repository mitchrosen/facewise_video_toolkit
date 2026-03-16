import numpy as np

from facekit.tracking.face_structures import FaceTrack, FaceObservation
from facekit.common.obs_consts import Source
from facekit.pipeline.track_across_segments import _attach_and_persist_embedded_obs
from facekit.tracking.aggregator import ShotFaceTrackAggregator

def _emb(seed: int) -> np.ndarray:
    v = np.zeros(512, dtype=np.float32)
    v[seed] = 1.0
    return v

def _obs(frame_idx, embedding, source, track_id):
    return FaceObservation(
        frame_idx=frame_idx,
        bbox=(0, 0, 10, 10),
        track_id=track_id,
        source=source,
        embedding=np.asarray(embedding, dtype=np.float32),
    )

def test_attach_embeddings_registers_track_embedding_samples():
    """
    Embedded observations that already belong to a track should register
    TrackEmbeddingSample metadata on that track.
    """

    aggregator = ShotFaceTrackAggregator(shot_number=1)

    track = FaceTrack(shot_id=1, track_id=1)
    track.observations = [
        _obs(10, _emb(0), Source.DETECTED, track_id=1),
        _obs(20, _emb(1), Source.TRACKED, track_id=1),
    ]

    aggregator.tracks.append(track)

    embedded_obs = [
        track.observations[0],
        track.observations[1],
    ]

    _attach_and_persist_embedded_obs(
        embedded_obs=embedded_obs,
        aggregator=aggregator,
        checkpoint=None,
        shot_number=1,
        shot_first_frame=0,
    )

    assert len(track.embedding_samples) == 2

    frames = [s.frame_idx for s in track.embedding_samples]
    assert frames == [10, 20]

    indices = [s.track_local_index for s in track.embedding_samples]
    assert indices == [0, 1]