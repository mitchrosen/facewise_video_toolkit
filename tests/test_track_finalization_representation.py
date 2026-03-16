import numpy as np

from facekit.tracking.face_structures import FaceTrack
from facekit.tracking.aggregator import ShotFaceTrackAggregator
from facekit.embedding.embedding_selection import TrackEmbeddingSample
from facekit.common.obs_consts import Source


def _emb(seed):
    v = np.zeros(512, dtype=np.float32)
    v[seed] = 1.0
    return v


def test_finalize_tracks_computes_representative_embedding():
    """
    When the aggregator finalizes tracks, the track should compute its
    representative embedding from collected embedding samples.
    """

    aggregator = ShotFaceTrackAggregator(shot_number=1)

    track = FaceTrack(shot_id=1, track_id=1)

    track.embedding_samples = [
        TrackEmbeddingSample(10, 0, Source.DETECTED, _emb(0)),
        TrackEmbeddingSample(20, 1, Source.TRACKED, _emb(1)),
        TrackEmbeddingSample(30, 2, Source.TRACKED, _emb(2)),
    ]

    aggregator.tracks.append(track)

    aggregator.finalize_tracks()

    assert track.representative_embedding is not None
    assert track.embedding_stable is True