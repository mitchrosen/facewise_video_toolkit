import numpy as np, pytest
from facekit.tracking.aggregator import ShotFaceTrackAggregator
from facekit.tracking.face_structures import FaceObservation

def _one_track():
    ag = ShotFaceTrackAggregator(shot_number=1)
    ag.update_tracks_with_frame(0,[FaceObservation(0,(0,0,1,1))])
    return ag, ag.tracks[0].track_id

def test_attach_embeddings_rejects_non_array():
    ag, tid = _one_track()
    with pytest.raises(TypeError):
        ag.attach_embeddings(tid, embeddings=[np.ones((512,), np.float32)])  # list → TypeError

def test_attach_embeddings_rejects_nan_inf():
    ag, tid = _one_track()
    x = np.ones((2,512), dtype=np.float32)
    x[1,0] = np.inf
    with pytest.raises(ValueError):
        ag.attach_embeddings(tid, x)
