import numpy as np
import pytest

from facekit.output.json_v2 import ObservationsCollector  

def _make_landmarks(k: int = 5) -> np.ndarray:
    # Small, deterministic landmark set
    xy = np.zeros((k, 2), dtype=np.float32)
    for i in range(k):
        xy[i, 0] = 10.0 + i
        xy[i, 1] = 20.0 + 2 * i
    return xy

def test_obs_sidecar_persists_landmarks_and_rehydrates(tmp_path):
    """
    Contract:
      - If append_track_obs() is given landmarks, dump_npz() must persist them.
      - Rehydration iterator must expose them for downstream FaceObservation reconstruction.
    """
    out = tmp_path / "observations.npz"

    obs_collector = ObservationsCollector()
    landmarks = _make_landmarks(5)

    # One DET row with landmarks
    obs_collector.append_track_obs([{
        "shot": 0,
        "track_id": 1,
        "f": 123,
        "bbox_xyxy": [1, 2, 3, 4],
        "src": "detected",
        "conf": 0.9,
        "emb_idx": -1,
        "landmarks": landmarks,
    }], emb_idx_fn=lambda _r: -1)

    obs_collector.dump_npz(out)
    assert out.exists()

    arr = np.load(out, allow_pickle=False)["observations"]
    assert "landmarks" in (arr.dtype.names or ()), (
        "Expected 'landmarks' field in observations sidecar dtype after dump. "
        f"dtype names = {list(arr.dtype.names or [])}"
    )

    # Verify persisted values are finite and match shape at least in row-major sense
    persisted = np.asarray(arr["landmarks"][0])
    assert persisted.shape == landmarks.shape, f"Landmarks shape changed: {persisted.shape} vs {landmarks.shape}"
    assert np.allclose(persisted, landmarks)

    # Now verify rehydration iterator exposes landmarks
    obs_collector2 = ObservationsCollector()
    obs_collector2.load_npz(str(out))          
    tracks = list(obs_collector2.iter_rows()) 
    assert len(tracks) == 1
    (shot, tid, obs_list) = tracks[0]
    assert shot == 0 and tid == 1
    assert len(obs_list) == 1
    obs = obs_list[0]
    assert "landmarks" in obs, "iter_tracks() must include landmarks in observation dicts for rehydration."
    assert np.allclose(np.asarray(obs["landmarks"]), landmarks)
