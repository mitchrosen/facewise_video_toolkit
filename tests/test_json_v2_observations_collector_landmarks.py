import numpy as np

from facekit.output.json_v2 import ObservationsCollector, LANDMARKS_SHAPE
from facekit.common.obs_consts import src_to_code
from facekit.tracking.face_structures import FaceObservation
from facekit.common.obs_consts import Source

def _mk_landmarks() -> np.ndarray:
    # Deterministic non-zero landmarks (5x2)
    lm = np.zeros(LANDMARKS_SHAPE, dtype=np.float32)
    for i in range(LANDMARKS_SHAPE[0]):
        lm[i, 0] = 10.0 + i
        lm[i, 1] = 20.0 + 2 * i
    return lm

def test_observations_collector_landmarks_roundtrip_npz_and_iter_tracks(tmp_path):
    """
    Contract:
      - append_track_obs(..., landmarks=nonzero) persists landmarks into ObsRow.
      - dump_npz + load_npz preserves them.
      - iter_tracks rehydrates them (as nested lists) when non-zero.
      - zeros mean "absent" and are not emitted by iter_tracks.
    """
    obs_collector = ObservationsCollector()

    det_code = int(src_to_code("detected"))
    trk_code = int(src_to_code("tracked"))

    landmarks = _mk_landmarks()

    # Two rows in same (shot,track): first has landmarks, second has "absent" landmarks (zeros)
    obs_collector.append_track_obs(
        [
            {
                "shot": 7,
                "track_id": 3,
                "f": 100,
                "bbox_xyxy": [1.0, 2.0, 11.0, 22.0],
                "src": det_code,  # int ok
                "conf": 0.9,
                "landmarks": landmarks,
            },
            {
                "shot": 7,
                "track_id": 3,
                "f": 101,
                "bbox_xyxy": [2.0, 3.0, 12.0, 23.0],
                "src": trk_code,
                "conf": 0.8,
                # landmarks absent -> zeros
            },
        ]
    )

    out = tmp_path / "obs.npz"
    obs_collector.dump_npz(out)

    # === Verify persisted dtype + content ===
    arr = np.load(out, allow_pickle=False)["observations"]
    assert "landmarks" in (arr.dtype.names or ()), f"dtype names: {list(arr.dtype.names or [])}"
    assert arr.shape[0] == 2

    persisted0 = np.asarray(arr["landmarks"][0], dtype=np.float32)
    persisted1 = np.asarray(arr["landmarks"][1], dtype=np.float32)
    assert persisted0.shape == LANDMARKS_SHAPE
    assert np.allclose(persisted0, landmarks)
    assert np.allclose(persisted1, np.zeros(LANDMARKS_SHAPE, dtype=np.float32))

    # === Rehydrate via load_npz + iter_tracks ===
    oc2 = ObservationsCollector()
    n = oc2.load_npz(out)
    assert n == 2

    groups = list(oc2.iter_tracks())
    assert groups, "Expected at least one (shot,track) group"
    assert groups[0][0] == 7
    assert groups[0][1] == 3
    rows = groups[0][2]
    assert [r["f"] for r in rows] == [100, 101]

    # First row has landmarks emitted, second row does not (zeros treated as absent)
    assert "landmarks" in rows[0]
    assert np.allclose(np.asarray(rows[0]["landmarks"], dtype=np.float32), landmarks)
    assert "landmarks" not in rows[1]

    # === Demonstrate rehydration into FaceObservation objects keeps landmarks ===
    # (This mirrors what your resume path should do somewhere.)
    rehydrated = []
    for r in rows:
        row_landmarks = r.get("landmarks")
        fo = FaceObservation(
            frame_idx=int(r["f"]),
            bbox=tuple(r["bbox_xyxy"]),
            source=r["src"] if isinstance(r["src"], Source) else Source(str(r["src"])),
            confidence=float(r["conf"]) if "conf" in r else None,
            landmarks=np.asarray(row_landmarks, dtype=np.float32) if row_landmarks is not None else None,
        )
        rehydrated.append(fo)

    assert rehydrated[0].landmarks is not None
    assert np.allclose(np.asarray(rehydrated[0].landmarks, dtype=np.float32), landmarks)
    assert rehydrated[1].landmarks is None
