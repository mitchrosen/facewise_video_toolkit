import numpy as np

from facekit.output.json_v2 import ObservationsCollector, LANDMARKS_SHAPE
from facekit.pipeline.resume_rehydrate import rehydrate_observation_tracks

def _nonzero_landmarks() -> list[list[float]]:
    # 5-point landmarks, clearly non-zero and deterministic
    return [
        [10.0, 20.0],
        [30.0, 40.0],
        [50.0, 60.0],
        [70.0, 80.0],
        [90.0, 100.0],
    ]

def _assert_landmarks_present(lms) -> None:
    """
    Accept either list-of-lists or numpy array, but it must be 5x2 and non-zero.
    """
    arr = np.asarray(lms, dtype=np.float32)
    assert arr.shape == LANDMARKS_SHAPE
    assert np.any(arr != 0.0)

def _assert_landmarks_absent(lms) -> None:
    """
    Accept None or all-zeros. ObservationsCollector uses all-zeros as "absent".
    """
    if lms is None:
        return
    arr = np.asarray(lms, dtype=np.float32)
    assert arr.shape == LANDMARKS_SHAPE
    assert not np.any(arr != 0.0)

def test_landmarks_roundtrip_through_collector_iter_tracks():
    oc = ObservationsCollector()

    oc.append_track_obs(
        [
            {
                "shot": 1,
                "track_id": 7,
                "f": 10,
                "bbox_xyxy": [10.0, 20.0, 110.0, 220.0],
                "src": "detected",
                "landmarks": _nonzero_landmarks(),
            },
            {
                # No landmarks on tracked frame (common), should serialize as zeros/absent
                "shot": 1,
                "track_id": 7,
                "f": 11,
                "bbox_xyxy": [12.0, 22.0, 112.0, 222.0],
                "src": "tracked",
            },
        ],
        emb_idx_fn=lambda _: -1,
    )

    groups = list(oc.iter_tracks())
    assert groups, "Expected at least one (shot, track_id, rows) group"
    (shot, tid, rows) = groups[0]
    assert shot == 1
    assert tid == 7
    assert [r["f"] for r in rows] == [10, 11]

    # iter_tracks only includes "landmarks" key when present+nonzero
    assert "landmarks" in rows[0]
    _assert_landmarks_present(rows[0]["landmarks"])

    assert "landmarks" not in rows[1]

def test_find_rows_only_with_landmarks_filters_correctly():
    oc = ObservationsCollector()
    oc.append_track_obs(
        [
            {
                "shot": 1,
                "track_id": 2,
                "f": 3,
                "bbox_xyxy": [1.0, 2.0, 30.0, 40.0],
                "src": "detected",
                "landmarks": _nonzero_landmarks(),
            },
            {
                "shot": 1,
                "track_id": 2,
                "f": 4,
                "bbox_xyxy": [5.0, 6.0, 50.0, 60.0],
                "src": "tracked",
                # no landmarks -> stored as zeros
            },
        ],
        emb_idx_fn=lambda _: -1,
    )

    # Without filter: both rows are candidates
    all_pos = oc.find_rows(shot=1, track_id=2, count=10)
    assert len(all_pos) == 2

    # With only_with_landmarks: only the detected row should match
    lm_pos = oc.find_rows(shot=1, track_id=2, count=10, only_with_landmarks=True)
    assert len(lm_pos) == 1
    assert oc.frame_at_pos(lm_pos[0]) == 3

def test_landmarks_survive_npz_dump_and_load(tmp_path):
    oc1 = ObservationsCollector()
    oc1.append_track_obs(
        [
            {
                "shot": 2,
                "track_id": 1,
                "f": 100,
                "bbox_xyxy": [100.0, 100.0, 200.0, 220.0],
                "src": "detected",
                "landmarks": _nonzero_landmarks(),
            },
            {
                "shot": 2,
                "track_id": 1,
                "f": 101,
                "bbox_xyxy": [101.0, 101.0, 201.0, 221.0],
                "src": "tracked",
            },
        ],
        emb_idx_fn=lambda _: -1,
    )

    npz_path = tmp_path / "obs_sidecar.npz"
    oc1.dump_npz(npz_path)

    oc2 = ObservationsCollector()
    loaded = oc2.load_npz(npz_path)
    assert loaded == 2

    groups = list(oc2.iter_tracks(shot=2, track_id=1))
    assert len(groups) == 1
    (_, _, rows) = groups[0]
    assert [r["f"] for r in rows] == [100, 101]

    assert "landmarks" in rows[0]
    _assert_landmarks_present(rows[0]["landmarks"])
    assert "landmarks" not in rows[1]

def test_rehydrate_observation_tracks_preserves_landmarks_and_bbox_ints():
    oc = ObservationsCollector()
    oc.append_track_obs(
        [
            {
                "shot": 1,
                "track_id": 2,
                "f": 3,
                "bbox_xyxy": [1.2, 2.8, 30.4, 40.9],
                "src": "detected",
                "landmarks": _nonzero_landmarks(),
            },
            {
                "shot": 1,
                "track_id": 2,
                "f": 4,
                "bbox_xyxy": [5.0, 6.0, 50.0, 60.0],
                "src": "tracked",
            },
        ],
        emb_idx_fn=lambda _: -1,
    )

    tracks = rehydrate_observation_tracks(
        oc, frame_max=10, track_order={(1, 2): 0}
    )
    assert tracks
    t = tracks[0]

    # Ensure bbox is drawable: ints and sane
    assert len(t.observations) == 2
    for ob in t.observations:
        x1, y1, x2, y2 = ob.bbox
        assert all(isinstance(v, int) for v in (x1, y1, x2, y2))
        assert all(np.isfinite([x1, y1, x2, y2]))
        assert x2 > x1 and y2 > y1

    # Landmark expectations: present on detected frame, absent on tracked frame
    o0 = t.observations[0]
    o1 = t.observations[1]

    # We don’t assume the exact type (list vs ndarray), only shape and non-zero vs absent.
    lm0 = getattr(o0, "landmarks", None)
    lm1 = getattr(o1, "landmarks", None)

    _assert_landmarks_present(lm0)
    _assert_landmarks_absent(lm1)
