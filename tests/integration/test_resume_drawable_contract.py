import numpy as np
from facekit.output.json_v2 import ObservationsCollector
from facekit.pipeline.resume_rehydrate import rehydrate_observation_tracks
from facekit.common.obs_consts import Source


def test_pre_anchor_tracks_produce_drawable_rects():
    oc = ObservationsCollector()

    # Provide landmarks for detected rows (5x2). Tracked rows may omit them.
    lms = np.array(
        [
            [10.0, 11.0],
            [12.0, 13.0],
            [14.0, 15.0],
            [16.0, 17.0],
            [18.0, 19.0],
        ],
        dtype=np.float32,
    ).tolist()

    oc.append_track_obs(
        [
            {
                "shot": 1,
                "track_id": 2,
                "f": 3,
                "bbox_xyxy": [1.2, 2.8, 30.4, 40.9],
                "src": "detected",
                "landmarks": lms,
            },
            {
                "shot": 1,
                "track_id": 2,
                "f": 4,
                "bbox_xyxy": [5.0, 6.0, 50.0, 60.0],
                "src": "tracked",
                # landmarks intentionally omitted
            },
        ],
        emb_idx_fn=lambda _: -1,
    )

    tracks = rehydrate_observation_tracks(oc, frame_max=10, track_order={(1, 2): 0})
    assert tracks
    t = tracks[0]

    for o in t.observations:
        # --- bbox must be drawable (finite; positive area after int conversion) ---
        x1, y1, x2, y2 = o.bbox
        vals = np.asarray([x1, y1, x2, y2], dtype=np.float32)
        assert np.all(np.isfinite(vals))

        ix1, iy1, ix2, iy2 = map(int, vals.tolist())
        assert ix2 > ix1 and iy2 > iy1

        # --- landmarks contract: detected obs should carry landmarks after rehydrate ---
        if o.source == Source.DETECTED:
            assert o.landmarks is not None
            arr = np.asarray(o.landmarks, dtype=np.float32)
            assert arr.shape == (5, 2)
            assert np.any(arr != 0.0)
