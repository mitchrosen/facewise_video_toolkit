import numpy as np
from facekit.output.json_v2 import ObservationsCollector
from facekit.pipeline.resume_rehydrate import rehydrate_observation_tracks


def test_pre_anchor_tracks_produce_drawable_rects():
    oc = ObservationsCollector()
    oc.append_track_obs(
        [
            {
                "shot": 1,
                "track_id": 2,
                "f": 3,
                "bbox_xyxy": [1.2, 2.8, 30.4, 40.9],
                "src": "detected",
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

    for o in t.observations:
        x1, y1, x2, y2 = o.bbox
        assert all(isinstance(v, int) for v in (x1, y1, x2, y2))
        assert all(np.isfinite([x1, y1, x2, y2]))
        assert x2 > x1 and y2 > y1
