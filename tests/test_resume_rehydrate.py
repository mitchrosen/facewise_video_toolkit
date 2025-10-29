import numpy as np
import pytest
from facekit.output.json_v2 import ObservationsCollector
from facekit.common.obs_consts import Source
from facekit.pipeline.resume_rehydrate import rehydrate_tracks_from_observations


def _rows(*items):
    return list(items)

def test_rehydrate_sanitizes_and_filters_bad_rows():
    oc = ObservationsCollector()
    oc.append_track_obs(_rows(
        # ok
        {"shot":1,"track_id":7,"f":5,"bbox_xyxy":[0.0,0.0,10.0,10.0],"src":Source.DETECTED,"conf":0.9},
        # reversed coords -> should be fixed or dropped by sanitize (we expect fixed)
        {"shot":1,"track_id":7,"f":6,"bbox_xyxy":[12.0,12.0,2.0,2.0],"src":Source.TRACKED},
        # NaNs -> should be dropped
        {"shot":1,"track_id":7,"f":7,"bbox_xyxy":[np.nan,2.0,12.0,12.0],"src":Source.DETECTED},
        # zero-width -> should be dropped
        {"shot":1,"track_id":7,"f":8,"bbox_xyxy":[5.0,5.0,5.0,10.0],"src":Source.TRACKED},
    ), emb_idx_fn=lambda _: -1)

    tracks = rehydrate_tracks_from_observations(
        oc, frame_max=100, track_order={(1, 7): 0}
    )
    assert len(tracks) == 1
    t = tracks[0]
    # Only valid rows remain; reversed one should be corrected.
    frames = [o.frame_idx for o in t.observations]
    assert frames == sorted(frames)
    for o in t.observations:
        x1,y1,x2,y2 = o.bbox
        # ints & finite & proper rectangle
        assert all(isinstance(v, int) for v in (x1,y1,x2,y2))
        assert x2 > x1 and y2 > y1
