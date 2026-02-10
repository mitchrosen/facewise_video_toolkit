from facekit.output.json_v2 import ObservationsCollector
from facekit.common.obs_consts import Source

def test_iter_tracks_groups_orders_and_filters_by_frame_max():
    oc = ObservationsCollector()
    oc.append_track_obs(
        [
            {"shot": 1, "track_id": 7, "f": 5, "bbox_xyxy": [0, 0, 10, 10], "src": Source.DETECTED},
            {"shot": 1, "track_id": 7, "f": 9, "bbox_xyxy": [1, 1, 11, 11], "src": Source.TRACKED},
            {"shot": 1, "track_id": 7, "f": 12, "bbox_xyxy": [2, 2, 12, 12], "src": Source.DETECTED},
        ],
        emb_idx_fn=lambda _: -1,
    )

    groups_all = list(oc.iter_tracks(frame_max=None))
    groups_10  = list(oc.iter_tracks(frame_max=10))

    assert len(groups_all) == 1
    s,t,rows_all = groups_all[0]
    assert (s,t) == (1,7)
    assert [r["f"] for r in rows_all] == [5,9,12]

    s2,t2,rows_10 = groups_10[0]
    assert (s2,t2) == (1,7)
    assert [r["f"] for r in rows_10] == [5,9]
