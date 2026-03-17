import numpy as np
import pytest
from facekit.output.json_v2 import ObservationsCollector
from facekit.common.obs_consts import Source
from facekit.pipeline.resume_rehydrate import reconstruct_tracks_from_observations, prepare_tracks_for_resume

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

    tracks = reconstruct_tracks_from_observations(
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

def test_append_track_obs_allows_detected_without_landmarks():
    oc = ObservationsCollector()
    # Detected rows may omit landmarks under the current contract (no persisted landmarks).
    oc.append_track_obs(
        [{"shot":1,"track_id":1,"f":1,"bbox_xyxy":[0,0,10,10],"src":Source.DETECTED}],
        emb_idx_fn=lambda _: -1
    )

def test_rehydrate_tracks_rebuilds_track_level_identity_embedding():
    """
    Rehydration should restore usable track-level identity state for any track
    that has durable pre-anchor DET embeddings.

    This is the state that global identity resolution now prefers when present.
    A resumed run should therefore not stop at merely repopulating track.embeddings;
    it should also provide a representative_embedding and mark it stable.
    """
    collector = ObservationsCollector()

    collector.append_track_obs(
        [
            {
                "shot": 1,
                "track_id": 7,
                "f": 100,
                "bbox_xyxy": [0.0, 0.0, 10.0, 10.0],
                "src": Source.DETECTED,
                "conf": 0.9,
            },
            {
                "shot": 1,
                "track_id": 7,
                "f": 108,
                "bbox_xyxy": [1.0, 1.0, 11.0, 11.0],
                "src": Source.DETECTED,
                "conf": 0.95,
            },
        ],
        emb_idx_fn=lambda _: -1,
    )

    emb0 = np.zeros((512,), dtype=np.float32)
    emb0[0] = 1.0

    emb1 = np.zeros((512,), dtype=np.float32)
    emb1[0] = 1.0

    def emb_lookup(shot: int, track_id: int):
        if (shot, track_id) != (1, 7):
            return None
        return [100, 108], np.stack([emb0, emb1], axis=0)

    tracks = prepare_tracks_for_resume(
        collector,
        frame_max=200,
        track_order={(1, 7): 0},
        emb_lookup=emb_lookup,
        emb_array_lookup=None,
        anchor_shot_id=2,   # shot 1 is treated as completed/pre-anchor
        strict=True,
    )

    assert len(tracks) == 1
    track = tracks[0]

    # Sanity: embeddings were attached at the observation/track level.
    assert len(track.embeddings) == 2
    assert track.has_embedding()

    # Critical resume contract for identity resolution:
    assert track.embedding_stable is True
    assert track.representative_embedding is not None

    expected = np.mean(np.stack([emb0, emb1], axis=0), axis=0)
    np.testing.assert_allclose(
        track.representative_embedding,
        expected,
        atol=1e-6,
    )