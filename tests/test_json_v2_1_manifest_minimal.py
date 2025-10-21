import numpy as np
import pytest
from facekit.output.json_v2 import (
    V2WriterConfig, build_v2_1_manifest_from_tracks,
    EmbeddingCollector, ObservationsCollector, write_v2_json
)
from facekit.common.obs_consts import Source


def test_v21_shape_offsets_and_sidecars(tmp_path):
    class Obs:  # minimal
        def __init__(self, f, bb, emb=None, shot_id=1, track_id=1):
            self.frame_idx = f
            self.bbox = bb
            self.embedding = emb
            self.source = Source.DETECTED
            self.confidence = 0.9
            self.shot_id = shot_id
            self.track_id = track_id
    class T:
        def __init__(self):
            self.shot_id=1; self.track_id=1; self.global_id=0
            self.observations=[Obs(0,(0,0,10,10),np.zeros(512,dtype=np.float32)),
                               Obs(1,(1,1,11,11),np.zeros(512,dtype=np.float32))]

    cfg = V2WriterConfig(video_path="x.mp4", video_size=(100,100), total_frames=2, fps=30.0,
                         emb_store="sidecar", emb_sidecar_path=tmp_path/"embs.npz")

    embc = EmbeddingCollector("sidecar", dim=512)
    obsc = ObservationsCollector()
    m = build_v2_1_manifest_from_tracks([T()], cfg,
                                        face_metadata=[],
                                        detector=None, embedder=None,
                                        tracking_params={"detect_interval": 3},
                                        validator=None,
                                        emb_collector=embc,
                                        obs_collector=obsc)
    # basic shape
    assert m["schema_version"] == "2.1"
    assert "obs_offset" in m["shots"][0]["face_tracks"][0]
    assert "obs" not in m["shots"][0]["face_tracks"][0]

    # sidecars
    e_desc = embc.finalize_sidecar(cfg.emb_sidecar_path)
    o_desc = obsc.finalize_sidecar(tmp_path/"obs.npz")
    assert (tmp_path/"embs.npz").exists()
    assert (tmp_path/"obs.npz").exists()

    # write and sanity round-trip
    out = tmp_path/"m.json"
    write_v2_json(str(out), m)
    assert out.exists()
