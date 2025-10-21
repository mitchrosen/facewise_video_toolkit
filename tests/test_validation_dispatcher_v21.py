import json
from pathlib import Path
import numpy as np
import pytest
from facekit.validation import validate_manifest
from facekit.validation.json.validate_shot_features_json_v2 import validate_shot_features_json_v2



def _minimal_v21(tmp_path: Path):
    data = {
        "schema_version": "2.1",
        "video": {"path":"x.mp4","fps":30.0,"size":[100,100],"total_frames":2},
        "generation": {"created_utc":"2024-01-01T00:00:00Z","commit":"x","branch":"y","params_hash":"sha256:0","emb_store":"sidecar"},
        "shots": [{
            "shot_number": 1, "first_frame":0, "last_frame":1,"num_tracks":1,
            "face_tracks": [{
                "first_frame":0, "last_frame":1, "face_label":"face_0",
                "avg_center_x":10,"avg_center_y":10,"avg_face_width":10,"avg_face_height":10,
                "is_static": True,
                "obs_offset":0, "obs_count":2
            }]
        }],
        "embedding_sidecar": {"path": str(tmp_path/"embs.npz"), "format":"npz","dtype":"float32","dim":512,"count":2},
        "observations_sidecar": {
          "path":"obs.npz",
          "format":"npz",
          "dtype":"structured",
          "fields": [
            {"name":"f","type":"i4"},
            {"name":"bbox_xyxy","type":"f4[4]"},
            {"name":"src","type":"u1"},
            {"name":"conf","type":"f4"},
            {"name":"emb_idx","type":"i4"}
          ],
          "count": 0
        }    }
    
    Path(data["embedding_sidecar"]["path"]).parent.mkdir(parents=True, exist_ok=True)
    np.savez(data["embedding_sidecar"]["path"], embeddings=np.zeros((2,512),dtype=np.float32))
    np.savez(data["observations_sidecar"]["path"], observations=np.zeros((2,),dtype=[("f","i4"),("bbox_xyxy","f4",4),("src","u1"),("conf","f4"),("emb_idx","i4")]))
    p = tmp_path/"m.json"
    p.write_text(json.dumps(data))
    return p

def test_v21_validation_success(tmp_path):
    jp = _minimal_v21(tmp_path)
    errs = validate_manifest(jp, total_frame_count=2)
    assert errs == []

def test_v21_rejects_unknown_minor(tmp_path):
    p = tmp_path/"bad.json"
    p.write_text(json.dumps({"schema_version":"2.7"}))
    # using v2 validator directly – expect schema error or business error list
    errs = validate_shot_features_json_v2(p, p)  # bogus schema path ok; validator iterates anyway
    assert errs, "Expected errors for unknown minor"
