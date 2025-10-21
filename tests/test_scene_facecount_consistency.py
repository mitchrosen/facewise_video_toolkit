import json
from pathlib import Path
import pytest
from facekit.validation.json.validate_shot_features_json_v2 import validate_shot_features_json_v2


def test_facecount_vs_facedetails_rule(tmp_path):
    # Minimal v2.0-like scene JSON (your business rules live in the v2 validator file)
    data = {
        "schema_version": "2.0",
        "video": {"path":"x", "fps":30, "size":[10,10], "total_frames": 2},
        "generation": {"created_utc":"2024-01-01T00:00:00Z","commit":"x","branch":"y","params_hash":"p","emb_store":"none"},
        "shots":[
            {"shot_number":1,"first_frame":0,"last_frame":1,"num_tracks":0,"face_tracks":[]},
        ],
        # emulate scene-level rule if your validator checks another JSON too:
        "detected_faces":{
            "face_count": 0,
            "face_details": [[1,2,3,4]]  # should violate your business rule
        }
    }
    p = tmp_path/"s.json"
    p.write_text(json.dumps(data))

    # For schema, point to itself; we want the business-rule pass to run and report errors.
    errs = validate_shot_features_json_v2(p, p, total_frame_count=2)
    assert any("face_count == 0" in e or "face_count" in e for e in errs)
