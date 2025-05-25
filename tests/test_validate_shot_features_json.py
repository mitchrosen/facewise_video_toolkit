import os
import pytest
import json
from jsonschema.exceptions import ValidationError
from facekit.postprocessing.validate_shot_features_json import validate_shot_features_json

from pathlib import Path
SCHEMA_PATH = Path(__file__).parent.parent / "schemas" / "shot_features.schema.json"

VALID_JSON_EXAMPLE = {
    "shots": [
        {
            "shot_number": 1,
            "first_frame": 0,
            "last_frame": 100,
            "detected_faces": {
                "face_count": 1,
                "face_details": [
                    {
                        "center_x": 50.0,
                        "center_y": 50.0,
                        "face_width": 10.0,
                        "face_height": 20.0
                    }
                ]
            },
            "detected_graphics": {}
        },
        {
            "shot_number": 2,
            "first_frame": 101,
            "last_frame": 200,
            "detected_faces": {
                "face_count": 0
            },
            "detected_graphics": {}
        }
    ]
}

def test_valid_json_passes(tmp_path):
    path = tmp_path / "valid.json"
    with open(path, "w") as f:
        json.dump(VALID_JSON_EXAMPLE, f)

    errors = validate_shot_features_json(str(path), SCHEMA_PATH, total_frame_count=201)
    assert errors == []


def test_missing_required_field_returns_error(tmp_path):
    bad_data = {
        "shots": [
            {
                "shot_number": 1,
                # Missing 'first_frame'
                "last_frame": 100,
                "detected_faces": {"face_count": 0},
                "detected_graphics": {}
            }
        ]
    }
    path = tmp_path / "invalid_missing_field.json"
    path.write_text(json.dumps(bad_data))

    errors = validate_shot_features_json(str(path), SCHEMA_PATH)
    assert len(errors) == 1
    assert "first_frame" in errors[0]

def test_non_contiguous_shots_returns_error(tmp_path):
    bad_data = {
        "shots": [
            {
                "shot_number": 1,
                "first_frame": 0,
                "last_frame": 100,
                "detected_faces": {"face_count": 0},
                "detected_graphics": {}
            },
            {
                "shot_number": 2,
                "first_frame": 102,  # gap here
                "last_frame": 200,
                "detected_faces": {"face_count": 0},
                "detected_graphics": {}
            }
        ]
    }
    path = tmp_path / "invalid_non_contiguous.json"
    path.write_text(json.dumps(bad_data))

    errors = validate_shot_features_json(str(path), SCHEMA_PATH, total_frame_count=201)
    assert len(errors) == 1
    assert "contiguity" in errors[0]

def test_missing_detected_graphics_field_allowed(tmp_path):
    valid_data = {
        "shots": [
            {
                "shot_number": 1,
                "first_frame": 0,
                "last_frame": 100,
                "detected_faces": {
                    "face_count": 0
                }
                # intentionally omitting 'detected_graphics'
            }
        ]
    }
    path = tmp_path / "valid_missing_detected_graphics.json"
    with open(path, "w") as f:
        json.dump(valid_data, f)
    
    # Should not raise any exceptions
    errors = validate_shot_features_json(str(path), SCHEMA_PATH, total_frame_count=101)
    assert errors == []

def test_detected_graphics_must_be_empty_object_returns_error(tmp_path):
    bad_data = {
        "shots": [
            {
                "shot_number": 1,
                "first_frame": 0,
                "last_frame": 100,
                "detected_faces": {"face_count": 0},
                "detected_graphics": {"graphic_count": 1}  # invalid
            }
        ]
    }
    path = tmp_path / "invalid_nonempty_graphics.json"
    path.write_text(json.dumps(bad_data))

    errors = validate_shot_features_json(str(path), SCHEMA_PATH)
    assert len(errors) == 1
    assert "detected_graphics" in errors[0]

