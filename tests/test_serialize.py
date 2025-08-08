import pytest
import tempfile
import json
from pathlib import Path
import tempfile

from facekit.tracking.serialize import (
    save_tracks_to_json_file,
    tracks_to_json_dict,
    load_tracks_from_json_file,
    load_tracks_from_json_dict,
)
from facekit.tracking.face_structures import FaceTrack, FaceObservation

def create_dummy_tracks():
    return [
        FaceTrack(shot_id=0,track_id=0, observations=[
            FaceObservation(frame_idx=5, bbox=(10, 10, 20, 20), confidence=0.9),
            FaceObservation(frame_idx=6, bbox=(12, 12, 22, 22), confidence=0.91),
        ]),
        FaceTrack(shot_id=1, track_id=1, observations=[
            FaceObservation(frame_idx=5, bbox=(100, 100, 110, 110), confidence=0.85),
        ])
    ]

def test_save_and_load_tracks_dict_roundtrip():
    original_tracks = create_dummy_tracks()
    json_dict = tracks_to_json_dict(original_tracks)
    loaded_tracks = load_tracks_from_json_dict(json_dict)

    assert len(loaded_tracks) == len(original_tracks)
    for o, l in zip(original_tracks, loaded_tracks):
        assert o.track_id == l.track_id
        assert len(o.observations) == len(l.observations)
        for o1, o2 in zip(o.observations, l.observations):
            assert o1.frame_idx == o2.frame_idx
            assert o1.bbox == o2.bbox
            assert o1.confidence == o2.confidence

def test_save_and_load_tracks_file_roundtrip():
    original_tracks = create_dummy_tracks()
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "tracks.json"
        save_tracks_to_json_file(original_tracks, json_path)
        assert json_path.exists()

        loaded_tracks = load_tracks_from_json_file(json_path)

    assert len(loaded_tracks) == len(original_tracks)
    for o, l in zip(original_tracks, loaded_tracks):
        assert o.track_id == l.track_id
        assert len(o.observations) == len(l.observations)
        for o1, o2 in zip(o.observations, l.observations):
            assert o1.frame_idx == o2.frame_idx
            assert o1.bbox == o2.bbox
            assert o1.confidence == o2.confidence

def test_load_tracks_from_malformed_dict():
    # Missing bbox and confidence in the observation
    bad_json_dict = {
        "tracks": [
            {
                "track_id": 0,
                "observations": [{"frame_idx": 1}]
            }
        ]
    }
    with pytest.raises(KeyError):
        load_tracks_from_json_dict(bad_json_dict)

def test_load_tracks_from_malformed_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        bad_path = Path(tmpdir) / "bad_tracks.json"
        bad_data = {
            "tracks": [
                {
                    "track_id": 1,
                    "observations": [{"bbox": [1, 2, 3, 4]}]  # Missing frame_idx and confidence
                }
            ]
        }
        bad_path.write_text(json.dumps(bad_data, indent=2))
        with pytest.raises(KeyError):
            load_tracks_from_json_file(bad_path)