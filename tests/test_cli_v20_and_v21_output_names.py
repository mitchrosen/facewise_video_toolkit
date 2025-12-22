import pytest
import sys
from pathlib import Path
import json
import types

from facekit.cli import resolve_face_ids_v2_cli as cli
from facekit.common.obs_consts import Source
from facekit.tracking.face_structures import FaceObservation, FaceTrack

@pytest.mark.parametrize("ver,expected_suffix", [("2.0","_v2.json"), ("2.1","_v2_1.json")])
def test_default_output_names(monkeypatch, tmp_path, ver, expected_suffix):
    (tmp_path/"in.mp4").write_bytes(b"\x00")

    class DummyFP:
        def fps(self): return 30.0
        def size(self): return (100,100)
        def total_frames(self): return 2
        def __enter__(self): return self
        def __exit__(self, *a): return False
    class _Det: 
        def detect_faces_in_frame(self, frame, target_size=640): return ([],[],[])
    class _FD: 
        def __init__(self, *a, **k): pass
    class _Emb:
        def __init__(self, *a, **k): pass
        def set_max_batch_size(self, *a, **k): pass
        def get_embedding_batch(self, *a, **k):
            import numpy as np
            return np.zeros((0,512),np.float32)
    class _Obs(FaceObservation):
        def __init__(self, f, shot_id=1, track_id=1, src=Source.DETECTED):
            super().__init__(
                frame_idx=int(f),
                track_id=int(track_id),
                bbox=(0.0, 0.0, 1.0, 1.0),
                embedding=None,
                confidence=None,
                source=src,
                landmarks=None,
            )
            self.shot_id = shot_id
    class _Track:
        shot_id = 1
        track_id = 1
        observations = [_Obs(0), _Obs(1)]
        def first_frame(self): 
            return 0
        def last_frame(self): 
            return 1
    def fake_track_across_segments(**kw):
        obs = [_Obs(0), _Obs(1)]
        t = FaceTrack(shot_id=1, track_id=1, observations=obs, segment_id=0, global_id=0)
        return [t]
    def fake_load_model(*a, **k): return _Det()
    def fake_generate_shot_features_json(**kw):
        Path(kw["output_json_path"]).write_text(
            json.dumps({"shots":[{"shot_number":1,"first_frame":0,"last_frame":1}]})
        )

    monkeypatch.setattr(cli, "track_across_segments", types.SimpleNamespace(track_across_segments=fake_track_across_segments),)
    monkeypatch.setattr(cli, "generate_shot_features_json", fake_generate_shot_features_json)
    monkeypatch.setattr(cli, "ReaderCoordinator", lambda p: DummyFP())
    monkeypatch.setattr(cli, "load_yolo5face_model", fake_load_model)
    monkeypatch.setattr(cli, "FaceDetector", lambda m: _FD())
    monkeypatch.setattr(cli, "FaceEmbedder", lambda *a,**k: _Emb())
    monkeypatch.setattr(cli, "_validate_manifest_dict",
                    lambda manifest, schema_version, total_frame_count, schema_dir: [])

    argv = [
        "prog", "--input", str(tmp_path/"in.mp4"),
        "--schema-version", ver,
        "--emb-store", "none",
        "--output-global-json"
    ]
    monkeypatch.setenv("PYTEST_CLI_BLOCK","1")
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    out = tmp_path / f"in{expected_suffix}"
    assert out.exists()