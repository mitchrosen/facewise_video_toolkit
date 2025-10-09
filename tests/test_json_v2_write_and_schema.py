from pathlib import Path
import json
from unittest import mock
import pytest

# Make sure we can import helpers co-located with this file
import sys, os
sys.path.append(os.path.dirname(__file__))
from helpers_v2 import Obs, Track  # noqa: E402

from facekit.output.json_v2 import (
    V2WriterConfig,
    build_v2_manifest_from_tracks,
    derive_face_metadata,
    build_generation,
)
from facekit.validation import (
    validate_manifest,
    get_schema_path,
    UnknownSchemaVersion,
)


# ------------------------------ happy path ----------------------------------
def test_v2_writer_and_schema(tmp_path: Path):
    W, H = 1000, 500
    t1 = Track(1, 1, gid=0, obs=[Obs(108, (100, 100, 200, 300)), Obs(109, (110, 110, 210, 310))])
    t2 = Track(1, 2, gid=1, obs=[Obs(114, (400, 100, 600, 350), "tracking", 0.8)])

    cfg = V2WriterConfig(
        video_path="/vid.mp4",
        video_size=(W, H),
        fps=29.97,
        total_frames=1000,
    )

    face_meta = derive_face_metadata([t1, t2])
    manifest = build_v2_manifest_from_tracks(
        [t1, t2],
        cfg,
        face_metadata=face_meta,
        generation={"emb_store": "inline"},  # let commit/branch auto-fill
    )

    # basic sanity
    assert manifest["schema_version"] == "2.0"
    assert "generation" in manifest and "branch" in manifest["generation"]
    assert len(manifest["shots"]) == 1
    assert len(manifest["shots"][0]["face_tracks"]) == 2

    # Schema + business-rule validation via dispatcher (auto schema discovery)
    jpath = tmp_path / "out.v2.json"
    jpath.write_text(json.dumps(manifest, indent=2))
    expected_total = manifest["shots"][-1]["last_frame"] + 1
    errs = validate_manifest(jpath, total_frame_count=expected_total)
    assert not errs, f"Unexpected validation errors: {errs}"


# --------------------------- error routes -----------------------------------
def test_get_schema_path_errors_unknown_minor():
    # Assuming 2.1 does not exist in facekit/schemas
    with pytest.raises(UnknownSchemaVersion):
        get_schema_path("2.1")


def test_get_schema_path_errors_unknown_major():
    with pytest.raises(UnknownSchemaVersion):
        get_schema_path("3")


def test_dispatcher_no_validator_for_major(monkeypatch, tmp_path: Path):
    # Create a tiny V2-like manifest file
    manifest = {
        "schema_version": "2.0",
        "video": {"path": "x", "fps": 30, "size": [100, 100], "total_frames": 10},
        "generation": {
            "created_utc": "2025-10-03T00:00:00Z",
            "commit": "abc",
            "branch": "main",
            "params_hash": "sha256:deadbeef",
        },
        "shots": [{"shot_number": 1, "first_frame": 0, "last_frame": 9, "face_tracks": []}],
    }
    p = tmp_path / "m.json"
    p.write_text(json.dumps(manifest))

    # Remove support for major 2 in the registry to simulate "code not implemented"
    from facekit import validation as vmod
    saved = dict(vmod.SUPPORTED_VALIDATORS)
    try:
        vmod.SUPPORTED_VALIDATORS.pop(2, None)
        with pytest.raises(vmod.UnknownSchemaVersion):
            vmod.validate_manifest(p)
    finally:
        vmod.SUPPORTED_VALIDATORS = saved


# --------------------- build_generation auto-fill ---------------------------
def test_build_generation_autofills_commit_branch(monkeypatch):
    # Fake git CLI results
    with mock.patch("facekit.output.json_v2.subprocess.check_output") as chk:
        chk.side_effect = [
            b"deadbee\n",  # commit
            b"feature/x\n",  # branch
        ]
        gen = build_generation({})
    assert gen["commit"] == "deadbee"
    assert gen["branch"] == "feature/x"
    assert gen["params_hash"].startswith("sha256:")
