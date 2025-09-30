from pathlib import Path
import json
import numpy as np

# Ensure helpers import works from the tests directory
import sys, os
sys.path.append(os.path.dirname(__file__))
from helpers_v2 import Obs, Track, make_512  # noqa: E402

from facekit.output.json_v2 import (
    V2WriterConfig,
    build_v2_manifest_from_tracks,
    EmbeddingCollector,
)
from facekit.validation import validate_manifest


def _expected_total(m):
    return m["shots"][-1]["last_frame"] + 1


def test_inline_embeddings_present_and_valid(tmp_path: Path):
    emb = make_512(3)
    t = Track(
        1, 1, gid=0,
        obs=[
            Obs(10, (0, 0, 10, 10), embedding=emb[0]),
            Obs(11, (1, 1, 11, 11), embedding=emb[1]),
            Obs(12, (2, 2, 12, 12), embedding=emb[2]),
        ],
    )

    cfg = V2WriterConfig(
        video_path="x.mp4",
        video_size=(100, 100),
        fps=30.0,
        total_frames=1000,
        emb_store="inline",
    )

    # Inline does not need a collector explicitly, but passing one is fine too
    collector = EmbeddingCollector("inline", dim=512)
    m = build_v2_manifest_from_tracks([t], cfg, face_metadata=[], collector=collector)

    # Observe inline payloads
    obs = m["shots"][0]["face_tracks"][0]["obs"]
    assert all("embedding" in o for o in obs)
    assert all("emb_idx" not in o for o in obs)
    assert len(obs[0]["embedding"]) == 512

    # Validate
    p = tmp_path / "inline.json"
    p.write_text(json.dumps(m, indent=2))
    errs = validate_manifest(p, total_frame_count=_expected_total(m))
    assert not errs, f"Unexpected validation errors: {errs}"


def test_sidecar_indices_and_finalize_roundtrip(tmp_path: Path):
    emb = make_512(4)
    t = Track(
        1, 1, gid=0,
        obs=[
            Obs(5, (0, 0, 10, 10), embedding=emb[0]),
            Obs(6, (1, 1, 11, 11), embedding=emb[1]),
            Obs(7, (2, 2, 12, 12), embedding=emb[2]),
            Obs(8, (3, 3, 13, 13), embedding=emb[3]),
        ],
    )

    cfg = V2WriterConfig(
        video_path="x.mp4",
        video_size=(1920, 1080),
        fps=29.97,
        total_frames=1000,
        emb_store="sidecar",
        emb_sidecar_path=None,  # we'll pick a temp below
    )

    collector = EmbeddingCollector("sidecar", dim=512)
    m = build_v2_manifest_from_tracks([t], cfg, face_metadata=[], collector=collector)

    # obs carry indices
    obs = m["shots"][0]["face_tracks"][0]["obs"]
    idxs = [o.get("emb_idx") for o in obs]
    assert all(isinstance(i, int) for i in idxs)
    assert idxs == list(range(len(obs)))  # 0..N-1

    # finalize to a tmp npz
    sidecar = tmp_path / "embeddings_sidecar.npz"
    desc = collector.finalize_sidecar(sidecar)
    assert Path(desc["path"]).exists()
    assert desc["format"] == "npz"
    assert desc["dim"] == 512
    assert desc["count"] == 4

    loaded = np.load(desc["path"])["embeddings"]
    assert loaded.shape == (4, 512)
    # same order as encountered
    assert np.allclose(loaded, emb, atol=1e-6)

    # Attach sidecar descriptor (required when emb_store == "sidecar")
    m["embedding_sidecar"] = desc

    # Validate manifest
    p = tmp_path / "sidecar.json"
    p.write_text(json.dumps(m, indent=2))
    errs = validate_manifest(p, total_frame_count=_expected_total(m))
    assert not errs, f"Unexpected validation errors: {errs}"


def test_none_mode_skips_embeddings(tmp_path: Path):
    emb = make_512(2)
    t = Track(
        1, 1, gid=0,
        obs=[
            Obs(100, (0, 0, 10, 10), embedding=emb[0]),
            Obs(101, (1, 1, 11, 11), embedding=emb[1]),
        ],
    )

    cfg = V2WriterConfig(
        video_path="x.mp4",
        video_size=(640, 480),
        fps=24.0,
        total_frames=200,
        emb_store=None,               # <— do not serialize embeddings
    )

    m = build_v2_manifest_from_tracks([t], cfg, face_metadata=[])

    obs = m["shots"][0]["face_tracks"][0]["obs"]
    assert all("embedding" not in o and "emb_idx" not in o for o in obs)
    assert "embedding_sidecar" not in m

    # Validate
    p = tmp_path / "none.json"
    p.write_text(json.dumps(m, indent=2))
    errs = validate_manifest(p, total_frame_count=_expected_total(m))
    assert not errs, f"Unexpected validation errors: {errs}"


def test_sidecar_suffix_rules(tmp_path: Path):
    """finalize_sidecar should honor .npy vs default to .npz."""
    emb = make_512(1)
    t = Track(1, 1, gid=0, obs=[Obs(0, (0, 0, 10, 10), embedding=emb[0])])

    cfg = V2WriterConfig(video_path="x.mp4", video_size=(1, 1), total_frames=1, emb_store="sidecar")
    c = EmbeddingCollector("sidecar", dim=512)
    _ = build_v2_manifest_from_tracks([t], cfg, face_metadata=[], collector=c)

    # ask for .npy explicitly
    npy_path = tmp_path / "embs.npy"
    d_npy = c.finalize_sidecar(npy_path)
    assert d_npy["format"] == "npy"
    arr_npy = np.load(d_npy["path"])
    assert arr_npy.shape == (1, 512)

    # ask for no suffix → should force .npz
    npz_path = tmp_path / "embs"
    d_npz = c.finalize_sidecar(npz_path)
    assert d_npz["format"] == "npz"
    z = np.load(d_npz["path"])
    assert "embeddings" in z
    assert z["embeddings"].shape == (1, 512)
