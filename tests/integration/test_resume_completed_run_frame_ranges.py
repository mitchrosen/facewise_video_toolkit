from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

def _with_repo_env() -> dict:
    repo_root = Path(__file__).resolve().parents[2]
    existing = os.environ.get("PYTHONPATH", "")
    extras = [str(repo_root)] + [p for p in sys.path if p]
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        os.pathsep.join([*extras, existing])
        if existing
        else os.pathsep.join(extras)
    )
    return env

def _make_tiny_video(path: Path, frames=60, size=(192, 108)):
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, 30.0, size)
    blank = np.zeros((size[1], size[0], 3), dtype=np.uint8)

    for _ in range(frames):
        vw.write(blank)

    vw.release()

def _write_shots_json(path: Path, first: int, last: int):
    path.write_text(
        json.dumps(
            {
                "shots": [
                    {
                        "shot_number": 1,
                        "first_frame": int(first),
                        "last_frame": int(last),
                    }
                ]
            }
        )
    )

def _run_python(shim: Path, *args, ok=(0,), env=None, cwd=None):
    cp = subprocess.run(
        [sys.executable, str(shim), *args],
        text=True,
        capture_output=True,
        env=env,
        cwd=cwd or shim.parent,
    )

    if cp.returncode not in ok:
        raise AssertionError(
            f"Return code {cp.returncode} not in {ok}\n"
            f"=== STDOUT ===\n{cp.stdout}\n"
            f"=== STDERR ===\n{cp.stderr}\n"
        )

    return cp

def _load_rows(npz_path: Path) -> np.ndarray:
    with np.load(npz_path) as data:
        assert "observations" in data.files

        obs = data["observations"]

        if obs.size == 0:
            return np.empty((0, 8))

        bbox = obs["bbox_xyxy"].astype(np.float32)

        return np.stack(
            [
                obs["shot"].astype(int),
                obs["track_id"].astype(int),
                obs["f"].astype(int),
                bbox[:, 0],
                bbox[:, 1],
                bbox[:, 2],
                bbox[:, 3],
                obs["src"].astype(int),
            ],
            axis=1,
        )
    
def _load_embeddings(npz_path: Path) -> np.ndarray:
    with np.load(npz_path, allow_pickle=False) as data:
        assert "embeddings" in data.files
        return data["embeddings"]

SHIM_SOURCE = r"""
import os, sys, numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

def _dummy_loader(*a, **k):
    return object()

import facekit.detection.yolo5face_model as y5
y5.load_yolo5face_model = _dummy_loader

class _DummyEmbedder:
    def __init__(self, *a, **k):
        pass

    def get_embedding_batch(self, chips, batch_size=None, **kwargs):
        vecs = []

        for chip in chips:
            h = int(np.uint64(
                chip.sum()
                + chip.shape[0] * 1009
                + chip.shape[1] * 2741
            ))

            rng = np.random.RandomState(h % (2**32))
            v = rng.rand(512).astype(np.float32)
            v /= (np.linalg.norm(v) + 1e-12)
            vecs.append(v)

        return np.stack(vecs, axis=0)

    def get_embedding(self, chip, **kwargs):
        return self.get_embedding_batch([chip], **kwargs)[0]

import facekit.embedding.embedder as emb_mod
emb_mod.FaceEmbedder = _DummyEmbedder

from facekit.pipeline import track_across_segments as track_mod
from facekit.cli import resolve_face_ids_v2_cli as cli_mod

class _EmitOneMovingBox:
    def detect_faces_in_frame(self, frame, frame_index=None):
        f = int(frame_index or 0)

        x1 = 10.0 + f
        y1 = 10.0
        x2 = 50.0 + f
        y2 = 50.0

        boxes = np.array([[x1, y1, x2, y2]], np.float32)

        lms = np.array(
            [[
                [x1, y1],
                [x2, y1],
                [x1, y2],
                [x2, y2],
                [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
            ]],
            np.float32,
        )

        conf = np.array([0.99], np.float32)

        return boxes, lms, conf

def _fake_align(frame, landmarks, frame_idx=None, **k):
    return np.zeros((112, 112, 3), np.uint8)

track_mod.align_face_for_arcface = _fake_align

_orig_track = track_mod.track_across_segments

def _wrapped_track(*a, **k):
    k["detector"] = _EmitOneMovingBox()
    return _orig_track(*a, **k)

track_mod.track_across_segments = _wrapped_track

try:
    cli_mod.main()
except SystemExit:
    raise
"""

@pytest.mark.integration
@pytest.mark.parametrize(
    "start_frame,end_frame,expected_min_frame,expected_max_frame",
    [
        pytest.param(25, 35, 25, 35, id="explicit-start-explicit-end"),
        pytest.param(25, None, 25, 59, id="explicit-start-no-end"),
        pytest.param(None, 35, 0, 35, id="no-start-explicit-end"),
        pytest.param(25, 25, 25, 25, id="single-frame-range"),
    ],
)
def test_completed_run_resume_frame_range_matches_original_subset(
    tmp_path: Path,
    start_frame: int | None,
    end_frame: int | None,
    expected_min_frame: int,
    expected_max_frame: int,
):
    shim = tmp_path / "run_cli_with_stubs.py"
    shim.write_text(SHIM_SOURCE)

    total_frames = 60

    vid = tmp_path / "toy.mp4"
    shots_path = tmp_path / "shots.json"

    _make_tiny_video(vid, frames=total_frames, size=(192, 108))
    _write_shots_json(shots_path, 0, total_frames - 1)

    ckpt_parent = tmp_path / "ckpt"
    ckpt_parent.mkdir()

    full_obs = tmp_path / "full_obs.npz"
    rerun_obs = tmp_path / "rerun_obs.npz"

    common = [
        "--input", str(vid),
        "--shot-segmentation", str(shots_path),
        "--schema-version", "2.1",
        "--detect-interval", "1",
        "--embedding-batch-size-max", "16",
        "--embedding-queue-max-pending", "16",
        "--emb-store", "none",
        "--checkpoint-dir", str(ckpt_parent),
        "--log", "INFO",
    ]
    _run_python(
        shim,
        *common,
        "--obs-sidecar-path", str(full_obs),
        "--output-global-json", str(tmp_path / "full.json"),
        ok=(0,),
        cwd=tmp_path,
        env=_with_repo_env(),
    )

    range_args = ["--resume-latest"]

    if start_frame is not None:
        range_args.extend(["--start-frame", str(start_frame)])

    if end_frame is not None:
        range_args.extend(["--end-frame", str(end_frame)])

    _run_python(
        shim,
        *common,
        *range_args,
        "--obs-sidecar-path", str(rerun_obs),
        "--output-global-json", str(tmp_path / "rerun.json"),
        ok=(0,),
        cwd=tmp_path,
        env=_with_repo_env(),
    )

    full_rows = _load_rows(full_obs)
    rerun_rows = _load_rows(rerun_obs)

    assert full_rows.shape[0] > 0
    assert rerun_rows.shape[0] > 0

    expected_rows = full_rows[
        (full_rows[:, 2] >= expected_min_frame)
        & (full_rows[:, 2] <= expected_max_frame)
    ]

    assert expected_rows.shape[0] > 0

    assert rerun_rows.shape == expected_rows.shape
    assert np.array_equal(rerun_rows, expected_rows)

@pytest.mark.integration
def test_completed_run_resume_frame_range_user_embedding_sidecar_reflects_range(
    tmp_path: Path,
):
    shim = tmp_path / "run_cli_with_stubs.py"
    shim.write_text(SHIM_SOURCE)

    total_frames = 60

    vid = tmp_path / "toy.mp4"
    shots_path = tmp_path / "shots.json"

    _make_tiny_video(vid, frames=total_frames, size=(192, 108))
    _write_shots_json(shots_path, 0, total_frames - 1)

    ckpt_parent = tmp_path / "ckpt"
    ckpt_parent.mkdir()

    full_obs = tmp_path / "full_obs.npz"
    full_emb = tmp_path / "full_emb.npz"

    rerun_obs = tmp_path / "rerun_obs.npz"
    rerun_emb = tmp_path / "rerun_emb.npz"

    common = [
        "--input", str(vid),
        "--shot-segmentation", str(shots_path),
        "--schema-version", "2.1",
        "--detect-interval", "1",
        "--embedding-batch-size-max", "16",
        "--embedding-queue-max-pending", "16",
        "--emb-store", "sidecar",
        "--checkpoint-dir", str(ckpt_parent),
        "--log", "INFO",
    ]

    _run_python(
        shim,
        *common,
        "--obs-sidecar-path", str(full_obs),
        "--emb-sidecar-path", str(full_emb),
        "--output-global-json", str(tmp_path / "full.json"),
        ok=(0,),
        cwd=tmp_path,
        env=_with_repo_env(),
    )

    _run_python(
        shim,
        *common,
        "--resume-latest",
        "--start-frame", "25",
        "--end-frame", "35",
        "--obs-sidecar-path", str(rerun_obs),
        "--emb-sidecar-path", str(rerun_emb),
        "--output-global-json", str(tmp_path / "rerun.json"),
        ok=(0,),
        cwd=tmp_path,
        env=_with_repo_env(),
    )

    full_rows = _load_rows(full_obs)
    rerun_rows = _load_rows(rerun_obs)

    expected_rows = full_rows[
        (full_rows[:, 2] >= 25)
        & (full_rows[:, 2] <= 35)
    ]

    assert expected_rows.shape[0] > 0
    assert rerun_rows.shape == expected_rows.shape
    assert np.array_equal(rerun_rows, expected_rows)

    full_embeddings = _load_embeddings(full_emb)
    rerun_embeddings = _load_embeddings(rerun_emb)

    assert full_embeddings.shape[0] > rerun_embeddings.shape[0] > 0

    with np.load(rerun_obs, allow_pickle=False) as data:
        rerun_observations = data["observations"]

    used_emb_idx = sorted({
        int(idx)
        for idx in rerun_observations["emb_idx"]
        if int(idx) >= 0
    })

    assert used_emb_idx == list(range(rerun_embeddings.shape[0]))

@pytest.mark.integration
def test_resume_latest_ignores_tmp_run_dirs_for_frame_range_embedding_sidecar(
    tmp_path: Path,
):
    shim = tmp_path / "run_cli_with_stubs.py"
    shim.write_text(SHIM_SOURCE)

    total_frames = 60

    vid = tmp_path / "toy.mp4"
    shots_path = tmp_path / "shots.json"

    _make_tiny_video(vid, frames=total_frames, size=(192, 108))
    _write_shots_json(shots_path, 0, total_frames - 1)

    ckpt_parent = tmp_path / "ckpt"
    ckpt_parent.mkdir()

    full_obs = tmp_path / "full_obs.npz"
    full_emb = tmp_path / "full_emb.npz"

    range_obs = tmp_path / "range_obs.npz"
    range_emb = tmp_path / "range_emb.npz"

    common = [
        "--input", str(vid),
        "--shot-segmentation", str(shots_path),
        "--schema-version", "2.1",
        "--detect-interval", "1",
        "--embedding-batch-size-max", "16",
        "--embedding-queue-max-pending", "16",
        "--emb-store", "sidecar",
        "--checkpoint-dir", str(ckpt_parent),
        "--log", "INFO",
    ]

    _run_python(
        shim,
        *common,
        "--obs-sidecar-path", str(full_obs),
        "--emb-sidecar-path", str(full_emb),
        "--output-global-json", str(tmp_path / "full.json"),
        ok=(0,),
        cwd=tmp_path,
        env=_with_repo_env(),
    )

    tmp_run = ckpt_parent / ".tmp-run-99999999T999999Z-deadbeef"
    tmp_ckpt = tmp_run / "ckpt"
    tmp_ckpt.mkdir(parents=True)

    trap_embeddings = np.full((3, 512), 999.0, dtype=np.float32)
    np.savez(tmp_ckpt / "emb_ckpt.npz", embeddings=trap_embeddings)

    trap_obs_dtype = np.dtype(
        [
            ("f", "i4"),
            ("shot", "i4"),
            ("track_id", "i4"),
            ("bbox_xyxy", "f4", (4,)),
            ("src", "i4"),
            ("conf", "f4"),
            ("emb_idx", "i4"),
        ]
    )
    trap_obs = np.array(
        [
            (25, 1, 0, [0, 0, 1, 1], 0, 0.99, 0),
            (30, 1, 0, [0, 0, 1, 1], 0, 0.99, 1),
            (35, 1, 0, [0, 0, 1, 1], 0, 0.99, 2),
        ],
        dtype=trap_obs_dtype,
    )
    np.savez(tmp_ckpt / "obs_ckpt.npz", observations=trap_obs)

    (tmp_run / "status.json").write_text(
        json.dumps(
            {
                "schema_version": "2.3",
                "video_path": str(vid.resolve()),
                "last_embedding_safe_frame": 35,
                "last_embedding_safe_shot_number": 1,
                "last_embedding_safe_shot_first_frame": 0,
                "obs_rows_at_last_embedding_safe": 3,
                "emb_rows_at_last_embedding_safe": 3,
                "embedding_safe_frames": [
                    {
                        "frame_idx": 35,
                        "shot_number": 1,
                        "shot_first_frame": 0,
                    }
                ],
            }
        )
    )

    _run_python(
        shim,
        *common,
        "--resume-latest",
        "--start-frame", "25",
        "--end-frame", "35",
        "--obs-sidecar-path", str(range_obs),
        "--emb-sidecar-path", str(range_emb),
        "--output-global-json", str(tmp_path / "range.json"),
        ok=(0,),
        cwd=tmp_path,
        env=_with_repo_env(),
    )

    rows = _load_rows(range_obs)
    embeddings = _load_embeddings(range_emb)

    assert rows.shape[0] > 0
    assert rows[:, 2].min() >= 25
    assert rows[:, 2].max() <= 35

    assert embeddings.shape[0] > 0
    assert not np.any(embeddings == 999.0)

@pytest.mark.integration
def test_resume_latest_missing_checkpoint_sidecars_fails_fast(
    tmp_path: Path,
):
    shim = tmp_path / "run_cli_with_stubs.py"
    shim.write_text(SHIM_SOURCE)

    total_frames = 60

    vid = tmp_path / "toy.mp4"
    shots_path = tmp_path / "shots.json"

    _make_tiny_video(vid, frames=total_frames, size=(192, 108))
    _write_shots_json(shots_path, 0, total_frames - 1)

    ckpt_parent = tmp_path / "ckpt"
    ckpt_parent.mkdir()

    # Older valid-looking published run. If fallback is incorrectly allowed,
    # resume-latest could use this after rejecting the corrupt latest run.
    old_run = ckpt_parent / "run-20000101T000000Z-oldvalid"
    old_ckpt = old_run / "ckpt"
    old_ckpt.mkdir(parents=True)

    obs_dtype = np.dtype(
        [
            ("f", "i4"),
            ("shot", "i4"),
            ("track_id", "i4"),
            ("bbox_xyxy", "f4", (4,)),
            ("src", "i4"),
            ("conf", "f4"),
            ("emb_idx", "i4"),
        ]
    )

    old_obs = np.array(
        [
            (25, 1, 0, [0, 0, 1, 1], 0, 0.99, 0),
        ],
        dtype=obs_dtype,
    )
    old_embeddings = np.full((1, 512), 111.0, dtype=np.float32)

    np.savez(old_ckpt / "obs_ckpt.npz", observations=old_obs)
    np.savez(old_ckpt / "emb_ckpt.npz", embeddings=old_embeddings)

    (old_run / "status.json").write_text(
        json.dumps(
            {
                "schema_version": "2.3",
                "video_path": str(vid.resolve()),
                "last_embedding_safe_frame": 25,
                "last_embedding_safe_shot_number": 1,
                "last_embedding_safe_shot_first_frame": 0,
                "obs_rows_at_last_embedding_safe": 1,
                "emb_rows_at_last_embedding_safe": 1,
                "embedding_safe_frames": [
                    {
                        "frame_idx": 25,
                        "shot_number": 1,
                        "shot_first_frame": 0,
                    }
                ],
            }
        )
    )

    # Newest published run is corrupt: status exists, but required checkpoint
    # sidecars are missing. This must fail fast, not fall back to old_run.
    corrupt_run = ckpt_parent / "run-99999999T999999Z-corrupt"
    corrupt_run.mkdir()

    (corrupt_run / "status.json").write_text(
        json.dumps(
            {
                "schema_version": "2.3",
                "video_path": str(vid.resolve()),
                "last_embedding_safe_frame": 35,
                "last_embedding_safe_shot_number": 1,
                "last_embedding_safe_shot_first_frame": 0,
                "obs_rows_at_last_embedding_safe": 1,
                "emb_rows_at_last_embedding_safe": 1,
                "embedding_safe_frames": [
                    {
                        "frame_idx": 35,
                        "shot_number": 1,
                        "shot_first_frame": 0,
                    }
                ],
            }
        )
    )

    range_obs = tmp_path / "range_obs.npz"
    range_emb = tmp_path / "range_emb.npz"

    cp =_run_python(
        shim,
        "--input", str(vid),
        "--shot-segmentation", str(shots_path),
        "--schema-version", "2.1",
        "--detect-interval", "1",
        "--embedding-batch-size-max", "16",
        "--embedding-queue-max-pending", "16",
        "--emb-store", "sidecar",
        "--checkpoint-dir", str(ckpt_parent),
        "--resume-latest",
        "--start-frame", "25",
        "--end-frame", "35",
        "--obs-sidecar-path", str(range_obs),
        "--emb-sidecar-path", str(range_emb),
        "--output-global-json", str(tmp_path / "range.json"),
        "--log", "INFO",
        ok=(1,2),
        cwd=tmp_path,
        env=_with_repo_env(),
    )

    assert cp.returncode != 0
    assert "inconsistent checkpoint" in cp.stdout
    assert "obs_ckpt.npz is missing" in cp.stdout

    assert not range_obs.exists()
    assert not range_emb.exists()