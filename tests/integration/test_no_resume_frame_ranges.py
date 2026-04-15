from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.integration.test_resume_completed_run_frame_ranges import (
    SHIM_SOURCE,
    _load_rows,
    _make_tiny_video,
    _run_python,
    _with_repo_env,
    _write_shots_json,
)

def _load_embeddings(npz_path: Path) -> np.ndarray:
    with np.load(npz_path, allow_pickle=False) as data:
        assert "embeddings" in data.files
        return data["embeddings"]

@pytest.mark.integration
def test_no_resume_frame_range_ignores_existing_checkpoint_for_embedding_sidecar(
    tmp_path: Path,
):
    shim = tmp_path / "run_cli_with_stubs.py"
    shim.write_text(SHIM_SOURCE)

    total_frames = 60

    vid = tmp_path / "toy.mp4"
    shots_path = tmp_path / "shots.json"

    _make_tiny_video(vid, frames=total_frames, size=(192, 108))
    _write_shots_json(shots_path, 0, total_frames - 1)

    polluted_ckpt_parent = tmp_path / "polluted_ckpt"
    clean_ckpt_parent = tmp_path / "clean_ckpt"

    polluted_ckpt_parent.mkdir()
    clean_ckpt_parent.mkdir()

    # Create a fake/latest-looking checkpoint that must not be read when
    # --no-resume is supplied.
    trap_run = polluted_ckpt_parent / "run-99999999T999999Z-deadbeef"
    trap_ckpt = trap_run / "ckpt"
    trap_ckpt.mkdir(parents=True)

    trap_embeddings = np.full((3, 512), 999.0, dtype=np.float32)
    np.savez(trap_ckpt / "emb_ckpt.npz", embeddings=trap_embeddings)

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
    np.savez(trap_ckpt / "obs_ckpt.npz", observations=trap_obs)

    (trap_run / "status.json").write_text(
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

    polluted_obs = tmp_path / "polluted_obs.npz"
    polluted_emb = tmp_path / "polluted_emb.npz"

    clean_obs = tmp_path / "clean_obs.npz"
    clean_emb = tmp_path / "clean_emb.npz"

    common = [
        "--input", str(vid),
        "--shot-segmentation", str(shots_path),
        "--schema-version", "2.1",
        "--detect-interval", "1",
        "--embedding-batch-size-max", "16",
        "--embedding-queue-max-pending", "16",
        "--emb-store", "sidecar",
        "--start-frame", "25",
        "--end-frame", "35",
        "--no-resume",
        "--log", "INFO",
    ]

    _run_python(
        shim,
        *common,
        "--checkpoint-dir", str(polluted_ckpt_parent),
        "--obs-sidecar-path", str(polluted_obs),
        "--emb-sidecar-path", str(polluted_emb),
        "--output-global-json", str(tmp_path / "polluted.json"),
        ok=(0,),
        cwd=tmp_path,
        env=_with_repo_env(),
    )

    _run_python(
        shim,
        *common,
        "--checkpoint-dir", str(clean_ckpt_parent),
        "--obs-sidecar-path", str(clean_obs),
        "--emb-sidecar-path", str(clean_emb),
        "--output-global-json", str(tmp_path / "clean.json"),
        ok=(0,),
        cwd=tmp_path,
        env=_with_repo_env(),
    )

    polluted_rows = _load_rows(polluted_obs)
    clean_rows = _load_rows(clean_obs)

    polluted_embeddings = _load_embeddings(polluted_emb)
    clean_embeddings = _load_embeddings(clean_emb)

    assert polluted_rows.shape[0] > 0
    assert polluted_rows[:, 2].min() >= 25
    assert polluted_rows[:, 2].max() <= 35

    np.testing.assert_array_equal(polluted_rows, clean_rows)
    np.testing.assert_array_equal(polluted_embeddings, clean_embeddings)

    assert not np.any(polluted_embeddings == 999.0)