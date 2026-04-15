from pathlib import Path

import numpy as np


def test_copy_ckpt_sidecars_to_final_filters_observations_by_requested_frame_range(
    tmp_path: Path,
):
    from facekit.pipeline.checkpoint import CheckpointManager

    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake video")

    ckpt = CheckpointManager(
        tmp_path / "run",
        video_path=str(video_path),
    )

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

    obs = np.array(
        [
            (24, 1, 0, [24, 0, 34, 10], 0, 0.99, -1),
            (25, 1, 0, [25, 0, 35, 10], 0, 0.99, -1),
            (30, 1, 0, [30, 0, 40, 10], 0, 0.99, -1),
            (35, 1, 0, [35, 0, 45, 10], 0, 0.99, -1),
            (36, 1, 0, [36, 0, 46, 10], 0, 0.99, -1),
        ],
        dtype=obs_dtype,
    )

    ckpt.ckpt_dir.mkdir(parents=True, exist_ok=True)
    np.savez(ckpt.ckpt_dir / "obs_ckpt.npz", observations=obs)

    # Embeddings are not relevant to this test, but the method may try to copy
    # them if a target path is supplied. We omit emb_sidecar_path.
    final_obs = tmp_path / "filtered_obs.npz"

    ckpt.copy_ckpt_sidecars_to_final(
        obs_sidecar_path=str(final_obs),
        emb_sidecar_path=None,
        requested_start_frame=25,
        requested_end_frame=35,
    )

    with np.load(final_obs, allow_pickle=False) as data:
        out = data["observations"]

    assert out["f"].tolist() == [25, 30, 35]