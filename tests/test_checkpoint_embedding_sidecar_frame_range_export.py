from pathlib import Path

import numpy as np


def test_frame_range_export_compacts_embeddings_and_remaps_emb_idx(tmp_path: Path):
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
            # outside requested range, references embedding 0
            (24, 1, 0, [24, 0, 34, 10], 0, 0.99, 0),

            # inside requested range, references embeddings 2 and 4
            (25, 1, 0, [25, 0, 35, 10], 0, 0.99, 2),
            (30, 1, 0, [30, 0, 40, 10], 0, 0.99, 4),

            # inside requested range, no embedding
            (35, 1, 0, [35, 0, 45, 10], 0, 0.99, -1),

            # outside requested range, references embedding 5
            (36, 1, 0, [36, 0, 46, 10], 0, 0.99, 5),
        ],
        dtype=obs_dtype,
    )

    embeddings = np.array(
        [
            np.full((4,), 0.0, dtype=np.float32),
            np.full((4,), 1.0, dtype=np.float32),
            np.full((4,), 2.0, dtype=np.float32),
            np.full((4,), 3.0, dtype=np.float32),
            np.full((4,), 4.0, dtype=np.float32),
            np.full((4,), 5.0, dtype=np.float32),
        ],
        dtype=np.float32,
    )

    ckpt.ckpt_dir.mkdir(parents=True, exist_ok=True)
    np.savez(ckpt.ckpt_dir / "obs_ckpt.npz", observations=obs)
    np.savez(ckpt.ckpt_dir / "emb_ckpt.npz", embeddings=embeddings)

    final_obs = tmp_path / "filtered_obs.npz"
    final_emb = tmp_path / "filtered_emb.npz"

    ckpt.copy_ckpt_sidecars_to_final(
        obs_sidecar_path=str(final_obs),
        emb_sidecar_path=str(final_emb),
        requested_start_frame=25,
        requested_end_frame=35,
    )

    with np.load(final_obs, allow_pickle=False) as data:
        out_obs = data["observations"]

    with np.load(final_emb, allow_pickle=False) as data:
        out_emb = data["embeddings"]

    assert out_obs["f"].tolist() == [25, 30, 35]

    # Original emb_idx values 2 and 4 should be compacted to 0 and 1.
    assert out_obs["emb_idx"].tolist() == [0, 1, -1]

    assert out_emb.shape == (2, 4)
    np.testing.assert_array_equal(out_emb[0], embeddings[2])
    np.testing.assert_array_equal(out_emb[1], embeddings[4])

def test_frame_range_export_discards_unreferenced_embeddings(tmp_path: Path):
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
            # Inside requested range, two observations share embedding 2.
            (25, 1, 0, [25, 0, 35, 10], 0, 0.99, 2),
            (30, 1, 0, [30, 0, 40, 10], 0, 0.99, 2),

            # Inside requested range, references embedding 5.
            (35, 1, 0, [35, 0, 45, 10], 0, 0.99, 5),
        ],
        dtype=obs_dtype,
    )

    embeddings = np.array(
        [
            np.full((4,), 0.0, dtype=np.float32),
            np.full((4,), 1.0, dtype=np.float32),
            np.full((4,), 2.0, dtype=np.float32),
            np.full((4,), 3.0, dtype=np.float32),
            np.full((4,), 4.0, dtype=np.float32),
            np.full((4,), 5.0, dtype=np.float32),
        ],
        dtype=np.float32,
    )

    ckpt.ckpt_dir.mkdir(parents=True, exist_ok=True)
    np.savez(ckpt.ckpt_dir / "obs_ckpt.npz", observations=obs)
    np.savez(ckpt.ckpt_dir / "emb_ckpt.npz", embeddings=embeddings)

    final_obs = tmp_path / "filtered_obs.npz"
    final_emb = tmp_path / "filtered_emb.npz"

    ckpt.copy_ckpt_sidecars_to_final(
        obs_sidecar_path=str(final_obs),
        emb_sidecar_path=str(final_emb),
        requested_start_frame=25,
        requested_end_frame=35,
    )

    with np.load(final_obs, allow_pickle=False) as data:
        out_obs = data["observations"]

    with np.load(final_emb, allow_pickle=False) as data:
        out_emb = data["embeddings"]

    assert out_obs["f"].tolist() == [25, 30, 35]

    # Original embedding indices 2 and 5 should compact to 0 and 1.
    # Both observations that shared original embedding 2 should still share it.
    assert out_obs["emb_idx"].tolist() == [0, 0, 1]

    assert out_emb.shape == (2, 4)
    np.testing.assert_array_equal(out_emb[0], embeddings[2])
    np.testing.assert_array_equal(out_emb[1], embeddings[5])

def test_frame_range_export_preserves_embedding_lookup_semantics(tmp_path: Path):
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
            (20, 1, 0, [20, 0, 30, 10], 0, 0.99, 0),
            (25, 1, 0, [25, 0, 35, 10], 0, 0.99, 4),
            (30, 1, 1, [30, 0, 40, 10], 0, 0.99, 2),
            (35, 1, 0, [35, 0, 45, 10], 0, 0.99, -1),
            (40, 1, 0, [40, 0, 50, 10], 0, 0.99, 5),
        ],
        dtype=obs_dtype,
    )

    embeddings = np.array(
        [
            np.full((4,), 0.0, dtype=np.float32),
            np.full((4,), 1.0, dtype=np.float32),
            np.full((4,), 2.0, dtype=np.float32),
            np.full((4,), 3.0, dtype=np.float32),
            np.full((4,), 4.0, dtype=np.float32),
            np.full((4,), 5.0, dtype=np.float32),
        ],
        dtype=np.float32,
    )

    ckpt.ckpt_dir.mkdir(parents=True, exist_ok=True)
    np.savez(ckpt.ckpt_dir / "obs_ckpt.npz", observations=obs)
    np.savez(ckpt.ckpt_dir / "emb_ckpt.npz", embeddings=embeddings)

    final_obs = tmp_path / "filtered_obs.npz"
    final_emb = tmp_path / "filtered_emb.npz"

    ckpt.copy_ckpt_sidecars_to_final(
        obs_sidecar_path=str(final_obs),
        emb_sidecar_path=str(final_emb),
        requested_start_frame=25,
        requested_end_frame=35,
    )

    with np.load(final_obs, allow_pickle=False) as data:
        out_obs = data["observations"]

    with np.load(final_emb, allow_pickle=False) as data:
        out_emb = data["embeddings"]

    assert out_obs["f"].tolist() == [25, 30, 35]

    original_by_frame = {
        int(row["f"]): embeddings[int(row["emb_idx"])]
        for row in obs
        if int(row["emb_idx"]) >= 0
    }

    for row in out_obs:
        exported_idx = int(row["emb_idx"])
        if exported_idx < 0:
            continue

        frame = int(row["f"])

        np.testing.assert_array_equal(
            out_emb[exported_idx],
            original_by_frame[frame],
        )

def test_frame_range_export_with_no_observations_writes_empty_sidecars(tmp_path: Path):
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
            (10, 1, 0, [10, 0, 20, 10], 0, 0.99, 0),
            (20, 1, 0, [20, 0, 30, 10], 0, 0.99, 1),
        ],
        dtype=obs_dtype,
    )

    embeddings = np.array(
        [
            np.full((4,), 0.0, dtype=np.float32),
            np.full((4,), 1.0, dtype=np.float32),
        ],
        dtype=np.float32,
    )

    ckpt.ckpt_dir.mkdir(parents=True, exist_ok=True)
    np.savez(ckpt.ckpt_dir / "obs_ckpt.npz", observations=obs)
    np.savez(ckpt.ckpt_dir / "emb_ckpt.npz", embeddings=embeddings)

    final_obs = tmp_path / "filtered_obs.npz"
    final_emb = tmp_path / "filtered_emb.npz"

    ckpt.copy_ckpt_sidecars_to_final(
        obs_sidecar_path=str(final_obs),
        emb_sidecar_path=str(final_emb),
        requested_start_frame=30,
        requested_end_frame=40,
    )

    with np.load(final_obs, allow_pickle=False) as data:
        out_obs = data["observations"]

    with np.load(final_emb, allow_pickle=False) as data:
        out_emb = data["embeddings"]

    assert out_obs.shape == (0,)
    assert out_obs.dtype == obs_dtype

    assert out_emb.shape == (0, 4)
    assert out_emb.dtype == np.float32


def test_user_embedding_sidecar_reflects_requested_frame_range(tmp_path: Path):
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
            # Outside requested range.
            (20, 1, 0, [20, 0, 30, 10], 0, 0.99, 0),

            # Inside requested range.
            (25, 1, 0, [25, 0, 35, 10], 0, 0.99, 4),
            (30, 1, 1, [30, 0, 40, 10], 0, 0.99, 2),
            (35, 1, 0, [35, 0, 45, 10], 0, 0.99, -1),

            # Outside requested range.
            (40, 1, 0, [40, 0, 50, 10], 0, 0.99, 5),
        ],
        dtype=obs_dtype,
    )

    embeddings = np.array(
        [
            np.full((4,), 0.0, dtype=np.float32),
            np.full((4,), 1.0, dtype=np.float32),
            np.full((4,), 2.0, dtype=np.float32),
            np.full((4,), 3.0, dtype=np.float32),
            np.full((4,), 4.0, dtype=np.float32),
            np.full((4,), 5.0, dtype=np.float32),
        ],
        dtype=np.float32,
    )

    ckpt.ckpt_dir.mkdir(parents=True, exist_ok=True)
    np.savez(ckpt.ckpt_dir / "obs_ckpt.npz", observations=obs)
    np.savez(ckpt.ckpt_dir / "emb_ckpt.npz", embeddings=embeddings)

    final_emb = tmp_path / "user_requested_emb.npz"

    ckpt.copy_ckpt_sidecars_to_final(
        obs_sidecar_path=None,
        emb_sidecar_path=str(final_emb),
        requested_start_frame=25,
        requested_end_frame=35,
    )

    with np.load(final_emb, allow_pickle=False) as data:
        out_emb = data["embeddings"]

    assert out_emb.shape == (2, 4)
    np.testing.assert_array_equal(out_emb[0], embeddings[2])
    np.testing.assert_array_equal(out_emb[1], embeddings[4])

def test_user_embedding_sidecar_empty_when_range_has_no_embeddings(tmp_path: Path):
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
            # Outside requested range, has embeddings.
            (20, 1, 0, [20, 0, 30, 10], 0, 0.99, 0),
            (40, 1, 0, [40, 0, 50, 10], 0, 0.99, 1),

            # Inside requested range, no embeddings.
            (25, 1, 0, [25, 0, 35, 10], 0, 0.99, -1),
            (30, 1, 0, [30, 0, 40, 10], 0, 0.99, -1),
            (35, 1, 0, [35, 0, 45, 10], 0, 0.99, -1),
        ],
        dtype=obs_dtype,
    )

    embeddings = np.array(
        [
            np.full((4,), 0.0, dtype=np.float32),
            np.full((4,), 1.0, dtype=np.float32),
        ],
        dtype=np.float32,
    )

    ckpt.ckpt_dir.mkdir(parents=True, exist_ok=True)
    np.savez(ckpt.ckpt_dir / "obs_ckpt.npz", observations=obs)
    np.savez(ckpt.ckpt_dir / "emb_ckpt.npz", embeddings=embeddings)

    final_emb = tmp_path / "user_requested_emb.npz"

    ckpt.copy_ckpt_sidecars_to_final(
        obs_sidecar_path=None,
        emb_sidecar_path=str(final_emb),
        requested_start_frame=25,
        requested_end_frame=35,
    )

    with np.load(final_emb, allow_pickle=False) as data:
        out_emb = data["embeddings"]

    assert out_emb.shape == (0, 4)
    assert out_emb.dtype == np.float32