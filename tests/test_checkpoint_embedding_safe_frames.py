from pathlib import Path


def test_checkpoint_exposes_embedding_safe_frame_history(tmp_path: Path):
    """
    Checkpoint API:
    expose historical embedding-safe frames so _resolve_anchor() can choose
    the latest safe frame strictly before requested_start_frame.
    """
    from facekit.pipeline.checkpoint import CheckpointManager

    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake video")

    ckpt = CheckpointManager(
        tmp_path / "run",
        video_path=str(video_path),
    )

    ckpt.mark_embedding_safe(
        frame_idx=5,
        shot_number=1,
        shot_first_frame=0,
        open_tracks=[],
        note="test",
    )
    ckpt.mark_embedding_safe(
        frame_idx=12,
        shot_number=1,
        shot_first_frame=0,
        open_tracks=[],
        note="test",
    )
    ckpt.mark_embedding_safe(
        frame_idx=19,
        shot_number=1,
        shot_first_frame=0,
        open_tracks=[],
        note="test",
    )

    assert ckpt.get_embedding_safe_frames() == [5, 12, 19]