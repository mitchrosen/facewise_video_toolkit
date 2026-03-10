from __future__ import annotations

from pathlib import Path
import pytest

from facekit.pipeline.checkpoint import CheckpointManager


def test_open_with_write_disabled_and_no_resume_creates_no_dirs(tmp_path: Path) -> None:
    """
    --no-checkpoint-write disables *all* checkpoint write side effects,
        including checkpoint directory creation.
    """
    parent = tmp_path / "checkpoints_parent"
    video = tmp_path / "toy.mp4"
    video.write_bytes(b"")  # exists

    assert not parent.exists()

    # This should become a no-op / no-dir-creation path once we refactor.
    _ = CheckpointManager.open(
        parent_dir=parent,
        video_path=video,
        options_snapshot={"schema_version": "2.1"},
        no_resume=True,  # should not impact write results
        write_disabled=True,
    )

    assert not parent.exists(), "no-checkpoint-write must not create checkpoint directories"