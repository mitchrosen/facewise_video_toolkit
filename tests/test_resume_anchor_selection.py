from facekit.pipeline.resume_rehydrate import _select_embedding_safe_anchor


def test_select_embedding_safe_anchor_legacy_uses_final_safe_frame():
    assert _select_embedding_safe_anchor(
        [5, 12, 19, 40],
        requested_start_frame=None,
    ) == 40


def test_select_embedding_safe_anchor_explicit_start_uses_latest_prior_safe_frame():
    assert _select_embedding_safe_anchor(
        [5, 12, 19, 40],
        requested_start_frame=25,
    ) == 19


def test_select_embedding_safe_anchor_explicit_start_ignores_safe_frame_equal_to_start():
    assert _select_embedding_safe_anchor(
        [5, 12, 25, 40],
        requested_start_frame=25,
    ) == 12


def test_select_embedding_safe_anchor_explicit_start_without_prior_safe_frame_returns_zero():
    assert _select_embedding_safe_anchor(
        [25, 40],
        requested_start_frame=25,
    ) == 0

def test_resolve_anchor_uses_frame_component_of_checkpoint_anchor_tuple():
    """
    Current checkpoint contract:

    checkpoint.get_resume_anchor() returns:
        (frame_idx, shot_number, shot_first_frame)

    _resolve_anchor() should use only the frame component as the
    embedding-safe resume boundary.
    """

    from facekit.pipeline.resume_rehydrate import _resolve_anchor

    class _FakeCheckpoint:
        def get_resume_anchor(self):
            return (15, 2, 103)

    checkpoint = _FakeCheckpoint()

    result = _resolve_anchor(
        checkpoint,
        resume_enabled=True,
        requested_start_frame=None,
    )

    assert result == 15

def test_resolve_anchor_uses_frozen_checkpoint_anchor_frame():
    """
    Current implementation limitation:

    The checkpoint API exposes only a single frozen embedding-safe
    anchor, not historical safe-frame history.

    Therefore requested_start_frame does not yet influence
    anchor selection.
    """

    from facekit.pipeline.resume_rehydrate import _resolve_anchor

    class _FakeCheckpoint:
        def get_resume_anchor(self):
            return (40, 2, 103)

    checkpoint = _FakeCheckpoint()

    result = _resolve_anchor(
        checkpoint,
        resume_enabled=True,
        requested_start_frame=25,
    )

    # Current behavior:
    # still uses the checkpoint's single frozen safe boundary.
    assert result == 40

def test_resolve_anchor_selects_latest_prior_safe_frame_when_checkpoint_exposes_history():
    """
    When the checkpoint can expose historical embedding-safe frames,
    explicit requested_start_frame should select the latest safe frame
    strictly before the requested start.
    """

    from facekit.pipeline.resume_rehydrate import _resolve_anchor

    class _FakeCheckpoint:
        def get_embedding_safe_frames(self):
            return [5, 12, 19, 40]

        def get_resume_anchor(self):
            # Existing frozen-anchor API. This should not be used when the
            # richer historical safe-frame API is available and start is explicit.
            return (40, 2, 103)

    result = _resolve_anchor(
        _FakeCheckpoint(),
        resume_enabled=True,
        requested_start_frame=25,
    )

    assert result == 19

from pathlib import Path


def test_resolve_anchor_uses_checkpoint_embedding_safe_frame_history(tmp_path: Path):
    from facekit.pipeline.checkpoint import CheckpointManager
    from facekit.pipeline.resume_rehydrate import _resolve_anchor

    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake video")

    ckpt = CheckpointManager(
        tmp_path / "run",
        video_path=str(video_path),
    )

    for frame_idx in [5, 12, 19, 40]:
        ckpt.mark_embedding_safe(
            frame_idx=frame_idx,
            shot_number=1,
            shot_first_frame=0,
            open_tracks=[],
            note="test",
        )

    assert _resolve_anchor(
        ckpt,
        resume_enabled=True,
        requested_start_frame=25,
    ) == 19