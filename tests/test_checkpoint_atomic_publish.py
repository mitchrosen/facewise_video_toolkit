from __future__ import annotations

import json
from pathlib import Path

import pytest

from facekit.pipeline.checkpoint import CheckpointManager

def test_new_checkpoint_run_is_published_as_run_directory_not_tmp_directory(tmp_path: Path):
    """
    New run initialization should use atomic publish semantics:

    - build under a temporary non-resumable directory
    - publish by renaming to run-*
    - only run-* directories are resume candidates
    """
    parent = tmp_path / "ckpt_parent"
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake video")

    ckpt = CheckpointManager.open(
        parent_dir=parent,
        video_path=video_path,
        options_snapshot={},
        no_resume=False,
        force_new_run=False,
        run_id=None,
        resume_latest=False,
    )

    assert ckpt.root.parent == parent
    assert ckpt.root.name.startswith("run-")
    assert not ckpt.root.name.startswith(".tmp-run-")
    assert ckpt.root.exists()
    assert ckpt.ckpt_dir.exists()

    tmp_runs = list(parent.glob(".tmp-run-*"))
    assert tmp_runs == []

def test_resume_latest_ignores_unpublished_tmp_run_directories(tmp_path: Path):
    """
    Directories named .tmp-run-* are incomplete/unpublished runs.

    --resume-latest must not treat them as resumable candidates.
    """
    parent = tmp_path / "ckpt_parent"
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake video")

    tmp_run = parent / ".tmp-run-20990101T000000Z-deadbeef"
    (tmp_run / "ckpt").mkdir(parents=True)
    (tmp_run / "status.json").write_text(
        json.dumps({"last_embedding_safe_frame": 10}),
        encoding="utf-8",
    )

    ckpt = CheckpointManager.open(
        parent_dir=parent,
        video_path=video_path,
        options_snapshot={},
        no_resume=False,
        force_new_run=False,
        run_id=None,
        resume_latest=True,
    )

    assert ckpt.root != tmp_run
    assert ckpt.root.name.startswith("run-")
    assert ckpt.resume_enabled is False

def test_resume_latest_fails_if_latest_published_run_is_incomplete(tmp_path: Path):
    """
    A published run-* directory is a resume candidate.

    If the latest published run exists but lacks required resume artifacts,
    --resume-latest should fail fast rather than silently falling back to an
    older run or a fresh run.
    """
    parent = tmp_path / "ckpt_parent"
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake video")

    older_complete = parent / "run-20260101T000000Z-complete"
    (older_complete / "ckpt").mkdir(parents=True)
    (older_complete / "status.json").write_text(
        json.dumps({"last_embedding_safe_frame": 10}),
        encoding="utf-8",
    )

    latest_incomplete = parent / "run-20990101T000000Z-incomplete"
    (latest_incomplete / "ckpt").mkdir(parents=True)
    # Intentionally no status.json.

    with pytest.raises(Exception, match="status.json|missing|incomplete|resume"):
        CheckpointManager.open(
            parent_dir=parent,
            video_path=video_path,
            options_snapshot={},
            no_resume=False,
            force_new_run=False,
            run_id=None,
            resume_latest=True,
        )

def test_create_new_run_dir_uses_publish_helper(monkeypatch, tmp_path: Path):
    """
    New run creation should publish a staged temp run directory by calling
    _publish_run_dir(tmp_run_dir, final_run_dir).
    """
    calls = []

    def fake_publish(tmp_run_dir: Path, final_run_dir: Path) -> None:
        calls.append((Path(tmp_run_dir), Path(final_run_dir)))
        Path(final_run_dir).mkdir(parents=True, exist_ok=False)
        Path(final_run_dir, "ckpt").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        CheckpointManager,
        "_publish_run_dir",
        staticmethod(fake_publish),
    )

    parent = tmp_path / "ckpt_parent"
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake video")

    ckpt = CheckpointManager.open(
        parent_dir=parent,
        video_path=video_path,
        options_snapshot={},
        no_resume=False,
        force_new_run=False,
        run_id=None,
        resume_latest=False,
    )

    assert len(calls) == 1

    tmp_run_dir, final_run_dir = calls[0]

    assert tmp_run_dir.parent == parent
    assert final_run_dir.parent == parent

    assert tmp_run_dir.name.startswith(".tmp-run-")
    assert final_run_dir.name.startswith("run-")
    assert tmp_run_dir.name.removeprefix(".tmp-") == final_run_dir.name

    assert ckpt.root == final_run_dir

def test_publish_run_dir_renames_tmp_run_to_final_run(monkeypatch, tmp_path: Path):
    """
    _publish_run_dir should perform a same-parent directory rename from
    .tmp-run-* to run-*.
    """
    calls = []

    def fake_replace(src, dst):
        calls.append((Path(src), Path(dst)))
        Path(dst).mkdir(parents=True, exist_ok=False)

    monkeypatch.setattr("os.replace", fake_replace)

    parent = tmp_path / "ckpt_parent"
    tmp_run_dir = parent / ".tmp-run-20260101T000000Z-deadbeef"
    final_run_dir = parent / "run-20260101T000000Z-deadbeef"

    tmp_run_dir.mkdir(parents=True)

    CheckpointManager._publish_run_dir(tmp_run_dir, final_run_dir)

    assert calls == [(tmp_run_dir, final_run_dir)]

def test_publish_run_dir_rejects_cross_parent_publish(tmp_path: Path):
    """
    Atomic directory rename is only safe when temp and final directories share
    the same parent directory.
    """
    tmp_parent = tmp_path / "tmp_parent"
    final_parent = tmp_path / "final_parent"

    tmp_run_dir = tmp_parent / ".tmp-run-20260101T000000Z-deadbeef"
    final_run_dir = final_parent / "run-20260101T000000Z-deadbeef"

    tmp_run_dir.mkdir(parents=True)
    final_parent.mkdir(parents=True)

    with pytest.raises(ValueError, match="same parent|parent"):
        CheckpointManager._publish_run_dir(tmp_run_dir, final_run_dir)

def test_create_new_run_dir_current_symlink_points_to_published_run(tmp_path: Path):
    from facekit.pipeline.checkpoint import CheckpointManager

    run_dir = CheckpointManager._create_new_run_dir(
        tmp_path,
        {
            "video_path": "video.mp4",
            "detector_model_path": "det.pt",
            "embedding_model_path": "emb.onnx",
            "yolo_config_path": "yolo.yaml",
            "shot_segmentation_path": "shots.json",
            "detect_interval": 1,
        },
    )

    current = tmp_path / "current"

    assert run_dir.name.startswith("run-")
    assert not run_dir.name.startswith(".tmp-")

    assert current.is_symlink()
    assert current.readlink() == Path(run_dir.name)
    assert (tmp_path / current.readlink()).resolve() == run_dir.resolve()