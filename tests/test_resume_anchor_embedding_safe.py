from pathlib import Path
import json

from facekit.pipeline.checkpoint import CheckpointManager

def _write_run(parent: Path, run_id: str, status: dict) -> Path:
    run_dir = parent / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "status.json").write_text(json.dumps(status))
    (run_dir / "ckpt").mkdir(exist_ok=True)
    return run_dir

def test_resume_anchor_uses_last_embedding_safe_frame_when_present(tmp_path: Path):
    ckpt_parent = tmp_path / "ckpt_parent"
    ckpt_parent.mkdir()

    status = {
        "run_id": "run-000001",
        "last_detection_frame": 180,
        "last_embedding_safe_frame": 170,
        "last_embedding_safe_shot_number": 2,
        "last_embedding_safe_shot_first_frame": 103,
        "obs_rows_at_last_embedding_safe": 123,
        "emb_rows_at_last_embedding_safe": 456,
    }
    _write_run(ckpt_parent, "run-000001", status)

    dummy_video = tmp_path / "dummy.mp4"
    dummy_video.touch()

    mgr = CheckpointManager.open(
        parent_dir=ckpt_parent,
        video_path=dummy_video,
        options_snapshot={"test": True},
        no_resume=False,
        resume_latest=True,
        force_new_run=False,
        write_disabled=True,
    )

    anchor_frame, anchor_shot, anchor_shot_first = mgr.get_resume_anchor()
    assert anchor_frame == 170
    assert anchor_shot == 2
    assert anchor_shot_first == 103

def test_resume_anchor_is_none_when_no_embedding_safe_anchor_present(tmp_path: Path):
    ckpt_parent = tmp_path / "ckpt_parent"
    ckpt_parent.mkdir()

    status = {
        "run_id": "run-000002",
        "last_detection_frame": 180,
    }
    _write_run(ckpt_parent, "run-000002", status)

    dummy_video = tmp_path / "dummy.mp4"
    dummy_video.touch()

    mgr = CheckpointManager.open(
        parent_dir=ckpt_parent,
        video_path=dummy_video,
        options_snapshot={"test": True},
        no_resume=False,
        resume_latest=True,
        force_new_run=False,
        write_disabled=True,
    )

    assert mgr.get_resume_anchor() is None