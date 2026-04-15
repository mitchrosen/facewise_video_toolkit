from types import SimpleNamespace

from facekit.pipeline.track_across_segments import _init_shot_aggregator

def test_resume_starts_after_embedding_safe_frame():
    resume_plan = SimpleNamespace(
        anchor_frame=152,
        is_resume=True,
        first_processed_shot_number=2,
        prior_tracks_anchor=[],
        trackid_seed_by_shot={},
        segment_id_seed_by_shot={},
        reuse_tid_for_first_shot=None,
    )

    start_at, _, _ = _init_shot_aggregator(
        shot_idx=0,
        shot_number=2,
        execution_start_frame=153,
        execution_end_frame=299,
        detect_interval=8,
        resume_plan=resume_plan,
        iou_thresh=0.2,
        embedding_thresh=0.7,
        checkpoint=None,
    )

    assert start_at == 153

def test_resume_rejects_empty_execution_range_when_anchor_consumes_shot():
    resume_plan = SimpleNamespace(
        anchor_frame=102,
        is_resume=True,
        first_processed_shot_number=1,
        prior_tracks_anchor=[],
        trackid_seed_by_shot={},
        segment_id_seed_by_shot={},
        reuse_tid_for_first_shot=None,
    )

    try:
        _init_shot_aggregator(
            shot_idx=0,
            shot_number=1,
            execution_start_frame=103,
            execution_end_frame=102,
            detect_interval=8,
            resume_plan=resume_plan,
            iou_thresh=0.2,
            embedding_thresh=0.7,
            checkpoint=None,
        )
        assert False, "expected empty-work-range failure"
    except Exception as e:
        assert "empty execution range" in str(e)
