from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pytest

from facekit.pipeline import track_across_segments as tas

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeFrameProvider:
    def __init__(self, total_frames: int = 100):
        self._total_frames = total_frames
        self.reset_calls: list[int] = []
        self.next_calls = 0
        self.current_frame = 0
        self.frames_returned: list[int] = []

    def fps(self):
        return 30.0

    def size(self):
        return (640, 360)

    def total_frames(self):
        return self._total_frames

    def reset_to_frame(self, frame_idx: int):
        self.reset_calls.append(int(frame_idx))
        self.current_frame = int(frame_idx)

    def next(self):
        if self.current_frame >= self._total_frames:
            return None
        frame_idx = self.current_frame
        self.current_frame += 1
        self.next_calls += 1
        self.frames_returned.append(frame_idx)
        return np.zeros((32, 32, 3), dtype=np.uint8)

class _FakeDetector:
    def detect_faces_in_frame(self, frame):
        return None

class _FakeEmbedder:
    def get_embedding_batch(self, aligned_faces):
        n = len(aligned_faces)
        return np.zeros((n, 512), dtype=np.float32)

class _FakeTrack:
    def __init__(self, shot_id=1, track_id=0, segment_id=None, observations=None):
        self.shot_id = shot_id
        self.track_id = track_id
        self.segment_id = segment_id
        self.observations = observations or []
        self.embeddings = []

    def is_closed(self):
        return True

    def first_frame(self):
        return 0

    def last_frame(self):
        return -1

    def get_last_bbox(self):
        return None

class _FakeAggregator:
    def __init__(self, *args, **kwargs):
        self.tracks = []
        self.next_track_id = kwargs.get("next_tid_seed", 0)

    def finalize_tracks(self):
        return None

    def update_tracks_with_frame(self, frame_idx, observations):
        return None

    def observations_at(self, frame_idx, source=None, require_track_id=False):
        return []

    def attach_embeddings(self, track_id, embs):
        return None

    def resolve_segment_ids(self, segment_id_counter=0, embedding_threshold=0.7):
        return None

class _FakeFaceTracker:
    def __init__(self, tracker_type="CSRT"):
        self.tracker_type = tracker_type

    def update_trackers(self, frame):
        return {}

    def init_trackers(self, frame, boxes_xywh, track_ids):
        return None

class _FakeValidator:
    def __init__(self, params=None):
        self.params = params

    def validate(self, tracked_boxes, frame, frame_idx):
        return True

    def seed_validator(self, boxes_map, frame_idx, frame):
        return None

class _FakeResumePlan:
    def __init__(self, *, is_resume=False, anchor_frame=0, first_processed_shot_number=1):
        self.is_resume = is_resume
        self.anchor_frame = anchor_frame
        self.first_processed_shot_number = first_processed_shot_number
        self.prior_tracks_anchor = []
        self.open_track_ids_anchor = frozenset()
        self.trackid_seed_by_shot = {}
        self.segment_id_seed_by_shot = {}
        self.reuse_tid_for_first_shot = None

def _write_shots_json(path: Path, *, first_frame: int = 0, last_frame: int = 99):
    payload = {
        "shots": [
            {
                "shot_number": 1,
                "first_frame": first_frame,
                "last_frame": last_frame,
            }
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

def _patch_minimal_tracking_stack(
    monkeypatch,
    *,
    resume_plan: _FakeResumePlan | None = None,
):
    monkeypatch.setattr(tas, "ShotFaceTrackAggregator", _FakeAggregator)
    monkeypatch.setattr(tas, "FaceTracker", _FakeFaceTracker)
    monkeypatch.setattr(tas, "TrackerValidator", _FakeValidator)

    monkeypatch.setattr(
        tas,
        "_build_resume_plan",
        lambda shots, checkpoint, resume_enabled, all_tracks, requested_start_frame=None: (
            resume_plan if resume_plan is not None else _FakeResumePlan(is_resume=False, anchor_frame=0),
            shots,
        )
    )

    monkeypatch.setattr(
        tas,
        "_checkpoint_root_dir",
        lambda checkpoint: None,
    )
    monkeypatch.setattr(
        tas,
        "_checkpoint_observations_and_snapshot",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        tas,
        "_finalize_checkpoint_run",
        lambda checkpoint: None,
    )
    monkeypatch.setattr(
        tas,
        "_persist_embeddings_for_track",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        tas,
        "_attach_and_persist_embedded_obs",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        tas,
        "_maybe_enqueue_track_embedding_observations_for_frame",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        tas,
        "bootstrap_runtime_trackers_for_resume_frame",
        lambda **kwargs: False,
    )
    monkeypatch.setattr(
        tas,
        "extend_prev_track_for_overlapping_detection",
        lambda **kwargs: 0,
    )

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_track_across_segments_local_range_uses_requested_start_frame(monkeypatch, tmp_path: Path):
    """
    Local range execution:
    when resume is disabled and start_frame is specified, the frame provider
    should begin exactly at requested start_frame.
    """
    _patch_minimal_tracking_stack(monkeypatch, resume_plan=_FakeResumePlan(is_resume=False, anchor_frame=0))

    shot_json = tmp_path / "shots.json"
    _write_shots_json(shot_json, first_frame=0, last_frame=99)

    fp = _FakeFrameProvider(total_frames=100)
    detector = _FakeDetector()
    embedder = _FakeEmbedder()

    tas.track_across_segments(
        frame_source=fp,
        shot_json_path=str(shot_json),
        detector=detector,
        embedder=embedder,
        start_frame=25,
        end_frame=30,
        resume_enabled=False,
    )

    assert fp.reset_calls == [25]
    assert fp.frames_returned[0] == 25

def test_track_across_segments_stops_at_end_frame(monkeypatch, tmp_path: Path):
    """
    Frame-range semantics:
    when end_frame is specified, processing should stop at that inclusive frame.
    """
    _patch_minimal_tracking_stack(monkeypatch, resume_plan=_FakeResumePlan(is_resume=False, anchor_frame=0))

    shot_json = tmp_path / "shots.json"
    _write_shots_json(shot_json, first_frame=0, last_frame=99)

    fp = _FakeFrameProvider(total_frames=100)
    detector = _FakeDetector()
    embedder = _FakeEmbedder()

    tas.track_across_segments(
        frame_source=fp,
        shot_json_path=str(shot_json),
        detector=detector,
        embedder=embedder,
        start_frame=10,
        end_frame=15,
        resume_enabled=False,
    )

    assert fp.reset_calls == [10]
    assert fp.frames_returned == [10, 11, 12, 13, 14, 15]
    assert fp.next_calls == 6

def test_track_across_segments_resume_explicit_start_uses_anchor_plus_one(
    monkeypatch,
    tmp_path: Path,
):
    """
    History-preserving explicit-start semantics:
    if resume is enabled and an embedding-safe anchor exists before the
    requested start, execution should begin at anchor_frame + 1, not at the
    requested start and not at the shot first frame.
    """
    _patch_minimal_tracking_stack(
        monkeypatch,
        resume_plan=_FakeResumePlan(
            is_resume=True,
            anchor_frame=19,
            first_processed_shot_number=1,
        ),
    )

    shot_json = tmp_path / "shots.json"
    _write_shots_json(shot_json, first_frame=0, last_frame=99)

    fp = _FakeFrameProvider(total_frames=100)
    detector = _FakeDetector()
    embedder = _FakeEmbedder()

    tas.track_across_segments(
        frame_source=fp,
        shot_json_path=str(shot_json),
        detector=detector,
        embedder=embedder,
        start_frame=25,
        end_frame=30,
        resume_enabled=True,
    )

    assert fp.reset_calls == [20]
    assert fp.frames_returned == [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    assert fp.next_calls == 11

def test_track_across_segments_resume_skips_completed_first_shot(
    monkeypatch,
    tmp_path: Path,
):
    """
    If the embedding-safe anchor is the last frame of the first shot,
    track_across_segments() should skip that completed shot and begin with the
    next shot instead of handing an empty execution range to the helper.
    """
    _patch_minimal_tracking_stack(
        monkeypatch,
        resume_plan=_FakeResumePlan(
            is_resume=True,
            anchor_frame=49,
            first_processed_shot_number=1,
        ),
    )

    shot_json = tmp_path / "shots.json"
    payload = {
        "shots": [
            {"shot_number": 1, "first_frame": 0, "last_frame": 49},
            {"shot_number": 2, "first_frame": 50, "last_frame": 99},
        ]
    }
    shot_json.write_text(json.dumps(payload), encoding="utf-8")

    fp = _FakeFrameProvider(total_frames=100)

    tas.track_across_segments(
        frame_source=fp,
        shot_json_path=str(shot_json),
        detector=_FakeDetector(),
        embedder=_FakeEmbedder(),
        start_frame=60,
        end_frame=65,
        resume_enabled=True,
    )

    assert fp.reset_calls == [50]
    assert fp.frames_returned[:6] == [50, 51, 52, 53, 54, 55]

def test_track_across_segments_no_resume_ignores_checkpoint_anchor(
    monkeypatch,
    tmp_path: Path,
):
    """
    Non-history-preserving mode:
    even if a checkpoint anchor exists, resume_enabled=False must use local range execution 
    and begin exactly at requested start.
    """
    _patch_minimal_tracking_stack(
        monkeypatch,
        resume_plan=_FakeResumePlan(is_resume=False, anchor_frame=80),
    )

    shot_json = tmp_path / "shots.json"
    _write_shots_json(shot_json, first_frame=0, last_frame=99)

    fp = _FakeFrameProvider(total_frames=100)

    tas.track_across_segments(
        frame_source=fp,
        shot_json_path=str(shot_json),
        detector=_FakeDetector(),
        embedder=_FakeEmbedder(),
        start_frame=25,
        end_frame=30,
        resume_enabled=False,
    )

    assert fp.reset_calls == [25]
    assert fp.frames_returned == [25, 26, 27, 28, 29, 30]

def test_track_across_segments_legacy_resume_no_requested_start_runs_from_anchor_plus_one(
    monkeypatch,
    tmp_path: Path,
):
    """
    Legacy resume execution:
    when start_frame is None and resume is enabled with a valid embedding-safe
    anchor, execution should begin at anchor + 1.
    """
    _patch_minimal_tracking_stack(
        monkeypatch,
        resume_plan=_FakeResumePlan(
            is_resume=True,
            anchor_frame=19,
            first_processed_shot_number=1,
        ),
    )

    shot_json = tmp_path / "shots.json"
    _write_shots_json(shot_json, first_frame=0, last_frame=99)

    fp = _FakeFrameProvider(total_frames=100)

    tas.track_across_segments(
        frame_source=fp,
        shot_json_path=str(shot_json),
        detector=_FakeDetector(),
        embedder=_FakeEmbedder(),
        start_frame=None,
        end_frame=None,
        resume_enabled=True,
    )

    assert fp.reset_calls == [20]
    assert fp.frames_returned[:6] == [20, 21, 22, 23, 24, 25]

def test_track_across_segments_legacy_resume_requested_end_after_anchor(
    monkeypatch,
    tmp_path: Path,
):
    """
    Legacy resume execution:
    when start_frame is None and end_frame is after the embedding-safe anchor,
    execution should begin at anchor + 1 and stop at requested end.
    """
    _patch_minimal_tracking_stack(
        monkeypatch,
        resume_plan=_FakeResumePlan(
            is_resume=True,
            anchor_frame=19,
            first_processed_shot_number=1,
        ),
    )

    shot_json = tmp_path / "shots.json"
    _write_shots_json(shot_json, first_frame=0, last_frame=99)

    fp = _FakeFrameProvider(total_frames=100)

    tas.track_across_segments(
        frame_source=fp,
        shot_json_path=str(shot_json),
        detector=_FakeDetector(),
        embedder=_FakeEmbedder(),
        start_frame=None,
        end_frame=30,
        resume_enabled=True,
    )

    assert fp.reset_calls == [20]
    assert fp.frames_returned == [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    assert fp.next_calls == 11

def test_track_across_segments_legacy_resume_requested_end_within_safe_history_executes_nothing(
    monkeypatch,
    tmp_path: Path,
):
    """
    History-covered output only:
    when start_frame is None and requested end is already within persisted
    embedding-safe history, no new frames should be executed.
    """
    _patch_minimal_tracking_stack(
        monkeypatch,
        resume_plan=_FakeResumePlan(
            is_resume=True,
            anchor_frame=19,
            first_processed_shot_number=1,
        ),
    )

    shot_json = tmp_path / "shots.json"
    _write_shots_json(shot_json, first_frame=0, last_frame=99)

    fp = _FakeFrameProvider(total_frames=100)

    tas.track_across_segments(
        frame_source=fp,
        shot_json_path=str(shot_json),
        detector=_FakeDetector(),
        embedder=_FakeEmbedder(),
        start_frame=None,
        end_frame=15,
        resume_enabled=True,
    )

    # Final target behavior:
    # no execution should occur because requested output is entirely inside
    # already-safe history.
    assert fp.reset_calls == []
    assert fp.frames_returned == []
    assert fp.next_calls == 0

def test_track_across_segments_history_preserving_explicit_start_executes_from_prior_safe_boundary(
    monkeypatch,
    tmp_path: Path,
):
    """
    History-preserving range execution:
    when start_frame is explicit and a prior embedding-safe frame exists,
    execution begins at anchor + 1 and proceeds only through the
    requested execution range.
    """

    _patch_minimal_tracking_stack(
        monkeypatch,
        resume_plan=_FakeResumePlan(
            is_resume=True,
            anchor_frame=19,
            first_processed_shot_number=1,
        ),
    )

    shot_json = tmp_path / "shots.json"
    _write_shots_json(shot_json, first_frame=0, last_frame=99)

    fp = _FakeFrameProvider(total_frames=100)

    tas.track_across_segments(
        frame_source=fp,
        shot_json_path=str(shot_json),
        detector=_FakeDetector(),
        embedder=_FakeEmbedder(),
        start_frame=25,
        end_frame=30,
        resume_enabled=True,
    )

    assert fp.reset_calls == [20]
    assert fp.frames_returned == [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

    assert fp.next_calls == 11


def test_track_across_segments_history_preserving_explicit_start_without_prior_safe_frame_starts_at_zero(
    monkeypatch,
    tmp_path: Path,
):
    """
    History-preserving range execution fallback:
    when start_frame is explicit but there is no embedding-safe frame prior to
    the requested start, execution should start at frame 0.
    """
    _patch_minimal_tracking_stack(
        monkeypatch,
        resume_plan=_FakeResumePlan(
            is_resume=False,
            anchor_frame=0,
            first_processed_shot_number=1,
        ),
    )

    shot_json = tmp_path / "shots.json"
    _write_shots_json(shot_json, first_frame=0, last_frame=99)

    fp = _FakeFrameProvider(total_frames=100)

    tas.track_across_segments(
        frame_source=fp,
        shot_json_path=str(shot_json),
        detector=_FakeDetector(),
        embedder=_FakeEmbedder(),
        start_frame=25,
        end_frame=30,
        resume_enabled=True,
    )

    assert fp.reset_calls == [0]
    assert fp.frames_returned == list(range(0, 31))
    assert fp.next_calls == 31

def test_track_across_segments_history_preserving_explicit_start_uses_latest_prior_safe_frame(
    monkeypatch,
    tmp_path: Path,
):
    """
    History-preserving range execution:
    when multiple embedding-safe frames exist before requested start,
    execution should begin immediately after the latest prior safe frame.
    """

    class _ResumePlanWithSafeFrames(_FakeResumePlan):
        def __init__(self):
            super().__init__(
                is_resume=True,
                anchor_frame=19,  # expected selected anchor
                first_processed_shot_number=1,
            )
            self.embedding_safe_frames = [5, 12, 19, 40]

    _patch_minimal_tracking_stack(
        monkeypatch,
        resume_plan=_ResumePlanWithSafeFrames(),
    )

    shot_json = tmp_path / "shots.json"
    _write_shots_json(shot_json, first_frame=0, last_frame=99)

    fp = _FakeFrameProvider(total_frames=100)

    tas.track_across_segments(
        frame_source=fp,
        shot_json_path=str(shot_json),
        detector=_FakeDetector(),
        embedder=_FakeEmbedder(),
        start_frame=25,
        end_frame=30,
        resume_enabled=True,
    )

    # latest safe frame strictly before 25 is 19
    assert fp.reset_calls == [20]
    assert fp.frames_returned == [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]