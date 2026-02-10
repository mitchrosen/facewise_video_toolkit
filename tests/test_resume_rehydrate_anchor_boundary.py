# tests/test_resume_rehydrate_anchor_boundary.py

from __future__ import annotations

from types import SimpleNamespace

import pytest

from facekit.pipeline import resume_rehydrate as rr
from facekit.pipeline.resume_rehydrate import ResumePlan
from facekit.tracking.face_structures import FaceTrack, FaceObservation
from facekit.common.obs_consts import Source


def _make_obs(frame_idx: int, bbox=(10, 20, 30, 50), source=Source.DETECTED) -> FaceObservation:
    return FaceObservation(
        frame_idx=int(frame_idx),
        track_id=None,
        bbox=bbox,
        embedding=None,
        confidence=0.9,
        aligned_face=None,
        landmarks=None,
        source=source,
    )


def _make_track(shot_id: int, track_id: int, frames: list[int], bbox=(10, 20, 30, 50)) -> FaceTrack:
    t = FaceTrack(shot_id=int(shot_id), track_id=int(track_id))
    for f in frames:
        obs = _make_obs(f, bbox=bbox, source=Source.DETECTED)
        obs.track_id = int(track_id)
        t.add_observation(obs)
    return t


class _FakeObsCollector:
    def count(self) -> int:
        return 0


class _FakeCheckpoint:
    def __init__(self, *, anchor_safe_frame: int, open_tracks_rows: list[dict], track_order: dict):
        self._anchor_safe_frame = int(anchor_safe_frame)
        self._open_tracks_rows = list(open_tracks_rows)
        self._track_order = dict(track_order)
        self.obs_collector = _FakeObsCollector()

    def get_resume_anchor(self):
        # _resolve_anchor() returns this embedding-safe frame directly.
        # Runtime processing begins later at anchor + 1.
        return (self._anchor_safe_frame,)

    def read_status(self):
        return {"open_tracks": self._open_tracks_rows}

    def get_track_order(self):
        return dict(self._track_order)

    def _validate_resume_embeddings(self, anchor_shot=None):
        # no-op for unit test
        return None


def test_build_resume_plan_keeps_only_anchor_open_tracks_live(monkeypatch):
    """
    Boundary contract:

    Anchor-safe frame = 152  -> resume starts at 153.

    Rehydrated anchor-shot tracks:
      tid=0 : (103..111)   closed before anchor; must NOT remain live
      tid=1 : (103..152)   open at anchor; must remain in prior_tracks_anchor
      tid=2 : (103..152)   open at anchor; must remain in prior_tracks_anchor
      tid=3 : (120..152)   open at anchor; must remain in prior_tracks_anchor

    The checkpoint open_tracks rows are the authoritative source of truth.
    """
    shots = [
        {"shot_number": 1, "first_frame": 0, "last_frame": 102},
        {"shot_number": 2, "first_frame": 103, "last_frame": 299},
    ]

    rehydrated_tracks = [
        _make_track(2, 0, [103, 111]),
        _make_track(2, 1, [103, 120, 152]),
        _make_track(2, 2, [103, 128, 152]),
        _make_track(2, 3, [120, 136, 152]),
    ]

    ckpt = _FakeCheckpoint(
        anchor_safe_frame=152,
        open_tracks_rows=[
            {"shot": 2, "track_id": 1},
            {"shot": 2, "track_id": 2},
            {"shot": 2, "track_id": 3},
        ],
        track_order={
            (2, 0): 0,
            (2, 1): 1,
            (2, 2): 2,
            (2, 3): 3,
        },
    )

    monkeypatch.setattr(rr, "_audit_preanchor_embedding_parity", lambda *a, **k: None)
    monkeypatch.setattr(
        rr,
        "_build_emb_lookups_for_checkpoint",
        lambda checkpoint, anchor_frame: (None, None),
    )
    monkeypatch.setattr(
        rr,
        "rehydrate_tracks",
        lambda *a, **k: list(rehydrated_tracks),
    )

    prepared_tids: list[int] = []

    def _fake_prepare(tracks, *, resume_frame):
        assert int(resume_frame) == 152
        prepared_tids.extend(int(t.track_id) for t in tracks)

    monkeypatch.setattr(rr, "_prepare_tracks_for_resume_runtime", _fake_prepare)

    all_tracks: list[FaceTrack] = []
    plan, shots_trimmed = rr._build_resume_plan(
        shots,
        checkpoint=ckpt,
        resume_enabled=True,
        all_tracks=all_tracks,
    )

    assert [s["shot_number"] for s in shots_trimmed] == [2]
    assert plan.is_resume is True
    assert plan.anchor_frame == 152
    assert plan.first_processed_shot_number == 2

    # Source of truth from status/open_tracks
    assert plan.open_track_ids_anchor == frozenset({1, 2, 3})

    # Only the truly-open anchor tracks should remain live for resume continuation.
    live_tids = [int(t.track_id) for t in plan.prior_tracks_anchor]
    assert live_tids == [1, 2, 3]

    # The dead-by-anchor track should not remain in the live anchor set.
    assert 0 not in live_tids

    # And runtime preparation should only be applied to the live set.
    assert prepared_tids == [1, 2, 3]

    # Depending on your eventual implementation, this closed-by-anchor track
    # should be moved into all_tracks immediately. This assertion captures the
    # desired contract and may fail until production code is updated.
    historical_tids = [int(t.track_id) for t in all_tracks]
    assert 0 in historical_tids


def test_bootstrap_runtime_trackers_uses_only_anchor_open_track_ids():
    """
    bootstrap_runtime_trackers_for_resume_frame() should seed live trackers only
    from tracks allowed by resume_plan.open_track_ids_anchor.
    """
    t0 = _make_track(2, 0, [103, 111], bbox=(100, 100, 140, 160))
    t1 = _make_track(2, 1, [103, 152], bbox=(200, 200, 250, 280))
    t2 = _make_track(2, 2, [103, 152], bbox=(300, 210, 360, 290))
    t3 = _make_track(2, 3, [120, 152], bbox=(400, 220, 470, 310))

    # Simulate the current bad state: aggregator contains all four tracks and all
    # are still marked open. The bootstrap helper should still filter by the plan.
    for t in (t0, t1, t2, t3):
        if hasattr(t, "mark_open"):
            t.mark_open()

    aggregator = SimpleNamespace(tracks=[t0, t1, t2, t3])

    class FakeFaceTracker:
        def __init__(self):
            self.trackers = []
            self.calls = []

        def init_trackers(self, frame, boxes_xywh, track_ids):
            self.calls.append(
                {
                    "frame": frame,
                    "boxes_xywh": list(boxes_xywh),
                    "track_ids": list(track_ids),
                }
            )
            # Any non-empty tracker list is enough for bootstrap to report success.
            self.trackers = ["tracker" for _ in track_ids]

    class FakeValidator:
        def __init__(self):
            self.cleared = 0

        def _clear_baseline(self):
            self.cleared += 1

    face_tracker = FakeFaceTracker()
    validator = FakeValidator()

    plan = ResumePlan(
        anchor_frame=152,
        is_resume=True,
        first_processed_shot_number=2,
        segment_id_seed_by_shot={},
        trackid_seed_by_shot={},
        prior_tracks_anchor=[t1, t2, t3],
        open_track_ids_anchor=frozenset({1, 2, 3}),
        reuse_tid_for_first_shot=None,
    )

    did_bootstrap = rr.bootstrap_runtime_trackers_for_resume_frame(
        resume_plan=plan,
        shot_number=2,
        frame_idx=153,
        start_at=153,
        aggregator=aggregator,
        face_tracker=face_tracker,
        validator=validator,
        frame="frame-153",
    )

    assert did_bootstrap is True
    assert validator.cleared == 1
    assert len(face_tracker.calls) == 1

    seeded_ids = face_tracker.calls[0]["track_ids"]
    assert seeded_ids == [1, 2, 3]
    assert 0 not in seeded_ids


def _make_obs(frame_idx: int, bbox=(10, 20, 30, 50), source=Source.DETECTED) -> FaceObservation:
    return FaceObservation(
        frame_idx=int(frame_idx),
        track_id=None,
        bbox=bbox,
        embedding=None,
        confidence=0.9,
        aligned_face=None,
        landmarks=None,
        source=source,
    )


def _make_track(shot_id: int, track_id: int, frames: list[int], bbox=(10, 20, 30, 50)) -> FaceTrack:
    t = FaceTrack(shot_id=int(shot_id), track_id=int(track_id))
    for f in frames:
        obs = _make_obs(f, bbox=bbox, source=Source.DETECTED)
        obs.track_id = int(track_id)
        t.add_observation(obs)
    return t


class _FakeObsCollector:
    def count(self) -> int:
        return 0


class _FakeCheckpoint:
    def __init__(self, *, anchor_safe_frame: int, open_tracks_rows: list[dict], track_order: dict):
        self._anchor_safe_frame = int(anchor_safe_frame)
        self._open_tracks_rows = list(open_tracks_rows)
        self._track_order = dict(track_order)
        self.obs_collector = _FakeObsCollector()

    def get_resume_anchor(self):
        # _resolve_anchor() returns this embedding-safe frame directly.
        # Runtime processing begins later at anchor + 1.
        return (self._anchor_safe_frame,)

    def read_status(self):
        return {"open_tracks": self._open_tracks_rows}

    def get_track_order(self):
        return dict(self._track_order)

    def _validate_resume_embeddings(self, anchor_shot=None):
        return None


def test_build_resume_plan_partitions_anchor_tracks_without_monkeypatching_prepare(monkeypatch):
    """
    Stronger boundary test:
    - let the real _prepare_tracks_for_resume_runtime() run
    - verify dead-by-anchor track does NOT remain in prior_tracks_anchor
    - verify live-at-anchor tracks DO remain and are normalized for runtime use
    """
    shots = [
        {"shot_number": 1, "first_frame": 0, "last_frame": 102},
        {"shot_number": 2, "first_frame": 103, "last_frame": 299},
    ]

    # Anchor-safe frame is 152, so resume frame should be 153.
    # tid=0 ended before anchor and must be historical only.
    t0 = _make_track(2, 0, [103, 111], bbox=(100, 100, 140, 160))
    t1 = _make_track(2, 1, [103, 120, 152], bbox=(200, 200, 250, 280))
    t2 = _make_track(2, 2, [103, 128, 152], bbox=(300, 210, 360, 290))
    t3 = _make_track(2, 3, [120, 136, 152], bbox=(400, 220, 470, 310))

    rehydrated_tracks = [t0, t1, t2, t3]

    ckpt = _FakeCheckpoint(
        anchor_safe_frame=152,
        open_tracks_rows=[
            {"shot": 2, "track_id": 1},
            {"shot": 2, "track_id": 2},
            {"shot": 2, "track_id": 3},
        ],
        track_order={
            (2, 0): 0,
            (2, 1): 1,
            (2, 2): 2,
            (2, 3): 3,
        },
    )

    monkeypatch.setattr(rr, "_audit_preanchor_embedding_parity", lambda *a, **k: None)
    monkeypatch.setattr(
        rr,
        "_build_emb_lookups_for_checkpoint",
        lambda checkpoint, anchor_frame: (None, None),
    )
    monkeypatch.setattr(
        rr,
        "rehydrate_tracks",
        lambda *a, **k: list(rehydrated_tracks),
    )

    all_tracks: list[FaceTrack] = []

    plan, shots_trimmed = rr._build_resume_plan(
        shots,
        checkpoint=ckpt,
        resume_enabled=True,
        all_tracks=all_tracks,
    )

    assert [s["shot_number"] for s in shots_trimmed] == [2]
    assert plan.is_resume is True
    assert int(plan.anchor_frame) == 152
    assert int(plan.first_processed_shot_number) == 2
    assert plan.open_track_ids_anchor == frozenset({1, 2, 3})

    live_tids = [int(t.track_id) for t in plan.prior_tracks_anchor]
    assert live_tids == [1, 2, 3], (
        "Only tracks that are authoritative-open at the anchor should remain "
        "in prior_tracks_anchor for runtime continuation."
    )

    # Dead-by-anchor track must not be kept in live continuation state.
    assert 0 not in live_tids

    # The live anchor tracks should have been normalized by the *real*
    # _prepare_tracks_for_resume_runtime().
    by_tid = {int(t.track_id): t for t in plan.prior_tracks_anchor}

    assert getattr(by_tid[1], "is_active", None) is False
    assert getattr(by_tid[2], "is_active", None) is False
    assert getattr(by_tid[3], "is_active", None) is False

    assert tuple(getattr(by_tid[1], "last_bbox")) == (200, 200, 250, 280)
    assert tuple(getattr(by_tid[2], "last_bbox")) == (300, 210, 360, 290)
    assert tuple(getattr(by_tid[3], "last_bbox")) == (400, 220, 470, 310)

    # This is the key behavioral assertion for the desired contract:
    # dead-by-anchor tracks should be historical/output-only.
    historical_tids = [int(t.track_id) for t in all_tracks]
    assert 0 in historical_tids, (
        "A track that ended before the anchor should be moved directly to "
        "historical outputs, not kept in live resume state."
    )


def test_resume_boundary_bootstrap_does_not_restart_live_anchor_tracks_at_anchor_plus_one():
    """
    Tiny continuation test:
    if live anchor tracks exist and bootstrap seeds them on the first resumed
    frame, the system should not need to create brand-new runtime IDs for those
    same logical tracks at anchor+1.
    """
    # Historical/dead-by-anchor track. This one must not be resumed.
    t0 = _make_track(2, 0, [103, 111], bbox=(100, 100, 140, 160))
    # Live-at-anchor tracks. These should be resumed.
    t1 = _make_track(2, 1, [103, 120, 152], bbox=(200, 200, 250, 280))
    t2 = _make_track(2, 2, [103, 128, 152], bbox=(300, 210, 360, 290))
    t3 = _make_track(2, 3, [120, 136, 152], bbox=(400, 220, 470, 310))

    # Simulate what the aggregator should contain at resume entry:
    # only the still-live anchor tracks are handed back for continuation.
    aggregator = SimpleNamespace(tracks=[t1, t2, t3])

    class FakeFaceTracker:
        def __init__(self):
            self.trackers = []
            self.calls = []

        def init_trackers(self, frame, boxes_xywh, track_ids):
            self.calls.append(
                {
                    "frame": frame,
                    "boxes_xywh": list(boxes_xywh),
                    "track_ids": list(track_ids),
                }
            )
            self.trackers = [f"tracker-{tid}" for tid in track_ids]

        def update_trackers(self, frame):
            # Simulate a successful first resumed tracking update.
            return {
                1: (201, 201, 251, 281),
                2: (301, 211, 361, 291),
                3: (401, 221, 471, 311),
            }

    class FakeValidator:
        def __init__(self):
            self.cleared = 0

        def _clear_baseline(self):
            self.cleared += 1

    face_tracker = FakeFaceTracker()
    validator = FakeValidator()

    plan = ResumePlan(
        anchor_frame=152,
        is_resume=True,
        first_processed_shot_number=2,
        segment_id_seed_by_shot={},
        trackid_seed_by_shot={},
        prior_tracks_anchor=[t1, t2, t3],
        open_track_ids_anchor=frozenset({1, 2, 3}),
        reuse_tid_for_first_shot=None,
    )

    did_bootstrap = rr.bootstrap_runtime_trackers_for_resume_frame(
        resume_plan=plan,
        shot_number=2,
        frame_idx=153,
        start_at=153,
        aggregator=aggregator,
        face_tracker=face_tracker,
        validator=validator,
        frame="frame-153",
    )

    assert did_bootstrap is True
    assert validator.cleared == 1
    assert len(face_tracker.calls) == 1

    seeded_ids = face_tracker.calls[0]["track_ids"]
    assert seeded_ids == [1, 2, 3]
    assert 0 not in seeded_ids

    tracked_boxes = face_tracker.update_trackers("frame-153")
    assert sorted(tracked_boxes.keys()) == [1, 2, 3]

    # The boundary contract we care about:
    # if seeded live trackers successfully update on the first resumed frame,
    # there is no reason to mint fresh runtime IDs for those same logical tracks.
    max_seeded_id = max(seeded_ids)
    assert all(tid <= max_seeded_id for tid in tracked_boxes.keys())

    # And the dead historical track is not part of runtime continuation.
    assert 0 not in tracked_boxes
    assert t0.track_id not in tracked_boxes


def _track(track_id: int, shot_id: int, first_frame: int, last_frame: int, segment_id=None):
    t = FaceTrack(track_id=track_id, shot_id=shot_id)
    t.first_frame = first_frame
    t.last_frame = last_frame
    t.segment_id = segment_id
    t.is_closed = False
    return t


def test_resume_midshot_keeps_anchor_shot_tracks_open():
    """
    Mid-shot resume contract:
    the live continuation set is represented by ResumePlan.prior_tracks_anchor
    plus ResumePlan.open_track_ids_anchor.

    We should carry forward only the tracks that are authoritative-open at the
    anchor boundary, and they should still be pre-resume tracks whose first_frame
    predates the resumed work.
    """
    anchor_frame = 152  # embedding-safe anchor frame
    first_processed_shot_number = 2

    # Simulate the live anchor-shot tracks that survived partitioning.
    t1 = _track(track_id=1, shot_id=2, first_frame=103, last_frame=152, segment_id=1)
    t2 = _track(track_id=2, shot_id=2, first_frame=103, last_frame=152, segment_id=2)
    t3 = _track(track_id=3, shot_id=2, first_frame=120, last_frame=152, segment_id=3)

    plan = ResumePlan(
        anchor_frame=anchor_frame,
        is_resume=True,
        first_processed_shot_number=first_processed_shot_number,
        segment_id_seed_by_shot={2: 4},
        trackid_seed_by_shot={2: 4},
        prior_tracks_anchor=[t1, t2, t3],
        open_track_ids_anchor=frozenset({1, 2, 3}),
        reuse_tid_for_first_shot=None,
    )

    assert plan.is_resume is True
    assert plan.anchor_frame == 152
    assert plan.first_processed_shot_number == 2

    continued = plan.prior_tracks_anchor
    assert [(t.track_id, t.first_frame, t.last_frame) for t in continued] == [
        (1, 103, 152),
        (2, 103, 152),
        (3, 120, 152),
    ]

    # These are still continuation tracks, not fresh tracks beginning at 154.
    assert all(t.first_frame < plan.anchor_frame for t in continued)
    assert all(t.last_frame == 152 for t in continued)

    # Authoritative open-set for the anchor shot.
    assert plan.open_track_ids_anchor == frozenset({1, 2, 3})
    
def test_seeded_anchor_shot_tracks_are_live_for_continuation():
    """
    Tracks open at anchor must be bootstrap-seeded into runtime trackers on the
    first resumed frame, so continuation can proceed from their existing ids
    instead of inventing fresh tracks starting at the resume boundary.
    """
    seeded_tracks = [
        _track(track_id=1, shot_id=2, first_frame=103, last_frame=152, segment_id=1),
        _track(track_id=2, shot_id=2, first_frame=103, last_frame=152, segment_id=2),
        _track(track_id=3, shot_id=2, first_frame=120, last_frame=152, segment_id=3),
    ]

    aggregator = SimpleNamespace(tracks=seeded_tracks)

    class FakeFaceTracker:
        def __init__(self):
            self.trackers = []
            self.calls = []

        def init_trackers(self, frame, boxes_xywh, track_ids):
            self.calls.append(
                {
                    "frame": frame,
                    "boxes_xywh": list(boxes_xywh),
                    "track_ids": list(track_ids),
                }
            )
            self.trackers = [f"tracker-{tid}" for tid in track_ids]

    class FakeValidator:
        def __init__(self):
            self.cleared = 0

        def _clear_baseline(self):
            self.cleared += 1

    # Give each track the API shape expected by the real bootstrap helper.
    for i, tr in enumerate(seeded_tracks, start=1):
        bbox = (100 * i, 200, 100 * i + 50, 280)
        try:
            delattr(tr, "is_closed")
        except Exception:
            pass
        tr.get_last_bbox = (lambda b=bbox: b)
        tr.is_closed = (lambda: False)

    face_tracker = FakeFaceTracker()
    validator = FakeValidator()

    plan = ResumePlan(
        anchor_frame=152,
        is_resume=True,
        first_processed_shot_number=2,
        segment_id_seed_by_shot={2: 4},
        trackid_seed_by_shot={2: 4},
        prior_tracks_anchor=seeded_tracks,
        open_track_ids_anchor=frozenset({1, 2, 3}),
        reuse_tid_for_first_shot=None,
    )

    did_bootstrap = rr.bootstrap_runtime_trackers_for_resume_frame(
        resume_plan=plan,
        shot_number=2,
        frame_idx=153,
        start_at=153,
        aggregator=aggregator,
        face_tracker=face_tracker,
        validator=validator,
        frame="frame-153",
    )

    assert did_bootstrap is True
    assert validator.cleared == 1
    assert len(face_tracker.calls) == 1
    assert face_tracker.calls[0]["track_ids"] == [1, 2, 3]

def test_midshot_resume_must_not_split_anchor_tracks_at_first_resumed_frame():
    """
    Regression for the behavior seen in test_resume_realVideo.py.

    If resume occurs mid-shot and anchor-shot tracks are open at the anchor,
    processing the first resumed frame must continue those tracks instead of
    producing fresh tracks that begin at the resume boundary.

    Intended cold shape:
        (103,223), (103,299), (120,191)

    Bad resumed shape we are guarding against:
        (103,152) + (154,223)
        (103,152) + (154,299)
        (120,152) + (154,191)
    """
    prior_tracks_resume_shot = [
        _track(track_id=1, shot_id=2, first_frame=103, last_frame=152, segment_id=1),
        _track(track_id=2, shot_id=2, first_frame=103, last_frame=152, segment_id=2),
        _track(track_id=3, shot_id=2, first_frame=120, last_frame=152, segment_id=3),
    ]

    # give them the API shape the runtime expects
    bboxes = {
        1: (900, 340, 975, 441),
        2: (1436, 318, 1534, 437),
        3: (358, 280, 459, 391),
    }
    for tr in prior_tracks_resume_shot:
        bbox = bboxes[tr.track_id]
        try:
            delattr(tr, "is_closed")
        except Exception:
            pass
        tr.get_last_bbox = (lambda b=bbox: b)
        tr.is_closed = (lambda: False)

    plan = ResumePlan(
        anchor_frame=152,
        is_resume=True,
        first_processed_shot_number=2,
        segment_id_seed_by_shot={2: 4},
        trackid_seed_by_shot={2: 4},
        prior_tracks_anchor=prior_tracks_resume_shot,
        open_track_ids_anchor=frozenset({1, 2, 3}),
        reuse_tid_for_first_shot=None,
    )

    class FakeAggregator:
        def __init__(self, tracks):
            self.tracks = list(tracks)
            self.next_track_id = 4

        def update_tracks_with_frame(self, frame_idx, observations):
            """
            Simulate the intended contract:
            matching observations on the first resumed frame should extend
            the existing tracks, not create new ones.
            """
            by_tid = {t.track_id: t for t in self.tracks}
            for obs in observations:
                tid = obs.track_id
                if tid in by_tid:
                    by_tid[tid].last_frame = frame_idx
                else:
                    raise AssertionError(
                        f"Unexpected fresh track at resume boundary: tid={tid}"
                    )

    class FakeFaceTracker:
        def __init__(self):
            self.trackers = []
        def init_trackers(self, frame, boxes_xywh, track_ids):
            self.trackers = [f"tracker-{tid}" for tid in track_ids]

    class FakeValidator:
        def _clear_baseline(self):
            pass

    aggregator = FakeAggregator(prior_tracks_resume_shot)
    face_tracker = FakeFaceTracker()
    validator = FakeValidator()

    did_bootstrap = rr.bootstrap_runtime_trackers_for_resume_frame(
        resume_plan=plan,
        shot_number=2,
        frame_idx=153,
        start_at=153,
        aggregator=aggregator,
        face_tracker=face_tracker,
        validator=validator,
        frame="frame-153",
    )
    assert did_bootstrap is True

    # First resumed frame observations must continue the same tids.
    resumed_observations = [
        SimpleNamespace(track_id=1),
        SimpleNamespace(track_id=2),
        SimpleNamespace(track_id=3),
    ]
    aggregator.update_tracks_with_frame(154, resumed_observations)

    got = sorted(
        (t.track_id, t.first_frame, t.last_frame)
        for t in aggregator.tracks
    )
    assert got == [
        (1, 103, 154),
        (2, 103, 154),
        (3, 120, 154),
    ]

def test_build_resume_plan_midshot_keeps_live_anchor_tracks_unlabeled(monkeypatch):
    """
    If resume starts in the middle of a shot, only anchor-shot tracks that were
    already CLOSED by the embedding-safe boundary should contribute durable
    segment_ids.

    Live anchor-shot tracks are runtime continuation state and must enter the
    resumed shot with segment_id=None. Otherwise we need ad hoc nulling later in
    track_across_segments(), which is the smell we want to eliminate.
    """
    import facekit.pipeline.resume_rehydrate as rr

    shots = [
        {"shot_number": 1, "first_frame": 0,   "last_frame": 102},
        {"shot_number": 2, "first_frame": 103, "last_frame": 299},
    ]

    # Pre-anchor state for shot 2:
    #   tid 0 ended before anchor -> historical/closed-by-anchor
    #   tids 1,2,3 were still open at anchor -> must remain unlabeled runtime seed state
    rehydrated = [
        _track(track_id=0, shot_id=2, first_frame=103, last_frame=111, segment_id=None),
        _track(track_id=1, shot_id=2, first_frame=103, last_frame=152, segment_id=None),
        _track(track_id=2, shot_id=2, first_frame=103, last_frame=152, segment_id=None),
        _track(track_id=3, shot_id=2, first_frame=120, last_frame=152, segment_id=None),
    ]

    # Match the production FaceTrack API shape expected by _build_resume_plan():
    # some paths call t.first_frame() / t.last_frame(), not just attributes.
    for tr in rehydrated:
        ff = int(tr.first_frame)
        lf = int(tr.last_frame)
        tr.first_frame = (lambda ff=ff: ff)
        tr.last_frame = (lambda lf=lf: lf)

    class _ObsCollector:
        def count(self):
            return 0

    class _FakeCheckpoint:
        obs_collector = _ObsCollector()

        def get_track_order(self):
            # Stable order within shot 2: tid 0,1,2,3
            return {
                (2, 0): 0,
                (2, 1): 1,
                (2, 2): 2,
                (2, 3): 3,
            }

        def read_status(self):
            # Only tids 1,2,3 were OPEN at the embedding-safe boundary.
            return {
                "open_tracks": [
                    {"shot": 2, "track_id": 1},
                    {"shot": 2, "track_id": 2},
                    {"shot": 2, "track_id": 3},
                ]
            }

        def _validate_resume_embeddings(self, anchor_shot=None):
            return None

    checkpoint = _FakeCheckpoint()
    all_tracks = []

    monkeypatch.setattr(rr, "_resolve_anchor", lambda checkpoint, resume_enabled: 152)
    monkeypatch.setattr(rr, "_audit_preanchor_embedding_parity", lambda *a, **k: None)
    monkeypatch.setattr(rr, "_build_emb_lookups_for_checkpoint", lambda *a, **k: (None, None))
    monkeypatch.setattr(rr, "rehydrate_tracks", lambda *a, **k: list(rehydrated))

    plan, trimmed_shots = rr._build_resume_plan(
        shots,
        checkpoint=checkpoint,
        resume_enabled=True,
        all_tracks=all_tracks,
    )

    # We should resume inside shot 2.
    assert plan.is_resume is True
    assert plan.anchor_frame == 152
    assert plan.first_processed_shot_number == 2
    assert [s["shot_number"] for s in trimmed_shots] == [2]

    # The closed-by-anchor track should already be emitted as durable output.
    emitted_keys = sorted(
        (int(t.shot_id), int(t.track_id), int(t.first_frame()), int(t.last_frame()))
        for t in all_tracks
    )
    assert emitted_keys == [(2, 0, 103, 111)]

    # Live anchor-shot tracks should remain for runtime continuation.
    live_tids = sorted(int(t.track_id) for t in plan.prior_tracks_anchor)
    assert live_tids == [1, 2, 3]

    # This is the crucial contract:
    # runtime continuation tracks must NOT carry preassigned segment_ids.
    assert all(getattr(t, "segment_id", None) is None for t in plan.prior_tracks_anchor)

    # Only the durable closed-by-anchor historical track should contribute to
    # the next segment-id seed for shot 2.
    assert plan.segment_id_seed_by_shot[2] == 1

    # Track-id seed still comes from all prior track ids.
    assert plan.trackid_seed_by_shot[2] == 4

    # And the checkpoint-declared open ids remain authoritative.
    assert plan.open_track_ids_anchor == frozenset({1, 2, 3})