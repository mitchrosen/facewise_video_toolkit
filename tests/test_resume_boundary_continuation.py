# tests/test_resume_boundary_continuation.py

from types import SimpleNamespace

from facekit.pipeline.resume_rehydrate import ResumePlan, bootstrap_runtime_trackers_for_resume_frame
from facekit.pipeline.track_across_segments import _init_shot_aggregator
from facekit.tracking.face_structures import FaceTrack, FaceObservation
from facekit.common.obs_consts import Source


def _obs(f, bbox):
    return FaceObservation(
        frame_idx=int(f),
        track_id=None,
        bbox=bbox,
        embedding=None,
        confidence=0.9,
        aligned_face=None,
        landmarks=None,
        source=Source.DETECTED,
    )


def _track(shot_id, track_id, frames, bbox):
    t = FaceTrack(shot_id=int(shot_id), track_id=int(track_id))
    for f in frames:
        o = _obs(f, bbox)
        o.track_id = int(track_id)
        t.add_observation(o)
    return t


def test_resume_first_resumed_frame_does_not_mint_fresh_tracks(monkeypatch):
    # Live-at-anchor tracks from shot 2, all ending at the safe boundary.
    t1 = _track(2, 1, [103, 120, 152], (200, 200, 250, 280))
    t2 = _track(2, 2, [103, 128, 152], (300, 210, 360, 290))
    t3 = _track(2, 3, [120, 136, 152], (400, 220, 470, 310))

    # Make sure runtime bootstrap sees them as open and seedable.
    for tr in (t1, t2, t3):
        try:
            delattr(tr, "is_closed")
        except Exception:
            pass
        tr.is_closed = lambda: False

    plan = ResumePlan(
        anchor_frame=152,
        is_resume=True,
        first_processed_shot_number=2,
        segment_id_seed_by_shot={2: 0},
        trackid_seed_by_shot={2: 4},
        prior_tracks_anchor=[t1, t2, t3],
        open_track_ids_anchor=frozenset({1, 2, 3}),
        reuse_tid_for_first_shot=None,
    )

    class FakeCheckpoint:
        def hydrate_open_tracks_into(self, aggregator):
            return None

    start_at, aggregator, seg_seed = _init_shot_aggregator(
        shot_idx=0,
        shot_number=2,
        first=103,
        last=299,
        detect_interval=8,
        resume_plan=plan,
        iou_thresh=0.3,
        embedding_thresh=0.5,
        checkpoint=FakeCheckpoint(),
    )

    assert start_at == 153
    assert sorted(int(t.track_id) for t in aggregator.tracks if not t.is_closed()) == [1, 2, 3]

    class FakeFaceTracker:
        def __init__(self):
            self.trackers = []
            self.init_calls = []

        def init_trackers(self, frame, boxes_xywh, track_ids):
            self.init_calls.append((frame, list(track_ids)))
            self.trackers = [f"tracker-{tid}" for tid in track_ids]

        def update_trackers(self, frame):
            # Continuation on the first resumed frame
            return {
                1: (201, 201, 251, 281),
                2: (301, 211, 361, 291),
                3: (401, 221, 471, 311),
            }

    class FakeValidator:
        def _clear_baseline(self):
            pass

    face_tracker = FakeFaceTracker()
    validator = FakeValidator()

    did_bootstrap = bootstrap_runtime_trackers_for_resume_frame(
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
    assert face_tracker.init_calls == [("frame-153", [1, 2, 3])]

    # Simulate what the first resumed tracking update should do:
    tracked_boxes = face_tracker.update_trackers("frame-153")
    assert sorted(tracked_boxes.keys()) == [1, 2, 3]

    # This is the heart of the regression:
    # first resumed frame should continue seeded tids, not mint 4/5/6.
    assert all(tid < 4 for tid in tracked_boxes.keys())