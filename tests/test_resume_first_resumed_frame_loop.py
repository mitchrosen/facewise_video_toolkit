# tests/test_resume_first_resumed_frame_loop.py

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from facekit.pipeline import track_across_segments as tas_mod
from facekit.pipeline.resume_rehydrate import ResumePlan
from facekit.tracking.face_structures import FaceTrack, FaceObservation
from facekit.common.obs_consts import Source


def _obs(frame_idx: int, bbox, source=Source.DETECTED) -> FaceObservation:
    return FaceObservation(
        frame_idx=int(frame_idx),
        track_id=None,
        bbox=tuple(int(v) for v in bbox),
        embedding=None,
        confidence=0.9,
        aligned_face=None,
        landmarks=None,
        source=source,
    )


def _track(shot_id: int, track_id: int, frames: list[int], bbox) -> FaceTrack:
    t = FaceTrack(shot_id=int(shot_id), track_id=int(track_id))
    for f in frames:
        o = _obs(f, bbox=bbox, source=Source.DETECTED)
        o.track_id = int(track_id)
        t.add_observation(o)
    return t


class FakeFrameProvider:
    def __init__(self, total_frames: int = 400, fps: float = 30.0, size=(64, 48)):
        self.total_frames = int(total_frames)
        self.fps = float(fps)
        self.size = tuple(int(v) for v in size)
        self.reset_calls: list[int] = []
        self.next_calls: list[int] = []
        self._cursor = 0

    def reset_to_frame(self, frame_idx: int):
        self._cursor = int(frame_idx)
        self.reset_calls.append(int(frame_idx))

    def next(self):
        if self._cursor >= self.total_frames:
            return None
        f = self._cursor
        self.next_calls.append(f)
        self._cursor += 1
        # tiny dummy frame
        w, h = self.size
        return np.zeros((h, w, 3), dtype=np.uint8)

class DummyDetector:
    def __init__(self):
        self.calls: list[int] = []

    def detect_faces_in_frame(self, frame, target_size=640):
        # In this test we expect tracking-first continuity after bootstrap,
        # so a detector call on the boundary would be suspicious.
        self.calls.append(1)
        return ([], [], [])


class DummyEmbedder:
    def set_max_batch_size(self, n):
        return None

    def get_embedding_batch(self, aligned_faces):
        if not aligned_faces:
            return np.zeros((0, 512), dtype=np.float32)
        return np.zeros((len(aligned_faces), 512), dtype=np.float32)


class FakeFaceTracker:
    def __init__(self):
        self.trackers = []
        self.init_calls = []
        self.update_calls = []

    def init_trackers(self, frame, boxes_xywh, track_ids):
        self.init_calls.append(
            {
                "frame": frame,
                "track_ids": list(track_ids),
                "boxes_xywh": list(boxes_xywh),
            }
        )
        self.trackers = [f"tracker-{tid}" for tid in track_ids]

    def update_trackers(self, frame):
        self.update_calls.append(frame)
        # successful continuation for the seeded tids
        return {
            1: (201, 201, 251, 281),
            2: (301, 211, 361, 291),
            3: (401, 221, 471, 311),
        }


class FakeValidator:
    def __init__(self, *args, **kwargs):
        self.seed_calls = []
        self.cleared = 0

    def seed_validator(self, boxes_map, frame_idx, frame):
        self.seed_calls.append((dict(boxes_map), int(frame_idx)))

    def _clear_baseline(self):
        self.cleared += 1

    def validate(self, tracked_boxes, frame, frame_idx):
        return True

class FakeAggregator:
    instances = []

    def __init__(self, *args, **kwargs):
        prior_tracks = kwargs.get("prior_tracks") or []
        self.tracks = list(prior_tracks)
        self.next_track_id = int(kwargs.get("next_track_id", 0))
        self.update_calls = []
        self.observations_by_frame = {}
        self.finalized = False
        FakeAggregator.instances.append(self)

    def rehydrate_open_tracks(self, rows):
        return 0

    def add_existing_track(self, track):
        self.tracks.append(track)

    def observations_at(self, frame_idx, source=None, require_track_id=False):
        out = list(self.observations_by_frame.get(int(frame_idx), []))
        if source is not None:
            out = [o for o in out if getattr(o, "source", None) == source]
        if require_track_id:
            out = [o for o in out if getattr(o, "track_id", None) is not None]
        return out

    def update_tracks_with_frame(self, frame_idx, observations):
        self.update_calls.append(
            (
                int(frame_idx),
                sorted(int(o.track_id) for o in observations),
            )
        )
        by_tid = {int(t.track_id): t for t in self.tracks}
        frame_obs = []
        for o in observations:
            tid = int(o.track_id)
            if tid not in by_tid:
                raise AssertionError(f"Unexpected fresh track at resume boundary: tid={tid}")
            by_tid[tid].add_observation(o)
            frame_obs.append(o)
        self.observations_by_frame[int(frame_idx)] = frame_obs

    def finalize_tracks(self):
        self.finalized = True

    def resolve_segment_ids(self, segment_id_counter=0, embedding_threshold=0.7):
        return int(segment_id_counter)


def test_track_across_segments_first_resumed_frame_continues_seeded_tracks(
    monkeypatch, tmp_path: Path
):
    # Shot 1 is irrelevant except that shot 2 begins at 103.
    shots_path = tmp_path / "shots.json"
    shots_path.write_text(
        json.dumps(
            {
                "shots": [
                    {"shot_number": 1, "first_frame": 0, "last_frame": 102},
                    {"shot_number": 2, "first_frame": 103, "last_frame": 155},
                ]
            }
        )
    )

    # Live-at-anchor tracks from shot 2.
    t1 = _track(2, 1, [103, 120, 152], (200, 200, 250, 280))
    t2 = _track(2, 2, [103, 128, 152], (300, 210, 360, 290))
    t3 = _track(2, 3, [120, 136, 152], (400, 220, 470, 310))

    # Give tracks the runtime shape expected by bootstrap/tracking code.
    for tr, bbox in [
        (t1, (200, 200, 250, 280)),
        (t2, (300, 210, 360, 290)),
        (t3, (400, 220, 470, 310)),
    ]:
        try:
            delattr(tr, "is_closed")
        except Exception:
            pass
        tr.is_closed = (lambda: False)
        tr.get_last_bbox = (lambda b=bbox: b)

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

    fp = FakeFrameProvider(total_frames=300)
    detector = DummyDetector()
    embedder = DummyEmbedder()
    fake_face_tracker = FakeFaceTracker()

    monkeypatch.setattr(tas_mod, "_build_resume_plan", lambda *a, **k: (plan, json.loads(shots_path.read_text())["shots"][1:]))
    monkeypatch.setattr(tas_mod, "ShotFaceTrackAggregator", FakeAggregator)
    monkeypatch.setattr(tas_mod, "FaceTracker", lambda *a, **k: fake_face_tracker)
    monkeypatch.setattr(tas_mod, "TrackerValidator", FakeValidator)
    monkeypatch.setattr(tas_mod, "align_face_for_arcface", lambda *a, **k: None)

    all_tracks = tas_mod.track_across_segments(
        frame_source=fp,
        shot_json_path=str(shots_path),
        detector=detector,
        embedder=embedder,
        detect_interval=8,   # 153 and 154 are not scheduled detect frames
        embedding_batch_size_max=8,
        embedding_queue_max_pending=99,
        checkpoint=None,
        resume_enabled=True,
    )

    # Resume should seek to anchor + 1.
    assert fp.reset_calls[0] == 153
    assert fp.next_calls[:2] == [153, 154]

    # Bootstrap should seed live tids into trackers on the first resumed frame.
    assert len(fake_face_tracker.init_calls) >= 1
    assert fake_face_tracker.init_calls[0]["track_ids"] == [1, 2, 3]

    # The first resumed frames should continue the existing tids, not mint 4/5/6.
    agg = FakeAggregator.instances[0]
    assert agg.update_calls[0] == (153, [1, 2, 3])
    assert agg.update_calls[1] == (154, [1, 2, 3])

    # No fresh tids should appear at the boundary.
    all_seen_tids = sorted({tid for _, tids in agg.update_calls for tid in tids})
    assert all_seen_tids == [1, 2, 3]

    # The tracks should continue from their pre-anchor starts.
    got = sorted(
        (int(t.track_id), int(t.first_frame()), int(t.last_frame()))
        for t in all_tracks
        if int(getattr(t, "shot_id", -1)) == 2
    )
    assert (1, 103, 155) in got
    assert (2, 103, 155) in got
    assert (3, 120, 155) in got

    # A detector-driven restart at the boundary would be suspicious in this test.
    assert detector.calls == []

def test_resume_boundary_tracking_failure_fallback_does_not_split_seeded_tracks(
    monkeypatch, tmp_path: Path
):
    # Shot 2 begins at 103; resume anchor is 152 so first resumed frame is 153.
    shots_path = tmp_path / "shots.json"
    shots_path.write_text(
        json.dumps(
            {
                "shots": [
                    {"shot_number": 1, "first_frame": 0, "last_frame": 102},
                    {"shot_number": 2, "first_frame": 103, "last_frame": 154},
                ]
            }
        )
    )

    FakeAggregator.instances.clear()

    # Live-at-anchor tracks from shot 2.
    t1 = _track(2, 1, [103, 120, 152], (200, 200, 250, 280))
    t2 = _track(2, 2, [103, 128, 152], (300, 210, 360, 290))
    t3 = _track(2, 3, [120, 136, 152], (400, 220, 470, 310))

    for tr, bbox in [
        (t1, (200, 200, 250, 280)),
        (t2, (300, 210, 360, 290)),
        (t3, (400, 220, 470, 310)),
    ]:
        try:
            delattr(tr, "is_closed")
        except Exception:
            pass
        tr.is_closed = (lambda: False)
        tr.get_last_bbox = (lambda b=bbox: b)

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

    fp = FakeFrameProvider(total_frames=300)

    class FailingFaceTracker(FakeFaceTracker):
        def update_trackers(self, frame):
            self.update_calls.append(frame)
            # Force tracking failure on the first resumed frame.
            return {1: None, 2: None, 3: None}

    class FallbackDetector:
        def __init__(self):
            self.calls = []

        def detect_faces_in_frame(self, frame, target_size=640):
            self.calls.append(frame)
            # 3 detections in the same spatial neighborhood as the seeded tracks.
            boxes_xyxy = [
                (201, 201, 251, 281),
                (301, 211, 361, 291),
                (401, 221, 471, 311),
            ]
            landmarks = [
                [(210, 210), (240, 210), (225, 230), (215, 250), (235, 250)],
                [(310, 220), (350, 220), (330, 240), (318, 265), (342, 265)],
                [(410, 230), (460, 230), (435, 250), (420, 285), (450, 285)],
            ]
            confidences = [0.9, 0.9, 0.9]
            return (boxes_xyxy, landmarks, confidences)

    class RecordingAggregator(FakeAggregator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.created_by_frame = []

        def update_tracks_with_frame(self, frame_idx, observations):
            before = {int(t.track_id) for t in self.tracks}
            super().update_tracks_with_frame(frame_idx, observations)
            after = {int(t.track_id) for t in self.tracks}
            new_ids = sorted(after - before)
            obs_list = list(observations)
            if obs_list and all(getattr(o, "source", None) == Source.DETECTED for o in obs_list):
                self.created_by_frame.append(
                    (int(frame_idx), int(len(new_ids)), new_ids)
                )
            return None

    detector = FallbackDetector()
    embedder = DummyEmbedder()
    fake_face_tracker = FailingFaceTracker()

    monkeypatch.setattr(
        tas_mod,
        "_build_resume_plan",
        lambda *a, **k: (plan, json.loads(shots_path.read_text())["shots"][1:]),
    )
    monkeypatch.setattr(tas_mod, "ShotFaceTrackAggregator", RecordingAggregator)
    monkeypatch.setattr(tas_mod, "FaceTracker", lambda *a, **k: fake_face_tracker)
    monkeypatch.setattr(tas_mod, "TrackerValidator", FakeValidator)
    monkeypatch.setattr(tas_mod, "align_face_for_arcface", lambda *a, **k: None)

    all_tracks = tas_mod.track_across_segments(
        frame_source=fp,
        shot_json_path=str(shots_path),
        detector=detector,
        embedder=embedder,
        detect_interval=8,
        embedding_batch_size_max=8,
        embedding_queue_max_pending=99,
        checkpoint=None,
        resume_enabled=True,
    )

    # We still started correctly.
    assert fp.reset_calls[0] == 153
    assert fake_face_tracker.init_calls[0]["track_ids"] == [1, 2, 3]

    # Tracking really did fail, so fallback detection should have been used.
    assert len(fake_face_tracker.update_calls) >= 1
    assert len(detector.calls) >= 1

    agg = RecordingAggregator.instances[-1]

    # This is the contract we want.
    # On fallback, the seeded live tracks should be continued rather than replaced
    # by fresh tids minted at the resume boundary.
    assert agg.created_by_frame[0][0] == 153
    assert agg.created_by_frame[0][1] == 0
    assert agg.created_by_frame[0][2] == []

    got = sorted(
        (int(t.track_id), int(t.first_frame()), int(t.last_frame()))
        for t in all_tracks
        if int(getattr(t, "shot_id", -1)) == 2
    )

    assert (1, 103, 154) in got
    assert (2, 103, 154) in got
    assert (3, 120, 154) in got

    # And no fresh tids should appear.
    assert all(tid in {1, 2, 3} for tid, _, _ in got)