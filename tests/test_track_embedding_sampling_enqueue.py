import numpy as np

from facekit.common.obs_consts import Source
from facekit.pipeline import track_across_segments as tas
from facekit.tracking.face_structures import FaceObservation


class DummyTrack:
    def __init__(self, track_id, observations):
        self.track_id = track_id
        self.observations = observations
        self.shot_id = 1


class DummyAggregator:
    def __init__(self, tracks, observations_for_frame):
        self.tracks = tracks
        self._observations_for_frame = observations_for_frame

    def observations_at(self, frame_idx, require_track_id=True):
        assert frame_idx == 42
        return self._observations_for_frame


class DummyQueue:
    def __init__(self):
        self.pending = []

    def enqueue(self, item):
        self.pending.append(item)


def make_obs(*, frame_idx, track_id, bbox, source, landmarks=None, aligned_face=None):
    return FaceObservation(
        frame_idx=frame_idx,
        track_id=track_id,
        bbox=bbox,
        embedding=None,
        confidence=None,
        aligned_face=aligned_face,
        landmarks=landmarks,
        source=source,
    )


def test_enqueue_tracked_observation_when_sampling_interval_hits(monkeypatch):
    prev_obs = make_obs(
        frame_idx=41,
        track_id=7,
        bbox=(100, 50, 200, 150),
        source=Source.TRACKED,
        landmarks=np.array(
            [
                [120.0, 80.0],
                [180.0, 80.0],
                [150.0, 105.0],
                [130.0, 130.0],
                [170.0, 130.0],
            ],
            dtype=np.float32,
        ),
    )
    tracked_obs = make_obs(
        frame_idx=42,
        track_id=7,
        bbox=(112, 68, 212, 168),
        source=Source.TRACKED,
        landmarks=np.array(
            [
                [132.0, 98.0],
                [192.0, 98.0],
                [162.0, 123.0],
                [142.0, 148.0],
                [182.0, 148.0],
            ],
            dtype=np.float32,
        ),
    )

    track = DummyTrack(track_id=7, observations=[prev_obs, tracked_obs])
    aggregator = DummyAggregator(tracks=[track], observations_for_frame=[tracked_obs])
    queue = DummyQueue()
    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    aligned_face = np.ones((112, 112, 3), dtype=np.uint8)

    def fake_align_face(frame_bgr, landmarks):
        assert frame_bgr is frame
        assert landmarks is tracked_obs.landmarks
        return aligned_face

    monkeypatch.setattr(tas, "align_face_for_arcface", fake_align_face)

    tas._maybe_enqueue_track_embedding_observations_for_frame(
        aggregator=aggregator,
        frame_idx=42,
        frame=frame,
        track_sample_interval=1,
        embedding_queue=queue,
    )

    assert tracked_obs.aligned_face is aligned_face
    assert len(queue.pending) == 1
    assert queue.pending[0] is tracked_obs


def test_do_not_enqueue_tracked_observation_off_sampling_interval(monkeypatch):
    prev_obs = make_obs(
        frame_idx=41,
        track_id=7,
        bbox=(100, 50, 200, 150),
        source=Source.TRACKED,
        landmarks=np.ones((5, 2), dtype=np.float32),
    )
    tracked_obs = make_obs(
        frame_idx=42,
        track_id=7,
        bbox=(112, 68, 212, 168),
        source=Source.TRACKED,
        landmarks=np.ones((5, 2), dtype=np.float32),
    )

    track = DummyTrack(track_id=7, observations=[prev_obs, tracked_obs])
    aggregator = DummyAggregator(tracks=[track], observations_for_frame=[tracked_obs])
    queue = DummyQueue()
    frame = np.zeros((256, 256, 3), dtype=np.uint8)

    def fake_align_face(frame_bgr, landmarks):
        raise AssertionError("alignment should not be attempted off sampling interval")

    monkeypatch.setattr(tas, "align_face_for_arcface", fake_align_face)

    tas._maybe_enqueue_track_embedding_observations_for_frame(
        aggregator=aggregator,
        frame_idx=42,
        frame=frame,
        track_sample_interval=3,  # track_local_index for tracked_obs is 1
        embedding_queue=queue,
    )

    assert tracked_obs.aligned_face is None
    assert queue.pending == []


def test_do_not_enqueue_tracked_observation_without_landmarks(monkeypatch):
    prev_obs = make_obs(
        frame_idx=41,
        track_id=7,
        bbox=(100, 50, 200, 150),
        source=Source.TRACKED,
        landmarks=np.ones((5, 2), dtype=np.float32),
    )
    tracked_obs = make_obs(
        frame_idx=42,
        track_id=7,
        bbox=(112, 68, 212, 168),
        source=Source.TRACKED,
        landmarks=None,
    )

    track = DummyTrack(track_id=7, observations=[prev_obs, tracked_obs])
    aggregator = DummyAggregator(tracks=[track], observations_for_frame=[tracked_obs])
    queue = DummyQueue()
    frame = np.zeros((256, 256, 3), dtype=np.uint8)

    def fake_align_face(frame_bgr, landmarks):
        raise AssertionError("alignment should not be attempted without landmarks")

    monkeypatch.setattr(tas, "align_face_for_arcface", fake_align_face)

    tas._maybe_enqueue_track_embedding_observations_for_frame(
        aggregator=aggregator,
        frame_idx=42,
        frame=frame,
        track_sample_interval=1,
        embedding_queue=queue,
    )

    assert tracked_obs.aligned_face is None
    assert queue.pending == []


def test_skip_observation_with_existing_aligned_face(monkeypatch):
    detected_obs = make_obs(
        frame_idx=42,
        track_id=7,
        bbox=(112, 68, 212, 168),
        source=Source.DETECTED,
        landmarks=np.ones((5, 2), dtype=np.float32),
        aligned_face=np.ones((112, 112, 3), dtype=np.uint8),
    )

    track = DummyTrack(track_id=7, observations=[detected_obs])
    aggregator = DummyAggregator(tracks=[track], observations_for_frame=[detected_obs])
    queue = DummyQueue()
    frame = np.zeros((256, 256, 3), dtype=np.uint8)

    def fake_align_face(frame_bgr, landmarks):
        raise AssertionError("alignment should not be attempted when aligned_face already exists")

    monkeypatch.setattr(tas, "align_face_for_arcface", fake_align_face)

    tas._maybe_enqueue_track_embedding_observations_for_frame(
        aggregator=aggregator,
        frame_idx=42,
        frame=frame,
        track_sample_interval=1,
        embedding_queue=queue,
    )

    assert len(queue.pending) == 0

def test_enqueue_tracked_observation_only_when_track_local_index_matches_interval(monkeypatch):
    history = [
        make_obs(
            frame_idx=38,
            track_id=7,
            bbox=(100, 50, 200, 150),
            source=Source.TRACKED,
            landmarks=np.ones((5, 2), dtype=np.float32),
        ),
        make_obs(
            frame_idx=39,
            track_id=7,
            bbox=(101, 51, 201, 151),
            source=Source.TRACKED,
            landmarks=np.ones((5, 2), dtype=np.float32),
        ),
        make_obs(
            frame_idx=40,
            track_id=7,
            bbox=(102, 52, 202, 152),
            source=Source.TRACKED,
            landmarks=np.ones((5, 2), dtype=np.float32),
        ),
        make_obs(
            frame_idx=41,
            track_id=7,
            bbox=(103, 53, 203, 153),
            source=Source.TRACKED,
            landmarks=np.ones((5, 2), dtype=np.float32),
        ),
    ]
    tracked_obs = make_obs(
        frame_idx=42,
        track_id=7,
        bbox=(104, 54, 204, 154),
        source=Source.TRACKED,
        landmarks=np.ones((5, 2), dtype=np.float32),
    )

    track = DummyTrack(track_id=7, observations=history + [tracked_obs])
    aggregator = DummyAggregator(tracks=[track], observations_for_frame=[tracked_obs])
    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    aligned_face = np.ones((112, 112, 3), dtype=np.uint8)

    def fake_align_face(frame_bgr, landmarks):
        return aligned_face

    monkeypatch.setattr(tas, "align_face_for_arcface", fake_align_face)

    queue = DummyQueue()
    tas._maybe_enqueue_track_embedding_observations_for_frame(
        aggregator=aggregator,
        frame_idx=42,
        frame=frame,
        track_sample_interval=3,  # track_local_index is 4, so do not enqueue
        embedding_queue=queue,
    )

    assert queue.pending == []
    assert tracked_obs.aligned_face is None

    queue = DummyQueue()
    tas._maybe_enqueue_track_embedding_observations_for_frame(
        aggregator=aggregator,
        frame_idx=42,
        frame=frame,
        track_sample_interval=2,  # 4 % 2 == 0, so enqueue
        embedding_queue=queue,
    )

    assert len(queue.pending) == 1
    assert queue.pending[0] is tracked_obs
    assert tracked_obs.aligned_face is aligned_face


def test_enqueue_tracked_observation_uses_latest_matching_track_history_position(monkeypatch):
    old_obs = make_obs(
        frame_idx=10,
        track_id=7,
        bbox=(0, 0, 100, 100),
        source=Source.TRACKED,
        landmarks=np.ones((5, 2), dtype=np.float32),
    )
    newer_obs = make_obs(
        frame_idx=41,
        track_id=7,
        bbox=(100, 50, 200, 150),
        source=Source.TRACKED,
        landmarks=np.ones((5, 2), dtype=np.float32),
    )
    tracked_obs = make_obs(
        frame_idx=42,
        track_id=7,
        bbox=(112, 68, 212, 168),
        source=Source.TRACKED,
        landmarks=np.ones((5, 2), dtype=np.float32),
    )

    track = DummyTrack(track_id=7, observations=[old_obs, newer_obs, tracked_obs])
    aggregator = DummyAggregator(tracks=[track], observations_for_frame=[tracked_obs])
    queue = DummyQueue()
    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    aligned_face = np.ones((112, 112, 3), dtype=np.uint8)

    call_count = {"n": 0}

    def fake_align_face(frame_bgr, landmarks):
        call_count["n"] += 1
        return aligned_face

    monkeypatch.setattr(tas, "align_face_for_arcface", fake_align_face)

    tas._maybe_enqueue_track_embedding_observations_for_frame(
        aggregator=aggregator,
        frame_idx=42,
        frame=frame,
        track_sample_interval=2,  # track_local_index is 2, so enqueue
        embedding_queue=queue,
    )

    assert call_count["n"] == 1
    assert len(queue.pending) == 1
    assert queue.pending[0] is tracked_obs
    assert tracked_obs.aligned_face is aligned_face


def test_do_not_enqueue_tracked_observation_when_latest_matching_history_lacks_landmarks(monkeypatch):
    earlier_obs = make_obs(
        frame_idx=40,
        track_id=7,
        bbox=(99, 49, 199, 149),
        source=Source.TRACKED,
        landmarks=np.ones((5, 2), dtype=np.float32),
    )
    latest_prior_obs = make_obs(
        frame_idx=41,
        track_id=7,
        bbox=(100, 50, 200, 150),
        source=Source.TRACKED,
        landmarks=None,
    )
    tracked_obs = make_obs(
        frame_idx=42,
        track_id=7,
        bbox=(112, 68, 212, 168),
        source=Source.TRACKED,
        landmarks=None,
    )

    track = DummyTrack(track_id=7, observations=[earlier_obs, latest_prior_obs, tracked_obs])
    aggregator = DummyAggregator(tracks=[track], observations_for_frame=[tracked_obs])
    queue = DummyQueue()
    frame = np.zeros((256, 256, 3), dtype=np.uint8)

    def fake_align_face(frame_bgr, landmarks):
        raise AssertionError("alignment should not be attempted without landmarks")

    monkeypatch.setattr(tas, "align_face_for_arcface", fake_align_face)

    tas._maybe_enqueue_track_embedding_observations_for_frame(
        aggregator=aggregator,
        frame_idx=42,
        frame=frame,
        track_sample_interval=1,
        embedding_queue=queue,
    )

    assert tracked_obs.aligned_face is None
    assert queue.pending == []


def test_do_not_enqueue_other_track_observation_even_if_earlier_matching_interval_exists(monkeypatch):
    track7_prev = make_obs(
        frame_idx=41,
        track_id=7,
        bbox=(100, 50, 200, 150),
        source=Source.TRACKED,
        landmarks=np.ones((5, 2), dtype=np.float32),
    )
    track7_curr = make_obs(
        frame_idx=42,
        track_id=7,
        bbox=(112, 68, 212, 168),
        source=Source.TRACKED,
        landmarks=np.ones((5, 2), dtype=np.float32),
    )
    track9_prev = make_obs(
        frame_idx=41,
        track_id=9,
        bbox=(300, 100, 380, 180),
        source=Source.TRACKED,
        landmarks=np.ones((5, 2), dtype=np.float32),
    )
    track9_curr = make_obs(
        frame_idx=42,
        track_id=9,
        bbox=(304, 104, 384, 184),
        source=Source.TRACKED,
        landmarks=np.ones((5, 2), dtype=np.float32),
    )

    track7 = DummyTrack(track_id=7, observations=[track7_prev, track7_curr])
    track9 = DummyTrack(track_id=9, observations=[track9_prev, track9_curr])

    aggregator = DummyAggregator(
        tracks=[track7, track9],
        observations_for_frame=[track7_curr, track9_curr],
    )
    queue = DummyQueue()
    frame = np.zeros((256, 256, 3), dtype=np.uint8)

    def fake_align_face(frame_bgr, landmarks):
        raise AssertionError("alignment should not be attempted off sampling interval")

    monkeypatch.setattr(tas, "align_face_for_arcface", fake_align_face)

    tas._maybe_enqueue_track_embedding_observations_for_frame(
        aggregator=aggregator,
        frame_idx=42,
        frame=frame,
        track_sample_interval=3,  # both current obs have track_local_index 1
        embedding_queue=queue,
    )

    assert queue.pending == []
    assert track7_curr.aligned_face is None
    assert track9_curr.aligned_face is None