import numpy as np

from facekit.embedding.embedding_queue import AlignedFaceEmbeddingQueue
from facekit.embedding.track_embedding_queueing import (
    maybe_enqueue_track_observation_for_embedding,
)
from facekit.tracking.face_structures import FaceObservation, Source


class DummyQueue:
    def __init__(self) -> None:
        self.pending = []

    def enqueue(self, item) -> None:
        self.pending.append(item)


class DummyEmbedder:
    def __init__(self, outputs):
        self.outputs = np.asarray(outputs, dtype=np.float32)

    def get_embedding_batch(self, aligned_faces, max_batch_size):
        assert len(aligned_faces) == self.outputs.shape[0]
        assert max_batch_size == 4
        return self.outputs


def make_observation(*, source, landmarks=None):
    return FaceObservation(
        frame_idx=42,
        track_id=7,
        bbox=(10, 20, 110, 120),
        embedding=None,
        confidence=0.9,
        aligned_face=None,
        source=source,
        landmarks=landmarks,
    )


def test_sampled_observation_with_successful_alignment_sets_aligned_face_and_enqueues():
    queue = DummyQueue()

    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    landmarks = np.array(
        [
            [80.0, 90.0],
            [176.0, 90.0],
            [128.0, 128.0],
            [96.0, 176.0],
            [160.0, 176.0],
        ],
        dtype=np.float32,
    )
    aligned_face = np.ones((112, 112, 3), dtype=np.uint8)

    obs = make_observation(source=Source.DETECTED, landmarks=landmarks)

    def fake_align_face_fn(frame_bgr, face_landmarks):
        assert frame_bgr is frame
        assert face_landmarks is landmarks
        return aligned_face

    enqueued = maybe_enqueue_track_observation_for_embedding(
        observation=obs,
        track_local_index=6,
        track_sample_interval=5,
        frame=frame,
        align_face_fn=fake_align_face_fn,
        embedding_queue=queue,
    )

    assert enqueued is True
    assert obs.aligned_face is aligned_face
    assert len(queue.pending) == 1
    assert queue.pending[0] is obs


def test_non_sampled_observation_is_not_aligned_or_enqueued():
    queue = DummyQueue()

    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    landmarks = np.zeros((5, 2), dtype=np.float32)

    obs = make_observation(source=Source.TRACKED, landmarks=landmarks)

    def fake_align_face_fn(frame_bgr, face_landmarks):
        raise AssertionError(
            "alignment should not be attempted for non-sampled observations"
        )

    enqueued = maybe_enqueue_track_observation_for_embedding(
        observation=obs,
        track_local_index=3,
        track_sample_interval=5,
        frame=frame,
        align_face_fn=fake_align_face_fn,
        embedding_queue=queue,
    )

    assert enqueued is False
    assert obs.aligned_face is None
    assert queue.pending == []


def test_alignment_failure_does_not_enqueue_observation():
    queue = DummyQueue()

    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    landmarks = np.zeros((5, 2), dtype=np.float32)

    obs = make_observation(source=Source.TRACKED, landmarks=landmarks)

    def fake_align_face_fn(frame_bgr, face_landmarks):
        return None

    enqueued = maybe_enqueue_track_observation_for_embedding(
        observation=obs,
        track_local_index=5,
        track_sample_interval=5,
        frame=frame,
        align_face_fn=fake_align_face_fn,
        embedding_queue=queue,
    )

    assert enqueued is False
    assert obs.aligned_face is None
    assert queue.pending == []


def test_missing_landmarks_does_not_enqueue_observation():
    queue = DummyQueue()

    frame = np.zeros((256, 256, 3), dtype=np.uint8)

    obs = make_observation(source=Source.TRACKED, landmarks=None)

    def fake_align_face_fn(frame_bgr, face_landmarks):
        raise AssertionError("alignment should not be attempted without landmarks")

    enqueued = maybe_enqueue_track_observation_for_embedding(
        observation=obs,
        track_local_index=5,
        track_sample_interval=5,
        frame=frame,
        align_face_fn=fake_align_face_fn,
        embedding_queue=queue,
    )

    assert enqueued is False
    assert obs.aligned_face is None
    assert queue.pending == []


def test_queue_flush_populates_embedding_and_clears_aligned_face_on_observation():
    obs = make_observation(source=Source.TRACKED, landmarks=None)
    obs.aligned_face = np.ones((112, 112, 3), dtype=np.uint8)

    queue = AlignedFaceEmbeddingQueue(
        embedder=DummyEmbedder([[1.0, 2.0, 3.0]]),
        max_pending=8,
        max_batch_size=4,
    )

    queue.enqueue(obs)
    queue.flush()

    assert np.allclose(obs.embedding, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert obs.aligned_face is None


def test_multiple_queued_observations_receive_correct_embeddings():
    obs_a = make_observation(source=Source.DETECTED, landmarks=None)
    obs_b = make_observation(source=Source.TRACKED, landmarks=None)

    obs_a.aligned_face = np.ones((112, 112, 3), dtype=np.uint8)
    obs_b.aligned_face = np.full((112, 112, 3), 2, dtype=np.uint8)

    queue = AlignedFaceEmbeddingQueue(
        embedder=DummyEmbedder(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
        max_pending=8,
        max_batch_size=4,
    )

    queue.enqueue(obs_a)
    queue.enqueue(obs_b)
    queue.flush()

    assert np.allclose(obs_a.embedding, np.array([1.0, 0.0], dtype=np.float32))
    assert np.allclose(obs_b.embedding, np.array([0.0, 1.0], dtype=np.float32))
    assert obs_a.aligned_face is None
    assert obs_b.aligned_face is None