import numpy as np

from facekit.common.obs_consts import Source
from facekit.tracking.face_structures import FaceObservation


def _mk_aligned_face(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(112, 112, 3), dtype=np.uint8)


class FakeEmbedder:
    def __init__(self):
        self.calls = []

    def get_embedding_batch(self, aligned_faces):
        # record batch size and return deterministic embeddings
        self.calls.append(len(aligned_faces))
        out = []
        for i, _ in enumerate(aligned_faces):
            vec = np.zeros((8,), dtype=np.float32)
            vec[i % 8] = 1.0
            out.append(vec)
        return out


def test_aligned_face_queue_flushes_when_over_limit_and_clears_aligned_faces():
    """Bounded-memory behavior: flush mid-shot and clear aligned_face once embedded."""

    # Import here so the test fails loudly until the production class exists.
    from facekit.embedding.embedding_queue import AlignedFaceEmbeddingQueue

    embedder = FakeEmbedder()
    q = AlignedFaceEmbeddingQueue(max_pending=3)

    obs = []
    for frame_idx in range(5):
        o = FaceObservation(
            frame_idx=frame_idx,
            source=Source.DETECTED,
            track_id=1,
            aligned_face=_mk_aligned_face(frame_idx),
            landmarks=[(1.0, 2.0)] * 5,
        )
        obs.append(o)
        q.enqueue(o)
        q.maybe_flush(embedder)

    # We expect one flush at 3, and then a final flush to drain the remaining 2.
    q.flush(embedder)

    assert embedder.calls == [3, 2]

    for o in obs:
        assert o.embedding is not None
        assert o.aligned_face is None


def test_aligned_face_queue_does_not_reembed_if_embedding_already_present():
    from facekit.embedding.embedding_queue import AlignedFaceEmbeddingQueue

    embedder = FakeEmbedder()
    q = AlignedFaceEmbeddingQueue(max_pending=10)

    o1 = FaceObservation(
        frame_idx=1,
        source=Source.DETECTED,
        track_id=1,
        aligned_face=_mk_aligned_face(1),
        landmarks=[(1.0, 2.0)] * 5,
        embedding=np.ones((8,), dtype=np.float32),
    )
    o2 = FaceObservation(
        frame_idx=2,
        source=Source.DETECTED,
        track_id=1,
        aligned_face=_mk_aligned_face(2),
        landmarks=[(1.0, 2.0)] * 5,
        embedding=None,
    )

    q.enqueue(o1)
    q.enqueue(o2)
    q.flush(embedder)

    # Only o2 should have been embedded.
    assert embedder.calls == [1]
    assert o1.embedding is not None
    assert o1.aligned_face is not None  # queue should not touch already-embedded obs
    assert o2.embedding is not None
    assert o2.aligned_face is None
