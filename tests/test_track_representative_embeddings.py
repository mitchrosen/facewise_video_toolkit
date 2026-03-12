import numpy as np

from facekit.embedding.embedding_selection import (
    TrackEmbeddingSample,
    select_consistent_embedding_subset,
)

def _norm(v):
    arr = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(arr)
    if n == 0:
        raise ValueError("zero vector not allowed in test")
    return arr / n


def _sample(frame_idx: int, emb, track_local_index: int | None = None) -> TrackEmbeddingSample:
    if track_local_index is None:
        track_local_index = frame_idx
    return TrackEmbeddingSample(
        frame_idx=frame_idx,
        track_local_index=track_local_index,
        source="tracking",
        embedding=_norm(emb),
        quality_score=None,
    )


def _frame_ids(samples):
    return [s.frame_idx for s in samples]


def test_select_representative_embeddings_uses_all_when_one_sample():
    samples = [
        _sample(10, [1.0, 0.0, 0.0]),
    ]

    chosen = select_consistent_embedding_subset(samples)

    assert _frame_ids(chosen) == [10]


def test_select_representative_embeddings_uses_all_when_two_samples():
    samples = [
        _sample(10, [1.0, 0.0, 0.0]),
        _sample(20, [0.99, 0.01, 0.0]),
    ]

    chosen = select_consistent_embedding_subset(samples)

    assert _frame_ids(chosen) == [10, 20]


def test_select_representative_embeddings_uses_all_when_three_samples():
    samples = [
        _sample(10, [1.0, 0.0, 0.0]),
        _sample(20, [0.99, 0.01, 0.0]),
        _sample(30, [0.98, 0.02, 0.0]),
    ]

    chosen = select_consistent_embedding_subset(samples)

    assert _frame_ids(chosen) == [10, 20, 30]


def test_select_representative_embeddings_discards_all_outside_best_similarity_cluster():
    samples = [
        _sample(10, [1.00, 0.00, 0.00]),
        _sample(20, [0.99, 0.01, 0.00]),
        _sample(30, [0.98, 0.02, 0.00]),
        _sample(40, [0.97, 0.03, 0.00]),
        _sample(50, [0.00, 1.00, 0.00]),
        _sample(60, [0.00, 0.00, 1.00]),
    ]

    chosen = select_consistent_embedding_subset(samples)

    assert _frame_ids(chosen) == [10, 20, 30, 40]


def test_select_representative_embeddings_may_discard_more_than_two_samples():
    samples = [
        _sample(10, [1.00, 0.00, 0.00]),
        _sample(20, [0.99, 0.01, 0.00]),
        _sample(30, [0.98, 0.02, 0.00]),
        _sample(40, [0.00, 1.00, 0.00]),
        _sample(50, [0.00, 0.00, 1.00]),
        _sample(60, [-1.00, 0.00, 0.00]),
        _sample(70, [0.00, -1.00, 0.00]),
        _sample(80, [0.00, 0.00, -1.00]),
    ]

    chosen = select_consistent_embedding_subset(samples)

    assert _frame_ids(chosen) == [10, 20, 30]


def test_select_representative_embeddings_prefers_larger_consistent_cluster_over_smaller_tighter_pair():
    samples = [
        _sample(10, [1.000, 0.000, 0.000]),
        _sample(20, [0.999, 0.001, 0.000]),
        _sample(30, [0.850, 0.527, 0.000]),
        _sample(40, [0.845, 0.535, 0.000]),
        _sample(50, [0.840, 0.543, 0.000]),
        _sample(60, [0.835, 0.550, 0.000]),
    ]

    chosen = select_consistent_embedding_subset(samples)

    assert len(chosen) == 4
    assert _frame_ids(chosen) == [30, 40, 50, 60]


def test_select_representative_embeddings_is_deterministic_for_same_inputs():
    samples = [
        _sample(10, [1.00, 0.00, 0.00]),
        _sample(20, [0.99, 0.01, 0.00]),
        _sample(30, [0.98, 0.02, 0.00]),
        _sample(40, [0.97, 0.03, 0.00]),
        _sample(50, [0.00, 1.00, 0.00]),
        _sample(60, [0.00, 0.00, 1.00]),
    ]

    chosen_a = select_consistent_embedding_subset(samples)
    chosen_b = select_consistent_embedding_subset(samples)

    assert _frame_ids(chosen_a) == _frame_ids(chosen_b)


def test_select_representative_embeddings_ignores_samples_without_embeddings():
    samples = [
        TrackEmbeddingSample(
            frame_idx=10,
            track_local_index=0,
            source="detection",
            embedding=_norm([1.0, 0.0, 0.0]),
            quality_score=None,
        ),
        TrackEmbeddingSample(
            frame_idx=20,
            track_local_index=10,
            source="tracking",
            embedding=None,
            quality_score=None,
        ),
        TrackEmbeddingSample(
            frame_idx=30,
            track_local_index=20,
            source="tracking",
            embedding=_norm([0.99, 0.01, 0.0]),
            quality_score=None,
        ),
    ]

    chosen = select_consistent_embedding_subset(samples)

    assert _frame_ids(chosen) == [10, 30]


def test_select_representative_embeddings_returns_empty_when_no_valid_embeddings():
    samples = [
        TrackEmbeddingSample(
            frame_idx=10,
            track_local_index=0,
            source="detection",
            embedding=None,
            quality_score=None,
        ),
        TrackEmbeddingSample(
            frame_idx=20,
            track_local_index=10,
            source="tracking",
            embedding=None,
            quality_score=None,
        ),
    ]

    chosen = select_consistent_embedding_subset(samples)

    assert chosen == []