import numpy as np

from facekit.common.obs_consts import Source
from facekit.embedding.embedding_selection import TrackEmbeddingSample
from facekit.tracking.face_structures import FaceTrack

def _norm(v):
    arr = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(arr))
    if n == 0.0:
        raise ValueError("zero vector not allowed in test")
    return arr / n

def _emb(seed: int) -> np.ndarray:
    v = np.zeros(512, dtype=np.float32)
    v[seed] = 1.0
    return v

def _sample(
    frame_idx: int,
    emb: np.array,
    *,
    track_local_index: int | None = None,
    source=Source.TRACKED,
) -> TrackEmbeddingSample:
    if track_local_index is None:
        track_local_index = frame_idx
    return TrackEmbeddingSample(
        frame_idx=int(frame_idx),
        track_local_index=int(track_local_index),
        source=source,
        embedding=None if emb is None else _norm(emb),
        quality_score=None,
    )

def _mean_embedding(samples):
    return np.mean(
        np.stack([s.embedding for s in samples if s.embedding is not None]),
        axis=0,
    )

def test_finalize_embedding_representation_selects_subset_and_computes_representative_embedding():
    """
    Finalization should retain a consistent subset of samples and compute the
    representative embedding from the retained subset.
    """
    track = FaceTrack(shot_id=1, track_id=7)
    track.embedding_samples = [
        _sample(10, [1.00, 0.00, 0.00], source=Source.DETECTED),
        _sample(20, [0.99, 0.01, 0.00], source=Source.TRACKED),
        _sample(30, [0.98, 0.02, 0.00], source=Source.TRACKED),
        _sample(40, [0.97, 0.03, 0.00], source=Source.TRACKED),
        _sample(50, [0.00, 1.00, 0.00], source=Source.TRACKED),  # outlier
    ]

    track.finalize_embedding_representation()

    assert [s.frame_idx for s in track.selected_embedding_samples] == [10, 20, 30, 40]
    assert np.allclose(
        track.representative_embedding,
        _mean_embedding(track.selected_embedding_samples),
    )
    assert track.embedding_stable is True


def test_finalize_embedding_representation_preserves_selected_sample_metadata():
    """
    Selected samples should retain the metadata needed for later auditing and
    debugging.
    """
    track = FaceTrack(shot_id=2, track_id=9)
    track.embedding_samples = [
        _sample(100, [1.00, 0.00, 0.00], track_local_index=0, source=Source.DETECTED),
        _sample(110, [0.99, 0.01, 0.00], track_local_index=10, source=Source.TRACKED),
        _sample(120, [0.98, 0.02, 0.00], track_local_index=20, source=Source.TRACKED),
        _sample(130, [0.00, 1.00, 0.00], track_local_index=30, source=Source.TRACKED),
    ]

    track.finalize_embedding_representation()

    assert [
        (s.frame_idx, s.track_local_index, s.source)
        for s in track.selected_embedding_samples
    ] == [
        (100, 0, Source.DETECTED),
        (110, 10, Source.TRACKED),
        (120, 20, Source.TRACKED),
    ]


def test_finalize_embedding_representation_marks_track_unstable_when_no_valid_embeddings():
    """
    A track with no usable embeddings should not crash finalization, should not
    invent a representative embedding, and should be marked unstable.
    """
    track = FaceTrack(shot_id=3, track_id=11)
    track.embedding_samples = [
        _sample(10, None, track_local_index=0, source=Source.DETECTED),
        _sample(20, None, track_local_index=10, source=Source.TRACKED),
    ]

    track.finalize_embedding_representation()

    assert track.selected_embedding_samples == []
    assert track.representative_embedding is None
    assert track.embedding_stable is False


def test_finalize_embedding_representation_uses_selected_subset_not_all_samples():
    """
    The representative embedding should be computed from the retained subset,
    not from all original samples.
    """
    track = FaceTrack(shot_id=4, track_id=13)

    inliers = [
        _sample(10, [1.00, 0.00, 0.00], source=Source.DETECTED),
        _sample(20, [0.99, 0.01, 0.00], source=Source.TRACKED),
        _sample(30, [0.98, 0.02, 0.00], source=Source.TRACKED),
    ]
    outlier = _sample(40, [0.00, 1.00, 0.00], source=Source.TRACKED)

    track.embedding_samples = inliers + [outlier]

    track.finalize_embedding_representation()

    expected_from_selected = _mean_embedding(track.selected_embedding_samples)
    expected_from_all = _mean_embedding(track.embedding_samples)

    assert np.allclose(track.representative_embedding, expected_from_selected)
    assert not np.allclose(track.representative_embedding, expected_from_all)

def test_finalize_embedding_representation_is_deterministic_across_resume_boundary():
    """
    A track finalized after a resume-style split should produce the same selected
    sample frames and representative embedding as a cold run over the same logical
    samples.
    """
    cold_track = FaceTrack(shot_id=1, track_id=1)
    resumed_track = FaceTrack(shot_id=1, track_id=1)

    all_samples = [
        _sample(100, _emb(0), track_local_index=0, source=Source.DETECTED),
        _sample(110, _emb(1), track_local_index=10, source=Source.TRACKED,),
        _sample(120, _emb(2), track_local_index=20, source=Source.TRACKED,),
        _sample(130, _emb(3), track_local_index=30, source=Source.TRACKED,),
    ]

    # Cold run: all samples arrive in one uninterrupted pass.
    for s in all_samples:
        cold_track.add_embedding_sample(s)

    # Resume-style run: pre-boundary samples exist first, then post-boundary samples
    # are attached later after resume.
    for s in all_samples[:2]:
        resumed_track.add_embedding_sample(s)
    for s in all_samples[2:]:
        resumed_track.add_embedding_sample(s)

    cold_track.finalize_embedding_representation()
    resumed_track.finalize_embedding_representation()

    assert [s.frame_idx for s in cold_track.selected_embedding_samples] == [
        s.frame_idx for s in resumed_track.selected_embedding_samples
    ]
    assert [s.track_local_index for s in cold_track.selected_embedding_samples] == [
        s.track_local_index for s in resumed_track.selected_embedding_samples
    ]
    assert np.allclose(
        cold_track.representative_embedding,
        resumed_track.representative_embedding,
    )
    assert cold_track.embedding_stable is True
    assert resumed_track.embedding_stable is True

def test_finalize_embedding_representation_is_stable_when_pre_boundary_samples_are_not_duplicated():
    """
    Re-attaching pre-boundary samples should not change the selected sample set
    or representative embedding for the logical track.
    """
    cold_track = FaceTrack(shot_id=1, track_id=1)
    resumed_track = FaceTrack(shot_id=1, track_id=1)

    all_samples = [
        _sample(100, [1.00, 0.00, 0.00], track_local_index=0, source=Source.DETECTED),
        _sample(110, [0.99, 0.01, 0.00], track_local_index=10, source=Source.TRACKED),
        _sample(120, [0.98, 0.02, 0.00], track_local_index=20, source=Source.TRACKED),
        _sample(130, [0.97, 0.03, 0.00], track_local_index=30, source=Source.TRACKED),
    ]

    # Cold run: one uninterrupted pass.
    for s in all_samples:
        cold_track.add_embedding_sample(s)

    # Resume-style bug: pre-boundary samples get attached twice.
    for s in all_samples[:2]:
        resumed_track.add_embedding_sample(s)
    for s in all_samples[:2]:
        resumed_track.add_embedding_sample(s)
    for s in all_samples[2:]:
        resumed_track.add_embedding_sample(s)

    cold_track.finalize_embedding_representation()
    resumed_track.finalize_embedding_representation()

    assert [s.frame_idx for s in cold_track.selected_embedding_samples] == [
        s.frame_idx for s in resumed_track.selected_embedding_samples
    ]
    assert [s.track_local_index for s in cold_track.selected_embedding_samples] == [
        s.track_local_index for s in resumed_track.selected_embedding_samples
    ]
    assert [s.source for s in cold_track.selected_embedding_samples] == [
        s.source for s in resumed_track.selected_embedding_samples
    ]
    assert np.allclose(
        cold_track.representative_embedding,
        resumed_track.representative_embedding,
    )
    assert cold_track.embedding_stable is True
    assert resumed_track.embedding_stable is True