import numpy as np
import pytest
from unittest.mock import patch
import torch
from facekit.tracking.tracking_resolution import GlobalIdentityResolver
from facekit.tracking.face_structures import FaceTrack

# -------------------------------
# Helper Functions
# -------------------------------

def make_vector(angle_rad=0.0, noise=0.0, seed=None, size=512):
    """
    Create a normalized vector with a specified angle from the base vector.

    Args:
        angle_rad (float): Desired angle from the base axis (in radians).
        noise (float): Stddev of Gaussian noise to add after constructing the angle.
        seed (int): Random seed for reproducibility.
        size (int): Dimensionality of the vector.

    Returns:
        np.ndarray: Normalized embedding vector.
    """
    if seed is not None:
        np.random.seed(seed)

    v = np.zeros(size, dtype=np.float32)
    v[0] = np.cos(angle_rad)
    if size > 1:
        v[1] = np.sin(angle_rad)

    if noise > 0:
        v += np.random.normal(0, noise, size).astype(np.float32)

    v /= np.linalg.norm(v)
    return v


def make_track(track_id, embedding, shot_id=0):
    track = FaceTrack(shot_id=shot_id, track_id=track_id)
    track.embeddings = [embedding]
    return track


# -------------------------------
# Tests
# -------------------------------

def test_clusters_get_same_global_id():
    """Tracks forming two separate clusters should get two distinct global IDs."""
    resolver = GlobalIdentityResolver(embedding_threshold=0.8)
    # Cluster A
    emb_a = make_vector(angle_rad=0, seed=1)
    emb_b = make_vector(angle_rad=np.arccos(0.99), seed=2)  # sim ≈ 0.99
    # Cluster B
    emb_c = make_vector(angle_rad=np.arccos(0.6), seed=3)   # sim ≈ 0.6
    emb_d = make_vector(angle_rad=np.arccos(0.61), seed=4)  # sim ≈ 0.61

    tracks = [make_track(0, emb_a), make_track(1, emb_b),
              make_track(2, emb_c), make_track(3, emb_d)]

    resolver.resolve_global_ids(tracks, start_id=0)
    ids = {t.global_id for t in tracks}
    assert len(ids) == 2, f"Expected 2 clusters but got {len(ids)}: {ids}"


def test_threshold_boundary_behavior():
    """At the threshold boundary: tracks at and above threshold should merge, below should not in direct pair,
    but graph connectivity may cause chaining merges."""
    threshold = 0.95
    resolver = GlobalIdentityResolver(embedding_threshold=threshold)

    emb_a = make_vector(angle_rad=0, seed=10)
    emb_b = make_vector(angle_rad=np.arccos(0.94), seed=11)  # below threshold to A
    emb_c = make_vector(angle_rad=np.arccos(0.95), seed=12)  # exactly threshold
    emb_d = make_vector(angle_rad=np.arccos(0.96), seed=13)  # above threshold

    # Direct pair: A and B should not merge
    tracks = [make_track(0, emb_a), make_track(1, emb_b)]
    resolver.resolve_global_ids(tracks, start_id=0)
    assert tracks[0].global_id != tracks[1].global_id, (
        f"B, {tracks[1].global_id}, should not merge with A, {tracks[0].global_id}"
    )

    # Direct pair: A and C should merge
    tracks = [make_track(0, emb_a.copy()), make_track(1, emb_c.copy())]
    resolver.resolve_global_ids(tracks, start_id=0)
    assert tracks[0].global_id == tracks[1].global_id, (
        f"C should merge with A"
    )

    # Direct pair: A and D should merge
    tracks = [make_track(0, emb_a.copy()), make_track(1, emb_d.copy())]
    resolver.resolve_global_ids(tracks, start_id=0)
    assert tracks[0].global_id == tracks[1].global_id, (
        f"D should merge with A"
    )

    # All together: graph connectivity should collapse them into 1 cluster
    tracks = [
        make_track(0, emb_a),
        make_track(1, emb_b),
        make_track(2, emb_c),
        make_track(3, emb_d)
    ]
    resolver.resolve_global_ids(tracks, start_id=0)
    assert len({t.global_id for t in tracks}) == 1, f"Expected 1 cluster by connectivity, got {[t.global_id for t in tracks]}"

def test_noise_effect_on_merging():
    """Adding noise should reduce similarity enough to split clusters if threshold is high."""
    resolver = GlobalIdentityResolver(embedding_threshold=0.99)
    base_angle = 0
    emb_ref = make_vector(angle_rad=base_angle, seed=100)
    emb_noisy = make_vector(angle_rad=base_angle, noise=0.1, seed=101)

    tracks = [make_track(0, emb_ref), make_track(1, emb_noisy)]
    resolver.resolve_global_ids(tracks, start_id=0)

    # With noise and high threshold, they should NOT merge
    assert tracks[0].global_id != tracks[1].global_id, (
        f"Expected separate clusters due to noise, got same ID {tracks[0].global_id}"
    )


def test_all_tracks_get_ids_even_if_no_embeddings():
    """Tracks without embeddings should still receive unique IDs."""
    resolver = GlobalIdentityResolver(embedding_threshold=0.8)
    t1 = FaceTrack(shot_id=0, track_id=0)  # no embeddings
    t2 = FaceTrack(shot_id=0, track_id=1)  # no embeddings

    tracks = [t1, t2]
    resolver.resolve_global_ids(tracks, start_id=10)

    ids = {t.global_id for t in tracks}
    assert len(ids) == 2, f"Expected unique IDs for tracks without embeddings, got {ids}"


def test_cluster_assignment_in_mixed_scenario():
    """Complex scenario with two clusters and an outlier."""
    resolver = GlobalIdentityResolver(embedding_threshold=0.85)
    # Cluster 1
    t1 = make_track(0, make_vector(angle_rad=0, seed=1))
    t2 = make_track(1, make_vector(angle_rad=np.arccos(0.98), seed=2))

    # Cluster 2
    t3 = make_track(2, make_vector(angle_rad=np.arccos(0.70), seed=3))
    t4 = make_track(3, make_vector(angle_rad=np.arccos(0.69), seed=4))

    # Outlier
    t5 = make_track(4, make_vector(angle_rad=np.arccos(0.0), seed=5))

    tracks = [t1, t2, t3, t4, t5]
    resolver.resolve_global_ids(tracks, start_id=0)

    ids = [t.global_id for t in tracks]
    assert len(set(ids)) == 3, f"Expected 3 clusters, got {len(set(ids))}: {ids}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_global_id_consistency_cpu_vs_gpu_with_mock():
    resolver = GlobalIdentityResolver(embedding_threshold=0.8)

    # Create two identical track sets
    tracks_gpu = [make_track(i, make_vector(angle_rad=0)) for i in range(4)]
    tracks_cpu = [make_track(i, make_vector(angle_rad=0)) for i in range(4)]

    # First run: actual GPU mode
    resolver.resolve_global_ids(tracks_gpu, start_id=0)

    # Second run: mock CUDA unavailable -> force CPU mode
    with patch("torch.cuda.is_available", return_value=False):
        resolver.resolve_global_ids(tracks_cpu, start_id=0)

    cpu_ids = [t.global_id for t in tracks_cpu]
    gpu_ids = [t.global_id for t in tracks_gpu]

    assert cpu_ids == gpu_ids, f"Mismatch between CPU and GPU modes: CPU={cpu_ids}, GPU={gpu_ids}"

def test_resolver_raises_when_cuda_requested_but_unavailable(monkeypatch):
    monkeypatch.setattr(torch, "cuda", type("X", (), {"is_available": lambda: False}))
    with pytest.raises(RuntimeError):
        GlobalIdentityResolver(device="cuda")
