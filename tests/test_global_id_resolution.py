import numpy as np
import pytest
from unittest.mock import patch
import torch
import random
import math

from facekit.tracking.tracking_resolution import GlobalIdentityResolver
from facekit.tracking.face_structures import FaceTrack, FaceObservation
from facekit.common.obs_consts import Source

# -------------------------------
# Helper Functions
# -------------------------------

def make_vector(angle_rad=0.0, noise=0.0, seed=None, size=512):
    """Create a normalized vector at a given angle, with optional Gaussian noise."""
    if seed is not None:
        np.random.seed(seed)
    v = np.zeros(size, dtype=np.float32)
    v[0] = np.cos(angle_rad)
    if size > 1:
        v[1] = np.sin(angle_rad)
    if noise > 0:
        v += np.random.normal(0, noise, size).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-9
    return v

class DummyObs:
    def __init__(self, frame_idx: int):
        self.frame_idx = frame_idx

def make_track(track_id, embedding, shot_id=0):
    t = FaceTrack(shot_id=shot_id, track_id=track_id)
    t.embeddings = [embedding.astype(np.float32)]
    t.observations = [DummyObs(0)]
    return t

def make_track_with_frames(track_id: int, emb: np.ndarray, shot_id: int, start: int, end: int) -> FaceTrack:
    t = FaceTrack(shot_id=shot_id, track_id=track_id)
    # Minimal observations so first_frame()/last_frame() are well-defined
    t.observations.append(
        FaceObservation(
            frame_idx=start,
            track_id=track_id,
            bbox=(0, 0, 10, 10),
            source=Source.DETECTED,
        )
    )
    t.observations.append(
        FaceObservation(
            frame_idx=end,
            track_id=track_id,
            bbox=(0, 0, 10, 10),
            source=Source.DETECTED,
        )
    )
    t.embeddings = [emb.astype(np.float32)]
    return t

# -------------------------------
# Tests
# -------------------------------

def test_clusters_get_same_global_id():
    """Two tight clusters should yield two distinct global IDs."""
    resolver = GlobalIdentityResolver(embedding_threshold=0.95)
    # Cluster A
    emb_a = make_vector(angle_rad=0, seed=1)
    emb_b = make_vector(angle_rad=np.arccos(0.99), seed=2)     # sim ~ 0.99
    # Cluster B: tight internally, but clearly separated from cluster A.
    # 0.40 / 0.42 rad are ~23° / ~24°, so internal sim is ~0.9998 while
    # cross-cluster sim to angle 0 stays below 0.95.
    emb_c = make_vector(angle_rad=0.40, seed=3)
    emb_d = make_vector(angle_rad=0.42, seed=4)

    # Put the two tight pairs in different shots so the resolver is allowed to merge them.
    tracks = [
        make_track(0, emb_a, shot_id=0),
        make_track(1, emb_b, shot_id=0),
        make_track(2, emb_c, shot_id=1),
        make_track(3, emb_d, shot_id=1),
    ]

    resolver.resolve_global_ids(tracks, start_id=0)
    ids = {t.global_id for t in tracks}
    assert len(ids) == 2, f"Expected 2 clusters but got {len(ids)}: {ids}"

def test_threshold_boundary_behavior():
    """
    At the threshold boundary:
    - Below threshold: no merge.
    - At / above threshold: should merge *when allowed by constraints* (i.e., not same-shot overlapping).
    - Connectivity can collapse to one cluster when constraints allow.
    """
    threshold = 0.95
    resolver = GlobalIdentityResolver(embedding_threshold=threshold)

    emb_a = make_vector(angle_rad=0, seed=10)
    emb_b = make_vector(angle_rad=np.arccos(0.94), seed=11)  # below threshold to A
    emb_c = make_vector(angle_rad=np.arccos(0.95), seed=12)  # exactly threshold
    emb_d = make_vector(angle_rad=np.arccos(0.96), seed=13)  # above threshold

    # A and B: below threshold => no merge.
    # Put them in different shots so only the similarity threshold is being tested.
    tracks = [make_track(0, emb_a, shot_id=0), make_track(1, emb_b, shot_id=1)]
    resolver.resolve_global_ids(tracks, start_id=0)
    assert tracks[0].global_id != tracks[1].global_id

    # A and C: at threshold => merge (different shots so constraint doesn't block)
    tracks = [make_track(0, emb_a.copy(), shot_id=0),
              make_track(1, emb_c.copy(), shot_id=1)]
    resolver.resolve_global_ids(tracks, start_id=0)
    assert tracks[0].global_id == tracks[1].global_id, "At threshold should merge when not overlapping in same shot"

    # A and D: above threshold => merge (different shots)
    tracks = [make_track(0, emb_a.copy(), shot_id=0),
              make_track(1, emb_d.copy(), shot_id=1)]
    resolver.resolve_global_ids(tracks, start_id=0)
    assert tracks[0].global_id == tracks[1].global_id

    # Connectivity collapse permitted when constraints allow
    tracks = [
        make_track(0, emb_a, shot_id=0),
        make_track(1, emb_b, shot_id=0),  # will stay separate from A
        make_track(2, emb_c, shot_id=1),  # can connect to A
        make_track(3, emb_d, shot_id=1)   # can connect to A/C
    ]
    resolver.resolve_global_ids(tracks, start_id=0)
    gids = [t.global_id for t in tracks]
    assert len(set(gids)) in (1, 2), f"Unexpected cluster count with connectivity constraints: {gids}"

def test_noise_effect_on_merging():
    """High threshold + noise should split."""
    resolver = GlobalIdentityResolver(embedding_threshold=0.99)
    emb_ref = make_vector(angle_rad=0, seed=100)
    emb_noisy = make_vector(angle_rad=0, noise=0.1, seed=101)
    # Different shots so only embedding similarity is under test.
    tracks = [make_track(0, emb_ref, shot_id=0), make_track(1, emb_noisy, shot_id=1)]
    resolver.resolve_global_ids(tracks, start_id=0)
    assert tracks[0].global_id != tracks[1].global_id

def test_cluster_assignment_in_mixed_scenario():
    """Two clusters and an outlier."""
    resolver = GlobalIdentityResolver(embedding_threshold=0.92)
    # Put the two clusters in different shots; outlier in its own shot.
    t1 = make_track(0, make_vector(angle_rad=0, seed=1),               shot_id=0)
    t2 = make_track(1, make_vector(angle_rad=np.arccos(0.98), seed=2), shot_id=0)
    # Cluster 2: tight internally, but clearly separated from cluster 1.
    # 0.50 / 0.52 rad give internal sim ~0.9998 and cross-cluster sim ~0.88.
    t3 = make_track(2, make_vector(angle_rad=0.50, seed=3), shot_id=1)
    t4 = make_track(3, make_vector(angle_rad=0.52, seed=4), shot_id=1)
    # Outlier
    t5 = make_track(4, make_vector(angle_rad=np.arccos(0.0),  seed=5), shot_id=2)

    tracks = [t1, t2, t3, t4, t5]
    resolver.resolve_global_ids(tracks, start_id=0)
    ids = [t.global_id for t in tracks]
    assert len(set(ids)) == 3, f"Expected 3 clusters, got {len(set(ids))}: {ids}"

def test_merge_at_threshold_when_not_overlapping():
    thr = 0.95
    resolver = GlobalIdentityResolver(embedding_threshold=thr)
    A = make_vector(angle_rad=0, seed=1)
    C = make_vector(angle_rad=np.arccos(thr), seed=2)
    t0 = make_track_with_frames(0, A, shot_id=0, start=0, end=29)
    t1 = make_track_with_frames(1, C, shot_id=1, start=0, end=29)  # different shot
    resolver.resolve_global_ids([t0, t1], start_id=0)
    assert t0.global_id == t1.global_id

def test_same_shot_overlapping_do_not_merge_at_or_above_threshold():
    thr = 0.95
    resolver = GlobalIdentityResolver(embedding_threshold=thr)
    A = make_vector(angle_rad=0, seed=3)
    C = make_vector(angle_rad=np.arccos(thr), seed=4)
    t0 = make_track_with_frames(0, A, shot_id=7, start=0, end=50)
    t1 = make_track_with_frames(1, C, shot_id=7, start=25, end=75)  # overlap
    resolver.resolve_global_ids([t0, t1], start_id=0)
    assert t0.global_id != t1.global_id

def test_same_shot_non_overlapping_can_merge():
    thr = 0.95
    resolver = GlobalIdentityResolver(embedding_threshold=thr)
    A = make_vector(angle_rad=0, seed=5)
    C = make_vector(angle_rad=np.arccos(thr), seed=6)
    t0 = make_track_with_frames(0, A, shot_id=3, start=0, end=29)
    t1 = make_track_with_frames(1, C, shot_id=3, start=30, end=60)  # non-overlap
    resolver.resolve_global_ids([t0, t1], start_id=0)
    assert t0.global_id == t1.global_id

def test_must_not_link_blocks_bridge():
    thr = 0.95
    resolver = GlobalIdentityResolver(embedding_threshold=thr)
    A = make_vector(angle_rad=0, seed=11)
    B = make_vector(angle_rad=np.arccos(thr), seed=12)
    C = make_vector(angle_rad=np.arccos(thr), seed=13)
    tA = make_track_with_frames(0, A, shot_id=5, start=0, end=60)   # overlaps with tB
    tB = make_track_with_frames(1, B, shot_id=5, start=30, end=90)
    tC = make_track_with_frames(2, C, shot_id=6, start=0, end=30)   # different shot
    resolver.resolve_global_ids([tA, tB, tC], start_id=0)
    assert tA.global_id != tB.global_id
    assert (tA.global_id == tC.global_id) ^ (tB.global_id == tC.global_id)

def test_deterministic_order():
    thr = 0.9
    vecs = [make_vector(angle_rad=0, seed=20+i) for i in range(6)]
    tracks = [make_track_with_frames(i, v, shot_id=i//3, start=10*i, end=10*i+5) for i, v in enumerate(vecs)]
    # First run
    resolver1 = GlobalIdentityResolver(embedding_threshold=thr)
    resolver1.resolve_global_ids(tracks, start_id=0)
    gids1 = [t.global_id for t in tracks]
    # Shuffle and rerun with a fresh resolver
    random.shuffle(tracks)
    resolver2 = GlobalIdentityResolver(embedding_threshold=thr)
    resolver2.resolve_global_ids(tracks, start_id=0)
    gids2 = [t.global_id for t in tracks]
    assert sorted(gids1) == sorted(gids2)

def test_tracks_with_same_segment_id_merge_within_shot():
    thr = 0.95
    resolver = GlobalIdentityResolver(embedding_threshold=thr)

    A = make_vector(angle_rad=0, seed=1)
    # Two tracks, same shot & same segment_id, maybe slightly different embs
    t0 = make_track_with_frames(0, A, shot_id=2, start=0, end=20);  t0.segment_id = 7
    t1 = make_track_with_frames(1, A, shot_id=2, start=25, end=45); t1.segment_id = 7

    resolver.resolve_global_ids([t0, t1], start_id=0)
    assert t0.global_id == t1.global_id, "Same (shot_id, segment_id) must share global_id"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_global_id_consistency_cpu_vs_gpu_with_mock():
    # Make two identical datasets
    tracks_gpu = [make_track(i, make_vector(angle_rad=0, seed=i)) for i in range(4)]
    tracks_cpu = [make_track(i, make_vector(angle_rad=0, seed=i)) for i in range(4)]
    # Resolver under CUDA available (real env)
    resolver_gpu = GlobalIdentityResolver(embedding_threshold=0.8)
    resolver_gpu.resolve_global_ids(tracks_gpu, start_id=0)
    # Resolver with CUDA forced unavailable → CPU
    with patch("torch.cuda.is_available", return_value=False):
        resolver_cpu = GlobalIdentityResolver(embedding_threshold=0.8, device="auto")
        resolver_cpu.resolve_global_ids(tracks_cpu, start_id=0)
    assert [t.global_id for t in tracks_cpu] == [t.global_id for t in tracks_gpu], \
        "Mismatch between CPU and GPU modes"

def test_resolver_raises_when_cuda_requested_but_unavailable(monkeypatch):
    monkeypatch.setattr(torch, "cuda", type("X", (), {"is_available": lambda: False}))
    with pytest.raises(RuntimeError):
        GlobalIdentityResolver(device="cuda")

def test_identity_resolution_prefers_track_representative_embedding():
    """
    Identity resolution should prefer a track's representative_embedding when
    that embedding is marked as stable.

    Tracks may contain multiple observation embeddings collected during
    processing. Once a representative_embedding has been finalized for a track,
    it becomes the canonical identity signal for that track.

    This test constructs a situation where:

      • observation embeddings would NOT cause the tracks to match
      • representative embeddings WOULD cause the tracks to match

    The resolver should therefore merge the tracks based on the
    representative embeddings.
    """

    resolver = GlobalIdentityResolver(embedding_threshold=0.95)

    rep = make_vector(angle_rad=0.0, seed=100)

    obs_a = make_vector(angle_rad=0.60, seed=101)
    obs_b = make_vector(angle_rad=1.20, seed=102)

    t0 = make_track_with_frames(0, obs_a, shot_id=0, start=0, end=20)
    t1 = make_track_with_frames(1, obs_b, shot_id=1, start=0, end=20)

    # Canonical identity signal for each track
    t0.representative_embedding = rep.copy()
    t1.representative_embedding = rep.copy()

    t0.embedding_stable = True
    t1.embedding_stable = True

    # Observation embeddings disagree with the representative identity
    t0.embeddings = [obs_a.astype(np.float32)]
    t1.embeddings = [obs_b.astype(np.float32)]

    resolver.resolve_global_ids([t0, t1], start_id=0)

    assert (
        t0.global_id == t1.global_id
    ), "Stable representative embeddings should drive identity matching"