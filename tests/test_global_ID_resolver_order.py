import math
import random
import numpy as np
import pytest
from types import SimpleNamespace

class FakeObs:
    def __init__(self, frame_idx):
        self.frame_idx = int(frame_idx)

class DummyObs:
    """
    Minimal stub to satisfy GlobalIdentityResolver's expectations:
    - .frame_idx: int
    - .source: optional (used only for audit logging)
    """
    def __init__(self, f, source=None):
        self.frame_idx = int(f)
        self.source = source

class FakeTrack:
    """
    Minimal test double for FaceTrack.

    Must expose:
      - shot_id: int
      - segment_id: int | None
      - track_id: int
      - embeddings: list[np.ndarray]
      - observations: non-empty list with .frame_idx (and optional .source)
      - first_frame()
      - last_frame()
    """
    def __init__(self, shot_id, segment_id, frame_range, embeddings, track_id=None):
        self.shot_id = int(shot_id)
        self.segment_id = segment_id

        # Canonicalize frame_range to (start, end) ints
        start, end = frame_range
        self.frame_range = (int(start), int(end))

        # Embeddings list (may be empty list)
        if embeddings is None:
            self.embeddings = []
        else:
            self.embeddings = list(embeddings)

        # Real tracks always have an int track_id; default to 0 if none provided.
        self.track_id = int(track_id) if track_id is not None else 0

        # Satisfy resolver precondition: every track has >= 1 observation.
        # We don't care about the full sequence here; a single representative
        # observation at the first frame is enough and does not affect the
        # order / overlap semantics tested (which use first_frame/last_frame).
        self.observations = [DummyObs(self.frame_range[0])]

    def first_frame(self):
        return self.frame_range[0]

    def last_frame(self):
        return self.frame_range[1]

# Import the class under test
from facekit.tracking.tracking_resolution import GlobalIdentityResolver

def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return (v / (n + 1e-9)).astype(np.float32)

def _make_emb(center, jitter=0.02, d=512, seed=None):
    """
    Create a unit embedding near `center` (also unit). Cosine similarity stays high.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    base = _unit(center)
    noise = rng.normal(0.0, jitter, size=d).astype(np.float32)
    return _unit(base + noise)

def _proto(d=512, seed=0):
    rng = np.random.default_rng(seed)
    return _unit(rng.normal(0, 1, size=d).astype(np.float32))


# ---------- Tests ----------

def test_ordering_by_earliest_frame_not_input_order():
    """
    Components must be assigned global IDs by earliest absolute frame,
    not by the input ordering of tracks/groups.
    """
    # Two identities with distinct prototypes
    A = _proto(seed=1)
    B = _proto(seed=2)

    # Shot 1: Face A appears first (frames 0..9), shot 2: Face B appears later (frames 20..29)
    tA1 = FakeTrack(shot_id=1, segment_id=0, frame_range=(0, 9),  embeddings=[_make_emb(A, seed=11)])
    tB2 = FakeTrack(shot_id=2, segment_id=0, frame_range=(20, 29), embeddings=[_make_emb(B, seed=21)])

    # Present them in reverse order to try to break determinism
    tracks = [tB2, tA1]

    resolver = GlobalIdentityResolver(embedding_threshold=0.70, device="cpu")
    next_id = resolver.resolve_global_ids(tracks, start_id=0)

    assert next_id == 2
    # The earlier face (A) must get global_id 0, even though it was passed second.
    assert tA1.global_id == 0
    assert tB2.global_id == 1


def test_must_not_link_prevents_merge_when_overlapping_within_same_shot():
    """
    Tracks in the same shot whose time intervals overlap must not be merged,
    even if their embeddings are similar.
    """
    P = _proto(seed=3)

    # Same shot, overlapping frames [10..30] and [25..40] -> overlap
    t1 = FakeTrack(shot_id=5, segment_id=0, frame_range=(10, 30), embeddings=[_make_emb(P, seed=31)])
    t2 = FakeTrack(shot_id=5, segment_id=1, frame_range=(25, 40), embeddings=[_make_emb(P, seed=32)])

    resolver = GlobalIdentityResolver(embedding_threshold=0.70, device="cpu")
    next_id = resolver.resolve_global_ids([t1, t2], start_id=10)

    # Since they overlap in the same shot, they cannot be merged
    assert t1.global_id != t2.global_id
    assert set([t1.global_id, t2.global_id]) == {10, 11}
    assert next_id == 12


def test_input_order_does_not_change_labels_determinism():
    """
    Shuffling input order must not change global ID assignments.
    We check multiple permutations.
    """
    A = _proto(seed=10)
    B = _proto(seed=20)
    C = _proto(seed=30)

    # Earliest frames: A(0..9), then B(5..14), then C(40..49) across different shots
    tA = FakeTrack(shot_id=1, segment_id=0, frame_range=(0, 9),   embeddings=[_make_emb(A, seed=101)], track_id=0)
    tB = FakeTrack(shot_id=1, segment_id=1, frame_range=(5, 14),  embeddings=[_make_emb(B, seed=102)], track_id=1)
    tC = FakeTrack(shot_id=2, segment_id=0, frame_range=(40, 49), embeddings=[_make_emb(C, seed=103)], track_id=0)

    base = [tA, tB, tC]

    # Baseline
    resolver = GlobalIdentityResolver(embedding_threshold=0.70, device="cpu")
    resolver.resolve_global_ids(base, start_id=0)
    baseline = [(tA.global_id, tB.global_id, tC.global_id)]

    # Try some permutations and re-check
    perms = [
        [tA, tC, tB],
        [tB, tA, tC],
        [tB, tC, tA],
        [tC, tA, tB],
        [tC, tB, tA],
    ]
    for perm in perms:
        # Reset ids
        for t in [tA, tB, tC]:
            t.global_id = None
        GlobalIdentityResolver(embedding_threshold=0.70, device="cpu").resolve_global_ids(perm, start_id=0)
        assert (tA.global_id, tB.global_id, tC.global_id) == baseline[0], f"Permutation broke determinism: {perm}"


def test_resume_invariance_vs_uninterrupted_ordering():
    """
    Simulate a golden uninterrupted run and a resumed run where
    later-shot components may appear earlier in the input list.
    The global IDs must match (earliest-frame ordering).
    """
    G = _proto(seed=123)  # one person across two shots
    # Shot 1: single face from frames 0..9
    shot1_t0 = FakeTrack(shot_id=1, segment_id=0, frame_range=(0, 9), embeddings=[_make_emb(G, seed=1001)], track_id=0)
    # Shot 2: same identity later from 100..119 (separate segment within shot 2)
    shot2_t0 = FakeTrack(shot_id=2, segment_id=0, frame_range=(100, 119), embeddings=[_make_emb(G, seed=1002)], track_id=0)

    # Uninterrupted/golden ordering (as they appear chronologically):
    golden_tracks = [shot1_t0, shot2_t0]
    GlobalIdentityResolver(embedding_threshold=0.70, device="cpu").resolve_global_ids(golden_tracks, start_id=0)
    golden_map = {(t.shot_id, t.segment_id): t.global_id for t in golden_tracks}

    # “Resumed” case: imagine we rehydrated, then the post-resume shot2 group comes
    # first in the input to the resolver (e.g., internal processing order), followed by shot1.
    for t in [shot1_t0, shot2_t0]:
        t.global_id = None  # reset

    resumed_tracks = [shot2_t0, shot1_t0]
    GlobalIdentityResolver(embedding_threshold=0.70, device="cpu").resolve_global_ids(resumed_tracks, start_id=0)
    resumed_map = {(t.shot_id, t.segment_id): t.global_id for t in resumed_tracks}

    assert resumed_map == golden_map, f"Resume drift: {resumed_map} != {golden_map}"


def test_no_embedding_leftovers_are_labeled_after_components_in_earliest_order():
    """
    Tracks that end up with no embeddings should still get global IDs,
    assigned after the clustered groups and in earliest-frame order.
    """
    A = _proto(seed=77)
    # Two tracks with embeddings (one identity)
    tA1 = FakeTrack(shot_id=1, segment_id=0, frame_range=(0, 9),  embeddings=[_make_emb(A, seed=7701)], track_id=0)
    tA2 = FakeTrack(shot_id=2, segment_id=0, frame_range=(50, 59), embeddings=[_make_emb(A, seed=7702)], track_id=0)

    # Leftovers without embeddings at frames 20..24 and 30..34
    tX = FakeTrack(shot_id=1, segment_id=None, frame_range=(20, 24), embeddings=[], track_id=1)
    tY = FakeTrack(shot_id=1, segment_id=None, frame_range=(30, 34), embeddings=None, track_id=2)

    tracks = [tY, tA2, tX, tA1]  # shuffled
    GlobalIdentityResolver(embedding_threshold=0.70, device="cpu").resolve_global_ids(tracks, start_id=0)

    # Cluster for A should get global_id 0 (since earliest frame is 0..9)
    assert tA1.global_id == 0
    assert tA2.global_id == 0  # same component

    # Leftovers come after → should be 1 and 2, ordered by earliest frame: tX(20..) then tY(30..)
    assert tX.global_id == 1
    assert tY.global_id == 2


class TTrack:
    """
    Minimal stand-in for FaceTrack for resolver tests.
    Attributes/Methods used by resolver:
      - shot_id, segment_id, embeddings (list[np.ndarray])
      - first_frame(), last_frame()
      - global_id (written by resolver)
    """
    def __init__(self, shot, seg, f0, f1, embs):
        self.shot_id = shot
        self.segment_id = seg
        self._f0 = f0
        self._f1 = f1
        self.embeddings = [np.asarray(e, dtype=np.float32) for e in embs]
        self.global_id = None
        self.observations = [DummyObs(f0)]
    def first_frame(self): return self._f0
    def last_frame(self):  return self._f1

def unit_vec(d=512, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(d,)).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-9
    return v

def similar_to(v, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    w = v + noise * rng.normal(size=v.shape).astype(np.float32)
    w /= np.linalg.norm(w) + 1e-9
    return w

def build_tracks():
    # Two shots; shot 1 appears earlier in time
    base0 = unit_vec(seed=1)
    base1 = unit_vec(seed=2)
    # shot 1, seg 0, frames [0..99]
    t10 = TTrack(shot=1, seg=0, f0=0, f1=99, embs=[similar_to(base0, seed=11), similar_to(base0, seed=12)])
    # shot 2 has three faces, frames [100..199]
    t20 = TTrack(shot=2, seg=0, f0=100, f1=160, embs=[similar_to(base1, seed=21), similar_to(base1, seed=22)])
    t21 = TTrack(shot=2, seg=1, f0=120, f1=180, embs=[unit_vec(seed=3)])
    t22 = TTrack(shot=2, seg=2, f0=130, f1=190, embs=[unit_vec(seed=4)])
    return [t10, t20, t21, t22]

def mapping(tracks):
    # {(shot, seg) -> gid}
    out = {}
    for t in tracks:
        out[(t.shot_id, t.segment_id)] = t.global_id
    return out

def test_time_first_is_deterministic_under_shuffle(tmp_path):
    tracks = build_tracks()
    # Run once in original order
    res1 = GlobalIdentityResolver(embedding_threshold=0.65)
    res1.resolve_global_ids(tracks)
    m1 = mapping(tracks)

    # Shuffle input tracks and run again
    tracks2 = build_tracks()
    random.shuffle(tracks2)
    res2 = GlobalIdentityResolver(embedding_threshold=0.65)
    res2.resolve_global_ids(tracks2)
    m2 = mapping(tracks2)

    assert m1 == m2, f"time_first ordering should be invariant; got {m1} vs {m2}"
    # Also assert shot1,seg0 gets the earliest ID
    earliest_gid = min(m1.values())
    assert m1[(1,0)] == earliest_gid

def test_must_not_link_blocks_overlap():
    # Overlapping intervals in same shot must not be merged even if embeddings are identical
    v = unit_vec(seed=9)
    tA = TTrack(shot=2, seg=0, f0=100, f1=150, embs=[v])
    tB = TTrack(shot=2, seg=1, f0=120, f1=160, embs=[v])  # overlaps with A
    tracks = [tA, tB]
    res = GlobalIdentityResolver(embedding_threshold=0.5)
    res.resolve_global_ids(tracks)
    assert tA.global_id != tB.global_id, "overlapping groups in same shot must not merge"

def test_resolver_rejects_tracks_without_observations():
    t = FakeTrack(shot_id=1, segment_id=0, frame_range=(0, 9), embeddings=[np.zeros(512)], track_id=0)
    t.observations = []  # force invariant break
    with pytest.raises(ValueError):
        GlobalIdentityResolver().resolve_global_ids([t], start_id=0)