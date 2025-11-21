# tests/test_global_label_determinism.py
import numpy as np
import pytest

from facekit.tracking.face_structures import FaceTrack, FaceObservation
from facekit.tracking.tracking_resolution import GlobalIdentityResolver  # adjust if your path differs
from facekit.common.obs_consts import Source

def _mk_track(tid: int, shot: int, first: int, last: int, embs: list[np.ndarray]) -> FaceTrack:
    """
    Build a FaceTrack with a contiguous span [first..last] and DET obs at first/last.
    We attach embeddings to the track via track.embeddings, and set observations so
    first_seen order is deterministic across tracks.
    """
    t = FaceTrack(track_id=tid, shot_id=shot)
    # two DET observations (first/last) so "first seen" is unambiguous
    obs_first = FaceObservation(
        frame_idx=first,
        track_id=tid,
        bbox=(0, 0, 10, 10),
        embedding=None,
        confidence=0.9,
        source=Source.DETECTED,
    )
    obs_last = FaceObservation(
        frame_idx=last,
        track_id=tid,
        bbox=(0, 0, 10, 10),
        embedding=None,
        confidence=0.9,
        source=Source.DETECTED,
    )
    t.observations = [obs_first, obs_last]
    # current code expects .embeddings list on the track (not per-obs) for resolver
    t.embeddings = [np.asarray(e, dtype=np.float32) for e in embs]
    return t

def _unit_vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(512,)).astype(np.float32)
    v /= (np.linalg.norm(v) + 1e-9)
    return v

@pytest.mark.parametrize("threshold", [0.70])
def test_global_ids_stable_with_or_without_preanchor_embeddings(threshold):
    """
    Arrange three identities A, B, C with distinct embeddings and first-seen frames:
      A → first_seen=0,  B → 103,  C → 117
    Case 1 (baseline): all have embeddings across their det frames.
    Case 2 (resume-like): pre-anchor (<=180) embeddings for A/B/C are *missing*.
    Expectation (desired behavior): global cluster assignments/ordering are identical.
    Reality (current): missing pre-anchor embeddings causes different grouping/order.
    """
    # Identity centroids
    A = _unit_vec(1)
    B = _unit_vec(2)
    C = _unit_vec(3)

    # Tracks (single-shot for simplicity). We stagger first frames to encode desired label order A,B,C.
    tA = _mk_track(tid=10, shot=2, first=0,   last=116, embs=[A, A, A])   # appears first
    tB = _mk_track(tid=11, shot=2, first=103, last=239, embs=[B, B, B])
    tC = _mk_track(tid=12, shot=2, first=117, last=299, embs=[C, C, C])

    # Baseline: all embeddings present
    r = GlobalIdentityResolver(embedding_threshold=threshold, device="cpu")
    tracks1 = [tA, tB, tC]
    _ = r.resolve_global_ids(tracks1, start_id=0)
    ids1 = {t.track_id: t.global_id for t in tracks1}

    # Resume-like: strip pre-anchor embeddings (simulate the gap observed in logs)
    # Here, we remove all embeddings to mimic the worst case seen; you can also remove only <=anchor.
    tA2 = _mk_track(tid=10, shot=2, first=0,   last=116, embs=[])
    tB2 = _mk_track(tid=11, shot=2, first=103, last=239, embs=[])
    tC2 = _mk_track(tid=12, shot=2, first=117, last=299, embs=[])
    r2 = GlobalIdentityResolver(embedding_threshold=threshold, device="cpu")
    tracks2 = [tA2, tB2, tC2]
    _ = r2.resolve_global_ids(tracks2, start_id=0)
    ids2 = {t.track_id: t.global_id for t in tracks2}

    # Desired assertion (will FAIL today): ordering should match even when embeddings are missing pre-anchor.
    # Concretely, whichever track is first-seen (A) should get the same smallest global id in both cases.
    # Current behavior assigns leftover/no-emb tracks *after* clustered ones, changing order.
    assert ids1[10] == ids2[10], f"Track A global_id changed: baseline={ids1[10]} resumeLike={ids2[10]}"
    assert ids1[11] == ids2[11], f"Track B global_id changed: baseline={ids1[11]} resumeLike={ids2[11]}"
    assert ids1[12] == ids2[12], f"Track C global_id changed: baseline={ids1[12]} resumeLike={ids2[12]}"
