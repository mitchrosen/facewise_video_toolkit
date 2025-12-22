import pytest
import numpy as np

from facekit.tracking.aggregator import ShotFaceTrackAggregator
from facekit.tracking.face_structures import FaceTrack, FaceObservation
from facekit.common.obs_consts import Source

# ---------------------------
# Helper functions
# ---------------------------
def _src_detection() -> Source:
    return Source.DETECTED

def _src_tracking() -> Source:
    return Source.TRACKED

def random_embedding(dim: int = 512, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    n = float(np.linalg.norm(v)) or 1.0
    return v / n

def _set_track_embedding_fields(track: FaceTrack, emb: np.ndarray) -> None:
    """
    Many parts of the codebase historically accessed different embedding fields.
    We set several common ones so the test isn't coupled to one internal detail.
    """
    emb = np.asarray(emb, dtype=np.float32).ravel()
    n = float(np.linalg.norm(emb)) or 1.0
    emb = emb / n

    # Common patterns seen in codebases:
    track.embeddings = [emb]
    setattr(track, "_embedding", emb)
    setattr(track, "embedding_avg", emb)
    setattr(track, "_representative_embedding", emb)

    # If code calls a method, provide it:
    if not hasattr(track, "get_representative_embedding"):
        setattr(track, "get_representative_embedding", lambda e=emb: e)

def make_track(
    tid: int,
    embedding: np.ndarray | None,
    *,
    first_frame: int = 0,
    last_frame: int | None = None,
) -> FaceTrack:
    """
    Minimal valid track for tests:
      - has >=1 observation
      - does NOT include aligned_face (new contract)
      - optionally has a representative embedding
    """
    t = FaceTrack(shot_id=0, track_id=int(tid))
    f0 = int(first_frame)

    t.add_observation(
        FaceObservation(
            frame_idx=f0,
            bbox=(0, 0, 10, 10),
            source=_src_detection(),
        )
    )

    if last_frame is not None and int(last_frame) > f0:
        t.add_observation(
            FaceObservation(
                frame_idx=int(last_frame),
                bbox=(1, 1, 11, 11),
                source=_src_tracking(),
            )
        )

    if embedding is not None:
        _set_track_embedding_fields(t, embedding)

    return t

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    na = float(np.linalg.norm(a)) or 1.0
    nb = float(np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / (na * nb))

def _make_strongly_dissimilar(base: np.ndarray) -> np.ndarray:
    """
    Produce a vector with very low cosine similarity to `base`.
    We:
      - draw a random vector
      - project out base
      - normalize
      - if degeneracy, fall back to negative base
    """
    base = np.asarray(base, dtype=np.float32).ravel()
    bu = base / (float(np.linalg.norm(base)) or 1.0)

    v = random_embedding(seed=999)
    # remove projection onto base to make it near-orthogonal
    v = v - float(np.dot(bu, v)) * bu
    nv = float(np.linalg.norm(v))
    if nv < 1e-6:
        v = -bu
    else:
        v = v / nv

    # If still not dissimilar enough due to numerical weirdness, invert:
    if abs(_cos(bu, v)) > 0.2:
        v = -bu

    return v.astype(np.float32)

# ---------------------------
# Exception Tests
# ---------------------------
def test_raises_when_track_missing_embedding():
    aggregator = ShotFaceTrackAggregator(shot_number=0)
    t = make_track(0, None)
    aggregator.tracks.append(t)

    with pytest.raises(RuntimeError, match="missing embedding"):
        aggregator.resolve_segment_ids(10)

def test_raises_on_missing_embedding_in_prior_track():
    aggregator = ShotFaceTrackAggregator(shot_number=0)
    t1 = make_track(0, None)
    t2 = make_track(1, random_embedding(seed=2), first_frame=1)
    aggregator.tracks.extend([t1, t2])

    with pytest.raises(RuntimeError, match=r"Track 0 missing embedding|missing embedding"):
        aggregator.resolve_segment_ids(0)

# ---------------------------
# Segment ID Assignment Tests
# ---------------------------
def test_reuses_segment_id_on_same_embedding_within_chunk():
    aggregator = ShotFaceTrackAggregator(shot_number=0)
    emb = random_embedding(seed=1)
    t1 = make_track(0, emb)
    t2 = make_track(1, emb, first_frame=1)

    aggregator.tracks.extend([t1, t2])
    counter = aggregator.resolve_segment_ids(0)

    assert t1.segment_id == t2.segment_id
    assert counter == 1  # only one new segment assigned

def test_does_not_merge_dissimilar_embeddings():
    aggregator = ShotFaceTrackAggregator(shot_number=0)
    emb_ref = random_embedding(seed=1)
    emb_far = _make_strongly_dissimilar(emb_ref)

    t1 = make_track(0, emb_ref)
    t2 = make_track(1, emb_far, first_frame=1)

    aggregator.tracks.extend([t1, t2])
    counter = aggregator.resolve_segment_ids(0)

    assert t1.segment_id != t2.segment_id
    assert counter == 2  # both tracks got unique segments

def test_segment_id_reuse_on_embedding_similarity():
    aggregator = ShotFaceTrackAggregator(shot_number=0)
    emb = random_embedding(seed=1)
    t1 = make_track(0, emb)
    t2 = make_track(1, emb * 0.99, first_frame=1)  # very similar direction

    aggregator.tracks.extend([t1, t2])
    counter = aggregator.resolve_segment_ids(0)

    assert t1.segment_id == t2.segment_id
    assert counter == 1

def test_no_segment_id_reuse_when_similarity_below_threshold():
    aggregator = ShotFaceTrackAggregator(shot_number=0)
    e1 = random_embedding(seed=1)
    e2 = _make_strongly_dissimilar(e1)

    t1 = make_track(0, e1)
    t2 = make_track(1, e2, first_frame=1)

    aggregator.tracks.extend([t1, t2])

    # Make it strict: only very high cosine could reuse; disable any IoU-based relax path.
    counter = aggregator.resolve_segment_ids(
        0, embedding_threshold=0.99, emb_relax_factor=1.0, iou_threshold=2.0
    )

    assert t1.segment_id != t2.segment_id
    assert counter == 2

def test_conflict_resolution_reuses_highest_similarity_first():
    aggregator = ShotFaceTrackAggregator(shot_number=0)
    emb_ref = random_embedding(seed=1)

    t1 = make_track(0, emb_ref)
    t2 = make_track(1, emb_ref * 0.99, first_frame=1)  # close
    t3 = make_track(2, _make_strongly_dissimilar(emb_ref), first_frame=2)  # far

    aggregator.tracks.extend([t1, t2, t3])
    counter = aggregator.resolve_segment_ids(0)

    assert t1.segment_id == t2.segment_id
    assert t3.segment_id != t1.segment_id
    assert counter == 2  # one cluster reused, one new id

def test_mixed_tracks_reuse_and_assign_new_ids():
    aggregator = ShotFaceTrackAggregator(shot_number=0)
    emb_ref = random_embedding(seed=1)

    t1 = make_track(0, emb_ref)
    t2 = make_track(1, emb_ref, first_frame=1)  # reuse
    t3 = make_track(2, _make_strongly_dissimilar(emb_ref), first_frame=2)  # new

    aggregator.tracks.extend([t1, t2, t3])
    counter = aggregator.resolve_segment_ids(0)

    assert t1.segment_id == t2.segment_id
    assert t3.segment_id != t1.segment_id
    assert counter == 2

def test_counter_increments_correctly_for_multiple_new_assignments():
    aggregator = ShotFaceTrackAggregator(shot_number=0)

    emb1 = random_embedding(seed=1)
    emb2 = random_embedding(seed=2)
    emb3 = random_embedding(seed=3)

    # Ensure they're not accidentally super-similar
    assert abs(_cos(emb1, emb2)) < 0.95
    assert abs(_cos(emb1, emb3)) < 0.95

    t1 = make_track(0, emb1)
    t2 = make_track(1, emb2, first_frame=1)
    t3 = make_track(2, emb3, first_frame=2)

    aggregator.tracks.extend([t1, t2, t3])
    counter = aggregator.resolve_segment_ids(100)

    assert counter == 103  # 100 + 3 new segment IDs
    ids = {t1.segment_id, t2.segment_id, t3.segment_id}
    assert ids == {100, 101, 102}
