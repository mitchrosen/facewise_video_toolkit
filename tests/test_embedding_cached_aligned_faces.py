import numpy as np
import pytest

from facekit.common.obs_consts import Source
from facekit.pipeline.track_across_segments import _collect_aligned_faces_for_embedding
from facekit.tracking.face_structures import FaceObservation, FaceTrack


class ExplodingFrameProvider:
    """FrameProvider stub that makes any frame re-read obvious."""

    def get_frame(self, frame_idx: int):  # pragma: no cover
        raise AssertionError(
            f"get_frame() was called for frame_idx={frame_idx}, but this test requires cached aligned_face usage"
        )


def _mk_aligned_face(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # ArcFace expects 112x112 RGB.
    return rng.integers(0, 255, size=(112, 112, 3), dtype=np.uint8)


def test_collect_uses_cached_aligned_face_and_never_rereads_frame():
    """If obs.aligned_face is present, embedding collection must NOT call get_frame()."""
    aligned0 = _mk_aligned_face(0)
    aligned1 = _mk_aligned_face(1)

    tr = FaceTrack(
        shot_id=0,
        track_id=123,
        observations=[
            FaceObservation(
                frame_idx=10,
                source=Source.DETECTED,
                track_id=123,
                landmarks=[(1.0, 2.0)] * 5,
                aligned_face=aligned0,
            ),
            FaceObservation(
                frame_idx=20,
                source=Source.DETECTED,
                track_id=123,
                landmarks=[(1.0, 2.0)] * 5,
                aligned_face=aligned1,
            ),
        ],
    )

    faces, frames = _collect_aligned_faces_for_embedding(
        tr,
        frame_provider=ExplodingFrameProvider(),
    )

    assert frames == [10, 20]
    assert len(faces) == 2
    # Ensure we got the exact cached arrays back (no copies required by contract, but nice to enforce).
    assert faces[0] is aligned0
    assert faces[1] is aligned1


def test_collect_skips_if_embedding_already_present():
    """Once an observation already has embedding set, it should not be returned for embedding again."""
    aligned0 = _mk_aligned_face(0)
    aligned1 = _mk_aligned_face(1)

    tr = FaceTrack(
        shot_id=0,
        track_id=7,
        observations=[
            FaceObservation(
                frame_idx=1,
                source=Source.DETECTED,
                track_id=7,
                landmarks=[(1.0, 2.0)] * 5,
                aligned_face=aligned0,
                embedding=np.ones((512,), dtype=np.float32),
            ),
            FaceObservation(
                frame_idx=2,
                source=Source.DETECTED,
                track_id=7,
                landmarks=[(1.0, 2.0)] * 5,
                aligned_face=aligned1,
                embedding=None,
            ),
        ],
    )

    faces, frames = _collect_aligned_faces_for_embedding(
        tr,
        frame_provider=ExplodingFrameProvider(),
    )

    assert frames == [2]
    assert len(faces) == 1
    assert faces[0] is aligned1


def test_collect_ignores_non_detected_sources_even_if_aligned_face_present():
    """We only embed DETECTED observations for now."""
    aligned0 = _mk_aligned_face(0)
    aligned1 = _mk_aligned_face(1)

    tr = FaceTrack(
        shot_id=0,
        track_id=8,
        observations=[
            FaceObservation(
                frame_idx=10,
                source=Source.TRACKED,
                track_id=8,
                landmarks=[(1.0, 2.0)] * 5,
                aligned_face=aligned0,
            ),
            FaceObservation(
                frame_idx=20,
                source=Source.DETECTED,
                track_id=8,
                landmarks=[(1.0, 2.0)] * 5,
                aligned_face=aligned1,
            ),
        ],
    )

    faces, frames = _collect_aligned_faces_for_embedding(
        tr,
        frame_provider=ExplodingFrameProvider(),
    )

    assert frames == [20]
    assert len(faces) == 1
    assert faces[0] is aligned1


def test_collect_respects_embedding_frame_policy():
    aligned0 = _mk_aligned_face(0)
    aligned1 = _mk_aligned_face(1)

    tr = FaceTrack(
        shot_id=0,
        track_id=9,
        observations=[
            FaceObservation(
                frame_idx=10,
                source=Source.DETECTED,
                track_id=9,
                landmarks=[(1.0, 2.0)] * 5,
                aligned_face=aligned0,
            ),
            FaceObservation(
                frame_idx=20,
                source=Source.DETECTED,
                track_id=9,
                landmarks=[(1.0, 2.0)] * 5,
                aligned_face=aligned1,
            ),
        ],
    )

    def policy(track, obs, frame_idx: int) -> bool:
        return frame_idx == 20

    faces, frames = _collect_aligned_faces_for_embedding(
        tr,
        frame_provider=ExplodingFrameProvider(),
        embedding_frame_policy=policy,
    )

    assert frames == [20]
    assert len(faces) == 1
    assert faces[0] is aligned1
