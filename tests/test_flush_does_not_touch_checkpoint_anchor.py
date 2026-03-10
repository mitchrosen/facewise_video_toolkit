from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest

import facekit.pipeline.track_across_segments as tas


@dataclass
class _FakeTrack:
    track_id: int


class _FakeAggregator:
    def __init__(self, tracks):
        self.tracks = tracks
        self.attached = []  # list of (tid, embs)

    def attach_embeddings(self, tid: int, embs: np.ndarray) -> None:
        self.attached.append((int(tid), np.asarray(embs)))


class _CheckpointThatMustNotBeUsedForAnchors:
    """
    Anchor methods should not be invoked from mainline flush logic.
    """
    def mark_embedding_safe(self, *a, **kw):
        raise AssertionError("Mainline flush must not call checkpoint.mark_embedding_safe()")


def test_attach_and_persist_embedded_obs_does_not_call_checkpoint_anchor(monkeypatch) -> None:
    """
    Should produce a flush-commit event/state and
    the checkpoint layer (if enabled) may observe it.
    """
    # Arrange embedded observations
    obs = [
        SimpleNamespace(frame_idx=10, track_id=1, embedding=np.ones((512,), dtype=np.float32)),
        SimpleNamespace(frame_idx=12, track_id=1, embedding=np.ones((512,), dtype=np.float32) * 2),
        SimpleNamespace(frame_idx=11, track_id=2, embedding=np.ones((512,), dtype=np.float32) * 3),
    ]

    agg = _FakeAggregator(tracks=[_FakeTrack(1), _FakeTrack(2)])

    # Allow per-track persistence to be a no-op in this test (we're not testing it here).
    monkeypatch.setattr(tas, "_persist_embeddings_for_track", lambda *a, **kw: None)

    ckpt = _CheckpointThatMustNotBeUsedForAnchors()

    # Act
    # Under the new design, this function should either:
    #   (a) return a flush-commit object (preferred), OR
    #   (b) call a non-checkpoint hook, OR
    #   (c) update a passed-in bookkeeper
    #
    # But it must NOT call ckpt.mark_embedding_safe().
    tas._attach_and_persist_embedded_obs(
        embedded_obs=obs,
        aggregator=agg,
        checkpoint=ckpt,
        shot_number=0,
        shot_first_frame=0,
    )

    # Assert: we still attached embeddings
    assert len(agg.attached) == 2
    tids = {tid for tid, _ in agg.attached}
    assert tids == {1, 2}