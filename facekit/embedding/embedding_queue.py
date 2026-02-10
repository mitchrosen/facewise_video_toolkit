from __future__ import annotations

import numpy as np
from typing import List, Optional, Sequence


class AlignedFaceEmbeddingQueue:
    """
    Bounded-memory queue for aligned faces.

    Contract:
    - enqueue(obs): queue an observation for embedding if needed
    - maybe_flush(): flush when pending >= max_pending
    - flush(...): embed queued aligned faces, write obs.embedding, clear obs.aligned_face
    - Never re-embed observations that already have embedding
    - Returns the list of observations that were embedded during the flush
      (so callers can attach/persist precisely those items).

    Notes:
    - max_pending is the flush fence (memory safety).
    - max_batch_size controls embedder batching (performance), not flush timing.
    """

    def __init__(
        self,
        max_batch_size: int = 32,
        max_pending: int = 1000,
        embedder: Optional[object] = None,
    ):
        if int(max_batch_size) <= 0:
            raise ValueError(f"max_batch_size must be > 0, got {max_batch_size}")
        if int(max_pending) <= 0:
            raise ValueError(f"max_pending must be > 0, got {max_pending}")

        self.max_batch_size = int(max_batch_size)
        self.max_pending = int(max_pending)
        self._embedder = embedder
        self._pending: List[object] = []

    @property
    def embedder(self) -> Optional[object]:
        return self._embedder

    @embedder.setter
    def embedder(self, embedder: object) -> None:
        self._embedder = embedder

    def set_embedder(self, embedder: object) -> None:
        self._embedder = embedder

    def __len__(self) -> int:
        return len(self._pending)

    def enqueue(self, obs) -> None:
        """
        Enqueue an observation if it needs embedding and has aligned_face available.
        Safe to call multiple times; duplicates are avoided by identity.
        """
        if obs is None:
            return

        if getattr(obs, "embedding", None) is not None:
            return
        if getattr(obs, "aligned_face", None) is None:
            return

        # Avoid accidental duplicate enqueues (same object identity)
        if any(o is obs for o in self._pending):
            return

        self._pending.append(obs)

    def maybe_flush(self, embedder: Optional[object] = None) -> List[object]:
        """
        Flush only if we are over the configured pending limit.
        Returns the list of embedded observations (empty if no flush happened).
        """
        if len(self._pending) >= self.max_pending:
            return self.flush(embedder)
        return []

    def flush(self, embedder: Optional[object] = None) -> List[object]:
        """
        Embed all pending aligned faces.
        Returns the list of observations embedded (empty if nothing embedded).
        """
        if not self._pending:
            return []

        if embedder is None:
            embedder = self._embedder

        if embedder is None:
            raise RuntimeError(
                "AlignedFaceEmbeddingQueue.flush() called without an embedder. "
                "Set q.embedder = <embedder> (or q.set_embedder(...)) or pass embedder to flush()."
            )

        # Filter: only embed obs still eligible at flush time.
        # (They may have been embedded externally, or had aligned_face cleared.)
        eligible: List[object] = []
        faces: List[np.ndarray] = []
        for obs in self._pending:
            if getattr(obs, "embedding", None) is not None:
                continue
            face = getattr(obs, "aligned_face", None)
            if face is None:
                continue
            eligible.append(obs)
            faces.append(face)

        # Clear pending no matter what (these obs are no longer "pending" after a flush attempt).
        self._pending.clear()

        if not eligible:
            return []

        # Batch embed
        embeddings = embedder.get_embedding_batch(faces, self.max_batch_size)

        if embeddings.shape[0] != len(eligible):
            raise RuntimeError(
                f"Embedder returned {embeddings.shape[0]} embeddings for {len(eligible)} inputs"
            )

        # Assign and clear aligned_face to reclaim memory
        embedded_obs: List[object] = []
        for obs, emb in zip(eligible, embeddings):
            obs.embedding = np.asarray(emb, dtype=np.float32)
            obs.aligned_face = None
            embedded_obs.append(obs)

        return embedded_obs
