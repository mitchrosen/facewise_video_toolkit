from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class TrackEmbeddingSample:
    frame_idx: int
    track_local_index: int
    source: str
    embedding: np.ndarray | None
    quality_score: float | None = None


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def _connected_components(adj: list[list[int]]) -> list[list[int]]:
    n = len(adj)
    seen = [False] * n
    components: list[list[int]] = []

    for start in range(n):
        if seen[start]:
            continue

        stack = [start]
        seen[start] = True
        comp: list[int] = []

        while stack:
            node = stack.pop()
            comp.append(node)
            for nbr in adj[node]:
                if not seen[nbr]:
                    seen[nbr] = True
                    stack.append(nbr)

        components.append(sorted(comp))

    return components


def _average_internal_similarity(
    comp: list[int],
    sim: np.ndarray,
) -> float:
    if len(comp) <= 1:
        return 1.0

    vals = []
    for i in range(len(comp)):
        for j in range(i + 1, len(comp)):
            vals.append(float(sim[comp[i], comp[j]]))
    return float(sum(vals) / len(vals))


def select_consistent_embedding_subset(
    samples: list[TrackEmbeddingSample],
    similarity_threshold: float = 0.95,
) -> list[TrackEmbeddingSample]:
    """
    Return the most self-consistent subset of valid embedding samples.

    Rules:
    - Ignore samples whose embedding is None.
    - If there are 3 or fewer valid samples, return all of them.
    - If there are 4 or more valid samples, return the largest/best mutually
      similar subset of size at least 3 and discard the rest.
    - Return results in ascending frame order.
    """
    valid_samples = [s for s in samples if s.embedding is not None]

    if len(valid_samples) <= 3:
        return sorted(valid_samples, key=lambda s: s.frame_idx)

    n = len(valid_samples)
    embs = [np.asarray(s.embedding, dtype=np.float32) for s in valid_samples]

    sim = np.eye(n, dtype=np.float32)
    adj: list[list[int]] = [[] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            sij = _cosine_similarity(embs[i], embs[j])
            sim[i, j] = sim[j, i] = sij
            if sij >= similarity_threshold:
                adj[i].append(j)
                adj[j].append(i)

    comps = _connected_components(adj)

    eligible = [c for c in comps if len(c) >= 3]

    if not eligible:
        # Fallback: choose the 3 samples with highest mean similarity to others
        mean_scores = []
        for i in range(n):
            mean_scores.append((float(np.mean(sim[i])), i))
        chosen_idx = sorted(i for _, i in sorted(mean_scores, reverse=True)[:3])
        return [valid_samples[i] for i in chosen_idx]

    eligible.sort(
        key=lambda comp: (len(comp), _average_internal_similarity(comp, sim)),
        reverse=True,
    )
    best = sorted(eligible[0])

    chosen = [valid_samples[i] for i in best]
    return sorted(chosen, key=lambda s: s.frame_idx)