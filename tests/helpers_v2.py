from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class Obs:
    frame_idx: int
    bbox: Tuple[float, float, float, float]  # xyxy
    source: str = "detection"
    confidence: float = 0.9
    embedding: Optional[np.ndarray] = None   # optional (512,) vector


class Track:
    def __init__(self, shot: int, tid: int, gid: int | None = None,
                 seg: int | None = None, obs: Optional[List[Obs]] = None):
        self.shot_id = shot
        self.track_id = tid
        self.global_id = gid
        self.segment_id = seg
        self.observations = obs or []

    def first_frame(self) -> int:
        return self.observations[0].frame_idx if self.observations else 0

    def last_frame(self) -> int:
        return self.observations[-1].frame_idx if self.observations else -1


def make_512(n: int) -> np.ndarray:
    """Return (n,512) float32 matrix with deterministic values (L2-normalized rows)."""
    a = np.arange(n * 512, dtype=np.float32).reshape(n, 512)
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return a / norms
