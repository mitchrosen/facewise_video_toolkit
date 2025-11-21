from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

from facekit.common.obs_consts import Source
from facekit.tracking.face_structures import FaceObservation


BBox = Tuple[int, int, int, int]

@dataclass
class Obs(FaceObservation):
    """
    Test helper that behaves like the old Obs but *is* a FaceObservation,
    so json_v2.normalize_obs_items_for_output is happy.
    """
    def __init__(
        self,
        frame_idx: int,
        bbox: BBox,
        source: Source = Source.DETECTED,
        confidence: float = 0.9,
        embedding: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(
            frame_idx=frame_idx,
            bbox=bbox,
            source=source,
            confidence=float(confidence),
            embedding=embedding,
        )


class Track:
    def __init__(self, shot_number: int, track_id: int, gid: int, obs: List[Obs]):
        self.shot_number = shot_number
        self.track_id = track_id
        self.global_id = gid
        self.observations = list(obs)

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
