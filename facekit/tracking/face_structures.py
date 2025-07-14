from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np
from facekit.utils.geometry import compute_iou

@dataclass
class FaceObservation:
    """
    Represents a single face observation in a specific frame.

    Attributes:
        frame_idx (int): Frame index where the face was observed.
        bbox (tuple): Bounding box in pixel coordinates (x1, y1, x2, y2).
        embedding (np.ndarray, optional): Facial feature vector.
        confidence (float, optional): Confidence score from the detector.
    """
    frame_idx: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    embedding: Optional[np.ndarray] = None
    confidence: Optional[float] = None

    def __post_init__(self):
        self.validate_bbox()

    def validate_bbox(self):
        if not (
            isinstance(self.bbox, tuple) and
            len(self.bbox) == 4 and
            all(isinstance(v, int) for v in self.bbox)
        ):
            raise ValueError(f"Invalid bbox: expected a 4-tuple of ints, got {self.bbox}")

@dataclass
class FaceTrack:
    """
    Represents a series of face observations believed to belong to the same person.

    Attributes:
        shot_id (int): Unique identifier for the shot that contains this track.
        track_id (int): Unique identifier for this track.
        observations (List[FaceObservation]): Chronologically ordered list of observations.
        is_active (bool): Whether this track is still active.
    
    Notes:
        Observations are also indexed internally by frame index for quick access.
        Duplicate frame indices are disallowed unless `force=True` is used.
    """
    shot_id: int
    track_id: int
    observations: List[FaceObservation] = field(default_factory=list)
    is_active: bool = True
    embeddings: List[np.ndarray] = field(default_factory=list)
    vchunk_id: Optional[int] = None  
    
    def __post_init__(self):
        self._frame_index_map = {}
        for obs in self.observations:
            if obs.frame_idx in self._frame_index_map:
                raise ValueError(f"Duplicate frame_idx {obs.frame_idx} found during initialization")
            self._frame_index_map[obs.frame_idx] = obs
    
    def add_observation(self, obs: FaceObservation, force: bool = False):
        """
        Add an observation to the track.

        Args:
            obs (FaceObservation): The observation to add.
            force (bool): If True, overwrite existing observation for the same frame index.

        Raises:
            ValueError: If an observation already exists for the frame index and force is False.
        """
        existing = self._frame_index_map.get(obs.frame_idx)
        if existing:
            if not force:
                raise ValueError(f"Observation for frame {obs.frame_idx} already exists. Use force=True to overwrite.")
        self._frame_index_map[obs.frame_idx] = obs
        self.observations.append(obs)
        if obs.embedding is not None:
            self.embeddings.append(obs.embedding)
 
    def has_embedding(self):
        return any(obs.embedding is not None for obs in self.observations)

    def get_bbox_by_observation_index(self, idx: int) -> Optional[Tuple[int, int, int, int]]:
        if 0 <= idx < len(self.observations):
            return self.observations[idx].bbox
        return None
    
    def get_bbox_by_frame(self, frame_idx: int) -> Optional[Tuple[int, int, int, int]]:
        """
        Retrieve the bounding box for a specific frame index.

        Args:
            frame_idx (int): The frame index to look up.

        Returns:
            Optional[Tuple[int, int, int, int]]: The bounding box if present, otherwise None.
        """
        obs = self._frame_index_map.get(frame_idx)
        return obs.bbox if obs else None

    def get_first_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Return the bounding box from the first observation, or None if empty.
        """
        return self.observations[0].bbox if self.observations else None

    def get_last_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Return the bounding box from the last observation, or None if no observations exist.
        """
        return self.observations[-1].bbox if self.observations else None

    def get_last_frame_idx(self) -> int:
        return self.observations[-1].frame_idx if self.observations else -1

    def compute_average_embedding(self) -> Optional[np.ndarray]:
        """
        Compute the average embedding across all observations in this track.

        Returns:
            Optional[np.ndarray]: The mean embedding vector, or None if no embeddings are available.
        """
        embeddings = [obs.embedding for obs in self.observations if obs.embedding is not None]
        if not embeddings:
            return None
        return np.mean(embeddings, axis=0)

    def duration(self) -> int:
        if not self.observations:
            return 0
        return self.observations[-1].frame_idx - self.observations[0].frame_idx + 1

    def shares_frames_with(self, other: 'FaceTrack') -> bool:
        """
        Returns True if this track and the other track share any frame indices.
        Useful for checking identity conflicts in the same frame.
        """
        return not set(self.get_frame_indices()).isdisjoint(other.get_frame_indices())

    def get_frame_indices(self) -> List[int]:
        """
        Return a list of frame indices covered by this track
        """
        return [obs.frame_idx for obs in self.observations]

    def get_average_bbox(self) -> Optional[Tuple[float, float, float, float]]:
        if not self.observations:
            return None
        x1s, y1s, x2s, y2s = zip(*(obs.bbox for obs in self.observations))
        return (
            float(np.mean(x1s)),
            float(np.mean(y1s)),
            float(np.mean(x2s)),
            float(np.mean(y2s))
        )
    
    def can_merge_with(
        self,
        other: 'FaceTrack',
        iou_thresh: float = 0.5,
        embedding_thresh: float = 0.6
    ) -> bool:
        """
        Determine if this track can be merged with another based on spatial and embedding distance (1 - similarity).

        Compares the last bbox and average embedding of this track with the first bbox and average embedding of the other.

        Args:
            other (FaceTrack): The other track to compare against.
            iou_thresh (float): Minimum IoU required to consider a merge.
            embedding_thresh (float): Maximum allowed cosine distance (1 - similarity) between embeddings.

        Returns:
            bool: True if mergeable, False otherwise.
        """
        # Check IoU continuity
        bbox_self = self.get_last_bbox()
        bbox_other = other.observations[0].bbox if other.observations else None
        if bbox_self is None or bbox_other is None:
            return False
        if compute_iou(bbox_self, bbox_other) < iou_thresh:
            return False

        # Check embedding cosine similarity
        e1 = self.compute_average_embedding()
        e2 = other.compute_average_embedding()
        if e1 is not None and e2 is not None:
            if np.linalg.norm(e1) == 0 or np.linalg.norm(e2) == 0:
                return False
            cos_sim = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
            return cos_sim > (1 - embedding_thresh)

        return True  # fallback if no embeddings are present
