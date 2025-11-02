from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Literal
import numpy as np
from facekit.utils.geometry import compute_iou
import logging
from facekit.common.obs_consts import Source

@dataclass
class FaceObservation:
    """
    Represents a single face observation in a specific frame.

    Attributes:
        frame_idx (int): Frame index where the face was observed.
        bbox (tuple): Bounding box in pixel coordinates (x1, y1, x2, y2).
        track_id (int, optional): Track ID for tracked faces.
        embedding (np.ndarray, optional): Facial feature vector.
        confidence (float, optional): Confidence score from the detector.
        aligned_face (np.ndarray, optional): Aligned face crop (e.g. ArcFace 112x112 RGB)
    """
    frame_idx: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    track_id: Optional[int] = None
    embedding: Optional[np.ndarray] = None
    confidence: Optional[float] = None
    aligned_face: Optional[np.ndarray] = None

    landmarks: Optional[List[Tuple[float,float]]] = None
    source: Optional[Source] = None

    def __post_init__(self):
        # Defensive: coerce bbox to tuple and validate
        self.bbox = tuple(self.bbox)
        self.validate_bbox()

    def validate_bbox(self):
        if not (
            isinstance(self.bbox, tuple) and
            len(self.bbox) == 4 and
            all(isinstance(v, int) for v in self.bbox)
        ):
            raise ValueError(
                f"Invalid bbox: could not convert to 4-tuple of ints — received {self.bbox}"
            )

@dataclass
class FaceTrack:
    """
    Represents a series of face observations believed to belong to the same person.

    Identification Fields:
        - shot_id (int): The shot number or video chunk this track belongs to.
        - track_id (int): Unique track identifier within a shot or chunk.
        - segment_id (Optional[int]): Identity assigned *within* a shot or chunk for matching faces.
        - global_id (Optional[int]): Identity resolved *across* the full video (multi-shot/global).

    Attributes:
        observations (List[FaceObservation]): Chronologically ordered observations.
        is_active (bool): True if matched in current frame; resets per frame.
        is_open (bool): True if track can accept new observations.
        embeddings (List[np.ndarray]): For computing similarity.
        last_landmarks (Optional[np.ndarray]): Landmark state for optical flow propagation.
        last_gray_roi (Optional[np.ndarray]): Cached grayscale ROI from last detection.
    """
    shot_id: int                      # The shot this track belongs to
    track_id: int                     # Unique within a shot
    segment_id: Optional[int] = None   # Local identity label (per-shot or chunk)
    global_id: Optional[int] = None   # Global identity label across shots

    observations: List[FaceObservation] = field(default_factory=list)
    is_active: bool = False       # Frame-level: assigned in current frame
    is_open: bool = True          # Track lifecycle
    embeddings: List[np.ndarray] = field(default_factory=list)

    last_landmarks: Optional[np.ndarray] = None     # shape (5,2), float32
    last_bbox: Optional[Tuple[int,int,int,int]] = None
    last_gray_roi: Optional[np.ndarray] = None      # previous ROI gray for LK

    last_frame_idx = -1
    last_det_frame_idx = -1
    last_bbox = (0, 0, 0, 0)

    #   Authoritative cached DET frame index (kept in sync on every DET add)
    _last_det_frame_idx: int | None = -1

    closed = False
    
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

        if not self.is_open:
            raise RuntimeError("Cannot add observation to a closed track")
        
        # Append after the closed check so we don't mutate on error
        self.observations.append(obs)

        existing = self._frame_index_map.get(obs.frame_idx)
        if existing and not force:
            raise ValueError(f"Observation for frame {obs.frame_idx} already exists. Use force=True to overwrite.")

        self._frame_index_map[obs.frame_idx] = obs

        # Store embedding if present
        if obs.embedding is not None:
            if not isinstance(obs.embedding, np.ndarray):
                logging.error(f"BAD EMBEDDING at frame {obs.frame_idx}: {obs.embedding}")
                raise TypeError(f"Embedding is not a numpy array (got {type(obs.embedding)}): frame {obs.frame_idx}")
            if obs.embedding.ndim != 1:
                logging.error(f"BAD EMBEDDING at frame {obs.frame_idx}: {obs.embedding}")
                raise ValueError(f"Embedding is not 1D (got shape {obs.embedding.shape}): frame {obs.frame_idx}")
            self.embeddings.append(obs.embedding)
 
        # For tracking continuity: store landmarks if this was a detection
        if obs.source == Source.DETECTED and obs.landmarks is not None:
            # prepare for optical flow
            self.last_known_landmarks = obs.landmarks

        # Update last_bbox helper
        if obs.bbox is not None:
            self.last_bbox = tuple(int(v) for v in obs.bbox[:4])
        
        # Keep the DET cache authoritative
        if getattr(obs, "source", None) == Source.DETECTED:
            self._last_det_frame_idx = int(obs.frame_idx)
      
        # Update last_bbox helper
        if obs.bbox is not None:
            self.last_bbox = tuple(int(v) for v in obs.bbox[:4])

    def reset_for_frame(self):
        self.is_active = False

    def mark_closed(self):
        """Mark this track as permanently closed (no more updates)."""
    
        self.is_open = False

    def mark_open(self):
        """Re-open this track (used during resume hydration)."""
        self.is_open = True
        
    def is_closed(self) -> bool:
        """Return True if this track has been permanently closed."""
        return not self.is_open

    def has_embedding(self):
        return bool(self.embeddings)

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

    def get_last_bbox(self):
        if not self.observations:
            return None
        return tuple(int(v) for v in self.observations[-1].bbox[:4])

    def get_first_bbox(self):
        return self.observations[0].bbox if self.observations else None

    def last_frame(self) -> Optional[int]:
        return int(self.observations[-1].frame_idx) if self.observations else None

    def last_det_frame(self) -> Optional[int]:
        """
        Authoritative 'last frame with a DETECTED observation'.
        Uses cached value; falls back to scan only if cache is missing
        (e.g., legacy/constructed objects).
        """
        if self._last_det_frame_idx is not None:
            return int(self._last_det_frame_idx)
        for o in reversed(self.observations or []):
            if getattr(o, "source", None) == Source.DETECTED:
                self._last_det_frame_idx = int(o.frame_idx)
                return self._last_det_frame_idx
        return None

    def first_frame(self):
        return self.observations[0].frame_idx if self.observations else float("inf")

    def compute_average_embedding(self) -> Optional[np.ndarray]:
        """
        Compute the average embedding across all observations in this track.

        Returns:
            Optional[np.ndarray]: The mean embedding vector, or None if no embeddings are available.
        """
        if not self.embeddings or any(e is None for e in self.embeddings):
            raise RuntimeError("Cannot compute average embedding: missing values")

        return np.mean(self.embeddings, axis=0)

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
    
    def count_aligned_faces(self):
        return sum(1 for obs in self.observations if obs.aligned_face is not None)

    def count_embeddings(self):
        return len(self.embeddings)

    def last_det_bbox(self) -> Optional[tuple[int,int,int,int]]:
        for o in reversed(self.observations or []):
            if getattr(o, "source", None) == Source.DETECTED and o.bbox is not None:
                x1,y1,x2,y2 = map(int, o.bbox[:4])
                return (x1,y1,x2,y2)
        return None