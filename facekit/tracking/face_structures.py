from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Literal
import numpy as np
from facekit.utils.geometry import compute_iou
import logging
from facekit.common.obs_consts import Source, code_to_src, src_to_code

@dataclass
class FaceObservation:
    """
    Represents a single face observation in a specific frame.

    Attributes:
        frame_idx (int):
            Absolute frame index where the face was observed.

        bbox (tuple[int, int, int, int]):
            Bounding box in pixel coordinates (x1, y1, x2, y2) in the
            coordinate space of the original video frame.

        track_id (int, optional):
            Track ID assigned by the per-shot tracker. None for unassigned
            detections prior to track association.

        embedding (np.ndarray, optional):
            512-D facial feature vector associated with this observation.
            Embeddings are computed **after** the shot finishes and are
            attached only to DETECTED observations.

        confidence (float, optional):
            Detector confidence score for this observation, if provided
            by the face detector.

        landmarks (np.ndarray | list[tuple[float, float]] | None):
            Facial landmark coordinates for this detection, typically shape
            (K, 2) in pixel coordinates relative to the full frame.

            Landmarks are **persisted** in checkpoints and rehydrated on resume.
            They are the canonical geometric representation used to:
              - perform late face alignment,
              - compute embeddings in a second pass,
              - seed optical-flow–based landmark propagation.
    """
    frame_idx: int
    source: Source  
    track_id: int | None = None
    bbox: tuple[int, int, int, int] | None = None
    embedding: np.ndarray | None = None
    confidence: float | None = None
    landmarks: np.ndarray | None = None  # shape (K,2) float32 

    def __post_init__(self) -> None:
        # STRICT: source must already be a Source enum (fail-fast; no coercion here)
        if not isinstance(self.source, Source):
            raise TypeError(
                f"FaceObservation.source must be Source enum, got {type(self.source).__name__}: {self.source!r}"
            )
        # Normalize bbox if present, else allow None
        if self.bbox is not None:
            try:
                x1, y1, x2, y2 = self.bbox  # may raise
            except Exception as e:
                raise ValueError(f"bbox must be a 4-sequence, got {self.bbox!r}") from e
            # Cast to ints (upstream may hand floats); store canonical (x1,y1,x2,y2)
            self.bbox = (int(x1), int(y1), int(x2), int(y2))
            self.validate_bbox()

        # Normalize landmarks to np.ndarray (K,2) float32 when provided
        if self.landmarks is not None and not isinstance(self.landmarks, np.ndarray):
            try:
                lm = np.asarray(self.landmarks, dtype=np.float32)
                if lm.ndim == 1 and lm.size % 2 == 0 and lm.size >= 2:
                    lm = lm.reshape((-1, 2))
                if lm.ndim == 2 and lm.shape[1] == 2 and lm.shape[0] >= 1:
                    self.landmarks = lm
                else:
                    # malformed -> drop
                    self.landmarks = None
            except Exception:
                self.landmarks = None

    def validate_bbox(self) -> None:
        # Allow None upstream (e.g., some FLOW/FALLBACK cases)
        if self.bbox is None:
            return
        x1, y1, x2, y2 = self.bbox
        if not all(isinstance(v, int) for v in (x1, y1, x2, y2)):
            raise ValueError(f"bbox must be a 4-tuple of ints, got {self.bbox!r}")
        # Optional geometry sanity (cheap invariants)
        if x2 < x1 or y2 < y1:
            raise ValueError(f"bbox has negative width/height: {self.bbox!r}")

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

    last_landmarks: Optional[np.ndarray] = None     # shape (K,2), float32
    last_bbox: Optional[Tuple[int,int,int,int]] = None
    last_gray_roi: Optional[np.ndarray] = None      # previous ROI gray for LK

    last_frame_idx: int = -1
    last_det_frame_idx: int = -1

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
            self.last_landmarks = np.asarray(obs.landmarks, dtype=np.float32)

        # Update last_bbox helper
        if obs.bbox is not None:
            self.last_bbox = tuple(int(v) for v in obs.bbox[:4])
        
        # Keep the DET cache authoritative
        if getattr(obs, "source", None) == Source.DETECTED:
            self._last_det_frame_idx = int(obs.frame_idx)

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
    
    def count_landmark_observations(self) -> int:
        return sum(1 for obs in self.observations if obs.landmarks is not None)
    
    def count_embeddings(self):
        return len(self.embeddings)

    def last_det_bbox(self) -> Optional[tuple[int,int,int,int]]:
        for o in reversed(self.observations or []):
            if getattr(o, "source", None) == Source.DETECTED and o.bbox is not None:
                x1,y1,x2,y2 = map(int, o.bbox[:4])
                return (x1,y1,x2,y2)
        return None