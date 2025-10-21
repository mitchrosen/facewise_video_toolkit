from typing import List, Dict, Tuple, Optional
import numpy as np
import logging
from facekit.tracking.face_structures import FaceTrack, FaceObservation
from facekit.utils.geometry import compute_iou
from facekit.common.obs_consts import Source


class ShotFaceTrackAggregator:
    """
    Aggregates and manages face tracks for a single shot.
    Handles:
        - Frame-by-frame assignment using IoU and embeddings
        - Occlusion and conflict resolution
        - Track lifecycle management
        - Persistent identity mapping with segment IDs
    """

    def __init__(self, shot_number: int, iou_threshold: float = 0.5, embedding_threshold: float = 0.7):
        self.shot_number = shot_number
        self.iou_threshold = iou_threshold
        self.embedding_threshold = embedding_threshold
        self.tracks: List[FaceTrack] = []
        self.next_track_id = 0
        self._by_frame: dict[int, list[FaceObservation]] = {}

    # -------------------
    # Internal Utilities
    # -------------------

    def _index_obs(self, obs: FaceObservation) -> None:
        self._by_frame.setdefault(obs.frame_idx, []).append(obs)
        
    # -------------------
    # Frame-Level Assignment
    # -------------------

    def update_tracks_with_frame(
        self,
        frame_idx: int,
        observations: Optional[List[FaceObservation]] = None,
        ) -> int:
        """
        Update aggregator with observations from `frame_idx`.

        Returns
        -------
        int
            Number of tracks that were created on this frame (0 if none).
        """
        if not observations:
            return 0
    
        # In this pipeline, a frame's observations are all from one source.
        sources = {obs.source for obs in observations}
        assert len(sources) == 1, f"Mixed observation sources in frame {frame_idx}: {sources}"
        source = next(iter(sources))

        if source == Source.TRACKED:
            # Tracking frames should not create new tracks.
            self.update_tracks_with_tracking_frame(frame_idx, observations)
            return 0

        if source == Source.DETECTED:
            # Have this return how many new tracks were made by associating detections.
            created_count = self.update_tracks_with_detection_frame(frame_idx, observations)
            return int(created_count)

        raise ValueError(f"Unknown source '{source}' in frame {frame_idx}")
 
    def update_tracks_with_detection_frame(
        self, frame_idx: int, observations: List[FaceObservation]
    ) -> int:
        """
        Associate detections to existing tracks, create new tracks for unmatched detections,
        and update internal state.

        Returns
        -------
        int
            Number of new tracks created on this frame.
        """
        if not observations:
            return 0

        # Contract: this method only handles detection observations.
        sources = {obs.source for obs in observations}
        assert sources == {Source.DETECTED}, f"Non-detection sources at frame {frame_idx}: {sources}"

        # Mark all tracks inactive; we’ll flip to active if matched this frame.
        for track in self.tracks:
            track.is_active = False

        assigned_tracks: set[int] = set()
        unmatched_obs: list[FaceObservation] = []

        # Greedy IoU assignment: each detection → at most one track; each track ← at most one detection.
        for obs in observations:
            best_track = None
            best_iou = 0.0

            for track in self.tracks:
                if track.is_closed():
                    continue
                if track.track_id in assigned_tracks:
                    continue

                last_bbox = track.get_last_bbox()
                if last_bbox is None:
                    continue

                iou = compute_iou(last_bbox, obs.bbox)
                if iou >= self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_track = track

            if best_track is not None:
                obs.track_id = best_track.track_id
                if hasattr(obs, "shot_id"):
                    obs.shot_id = self.shot_number

                best_track.add_observation(obs)

                best_track.is_active = True
                assigned_tracks.add(best_track.track_id)
                self._index_obs(obs)
            else:
                unmatched_obs.append(obs)

        # Create new tracks for unmatched detections
        num_created = 0
        for obs in unmatched_obs:
            new_track = FaceTrack(track_id=self.next_track_id, shot_id=self.shot_number)
            
            obs.track_id = new_track.track_id
            if hasattr(obs, "shot_id"):
                obs.shot_id = self.shot_number

            new_track.add_observation(obs)

            new_track.is_active = True
            self.tracks.append(new_track)
            self.next_track_id += 1
            self._index_obs(obs)
            num_created += 1

        # On a detection frame, any not-matched open tracks are closed.
        for track in self.tracks:
            if not track.is_active and not track.is_closed():
                track.mark_closed()

        return num_created


    def update_tracks_with_tracking_frame(self, frame_idx: int, observations: List[FaceObservation]):
        """
        Called when current frame contains tracking-only observations.
        Each observation must already have a valid track_id assigned.
        """

        for track in self.tracks:
            track.is_active = False

        for obs in observations:
            assert obs.track_id is not None, f"Tracking obs missing track_id in frame {frame_idx}: {obs}"

            track = next((t for t in self.tracks if t.track_id == obs.track_id), None)
            if track is None:
                raise ValueError(f"No open track with ID {obs.track_id} in frame {frame_idx}")

            if track.is_closed():
                raise RuntimeError(f"Tried to update closed track {track.track_id} in frame {frame_idx}")

            track.add_observation(obs)
            track.is_active = True
            self._index_obs(obs)

        for track in self.tracks:
            if not track.is_active and not track.is_closed():
                track.mark_closed()

    def add_frame_observations(self, 
                               frame_idx: int, 
                               observations: List[Tuple[np.ndarray, List[Tuple[int, int]], np.ndarray]]):
        """
        Convert detector outputs for one frame into FaceObservations and update tracks.

        observations: list of tuples with the following semantic shape per item:
            (bbox, landmarks, aligned_face)
              - bbox:       (x1, y1, x2, y2), may arrive as list -> coerced to tuple of ints
              - landmarks:  5-point landmarks (unused here other than having produced aligned_face)
              - aligned_face: ArcFace-aligned RGB crop (112x112x3) or None if alignment failed
        """
        face_observations = []
        for bbox, _landmarks, aligned_face in observations:
            x1, y1, x2, y2 = map(int, bbox[:4])
            obs = FaceObservation(
                frame_idx=frame_idx, 
                bbox=(x1,y1,x2,y2), 
                aligned_face=aligned_face,
                source=Source.DETECTED)
            face_observations.append(obs)
        self.update_tracks_with_frame(frame_idx, face_observations)

    def attach_embeddings(self, track_id: int, embeddings: np.ndarray, expected_dim: int = 512):
        track = next((t for t in self.tracks if t.track_id == track_id), None)
        if track is None:
            raise KeyError(f"No track with id {track_id}")
        # Require a 2-D ndarray
        if not isinstance(embeddings, np.ndarray):
            raise TypeError(
                f"attach_embeddings expected a numpy.ndarray of shape (K,{expected_dim}); "
                f"got {type(embeddings).__name__}"
            )
        if embeddings.ndim != 2 or embeddings.shape[1] != expected_dim:
            raise ValueError(
                f"Embeddings must be shape (K,{expected_dim}); got {embeddings.shape}"
            )
        if not np.isfinite(embeddings).all(): 
            raise ValueError(
                f"Embeddings must all be finite; got\n {embeddings}"
            )
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32, copy=False)
        

        # Cheap (re-)normalization to keep invariants
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

        for i in range(embeddings.shape[0]):
            track.embeddings.append(embeddings[i].copy())

    def resolve_segment_ids(
        self,
        segment_id_counter: int,
        embedding_threshold: float = 0.6,
        iou_threshold: float = 0.5,
        emb_relax_factor: float = 0.7,
        max_gap: int = 10,
    ):
        """
        Assign segment IDs to tracks in a single pass, looking backward in time.

        Strategy:
        1. Sort tracks by first_frame.
        2. For each track in temporal order:
            a) Consider all earlier tracks that do NOT overlap in time:
                - If embedding similarity >= embedding_threshold, assign the ID of the most similar track.
            b) If no match, consider earlier tracks ending within max_gap frames and IoU >= iou_threshold:
                - If embedding similarity >= embedding_threshold * emb_relax_factor, assign the ID of the most similar track.
            c) If still no match, assign a new segment_id using segment_id_counter and increment counter.

        Parameters:
            segment_id_counter: int
                The next available segment ID to assign.
            embedding_threshold: float
                Minimum embedding cosine similarity for strong match.
            iou_threshold: float
                Minimum IoU for spatial continuity in relaxed match.
            emb_relax_factor: float
                Multiplier for embedding_threshold in relaxed match.
            max_gap: int
                Maximum frame gap to consider for relaxed match.

        Returns:
            Updated segment_id_counter.

        Raises:
            ValueError if emb_relax_factor not in (0,1].
            RuntimeError if any track is missing embedding when required.
        """

        if not (0.0 < emb_relax_factor <= 1.0):
            raise ValueError(f"emb_relax_factor must be in (0,1]; got {emb_relax_factor}")

        relaxed_embedding_threshold = embedding_threshold * emb_relax_factor

        def embedding_similarity(e1, e2):
            if e1 is None or e2 is None:
                raise RuntimeError("Embedding missing for similarity computation")
            e1 = e1 / np.linalg.norm(e1)
            e2 = e2 / np.linalg.norm(e2)
            return float(np.dot(e1, e2))
        
        # Identify and summarize tracks missing embeddings
        missing_tracks = [
            t for t in self.tracks if not t.has_embedding() or any(e is None for e in t.embeddings)
        ]

        if missing_tracks:
            for t in missing_tracks:
                aligned_count = sum(1 for obs in t.observations if obs.aligned_face is not None)
                valid_embeds = sum(1 for e in t.embeddings if e is not None)
                logging.error(f"  - Track {t.track_id}: duration={t.duration()}, "
                    f"frames={t.get_frame_indices()}, "
                    f"aligned_faces={aligned_count}, "
                    f"embeddings={valid_embeds}/{len(t.embeddings)}")
            raise RuntimeError(f"Track {missing_tracks[0].track_id} missing embedding")

        # Sort tracks by first_frame
        sorted_tracks = sorted(self.tracks, key=lambda t: t.first_frame())

        for i, current in enumerate(sorted_tracks):
            best_match, best_score = None, -1.0

            # Pass 1a: Look for earlier non-overlapping tracks
            for t in sorted_tracks[:i]:
                if current.first_frame() > t.last_frame():
                    if not t.has_embedding():
                        raise RuntimeError(f"Track {t.track_id} missing embedding")
                    score = embedding_similarity(current.compute_average_embedding(),
                                                t.compute_average_embedding())
                    if score >= embedding_threshold and score > best_score:
                        best_match, best_score = t, score

            # Pass 1b: If no strong match, consider earlier tracks within gap and IoU
            if best_match is None:
                for t in sorted_tracks[:i]:
                    gap = current.first_frame() - t.last_frame()
                    if 0 < gap <= max_gap:
                        last_bbox = t.get_last_bbox()
                        first_bbox = current.get_first_bbox()
                        if last_bbox is not None and first_bbox is not None:
                            iou = compute_iou(last_bbox, first_bbox)
                            if iou >= iou_threshold:
                                score = embedding_similarity(current.compute_average_embedding(),
                                                            t.compute_average_embedding())
                                if score >= relaxed_embedding_threshold and score > best_score:
                                    best_match, best_score = t, score

            # Assign segment_id
            if best_match:
                current.segment_id = best_match.segment_id
            else:
                current.segment_id = segment_id_counter
                segment_id_counter += 1

        return segment_id_counter

    def finalize_tracks(self) -> List[FaceTrack]:
        """
        Close all remaining open tracks and return them.
        """

        for t in self.tracks:
            if not t.is_closed():
                t.mark_closed()
        return self.tracks

    def get_tracks_in_frame(self, frame_idx: int) -> List[FaceTrack]:
        """
        Get all tracks that contain an observation for the given frame index.
        """
        return [t for t in self.tracks if frame_idx in t.get_frame_indices()]

    def observations_at(
        self,
        frame_idx: int,
        *,
        source: Optional[Source] = None,
        require_track_id: bool = True,
    ) -> list[FaceObservation]:
        """
        Return the aggregator-owned observations for a given frame.

        Args:
            frame_idx: absolute frame index.
            source: if set, filter by Source (e.g., Source.DETECTED / Source.TRACKED).
            require_track_id: if True, only return observations that have an assigned track_id.

        Returns:
            A list of FaceObservation objects (owned by the aggregator).
            Do not mutate these; treat as read-only.
        """
        # Simple implementation (scan tracks). Easy to swap out later if you index by frame.
        out: list[FaceObservation] = []
        for o in self._by_frame.get(frame_idx, []):
            if require_track_id and (o.track_id is None):
                continue
            if (source is not None) and (getattr(o, "source", None) != source):
                continue
            out.append(o)
        return out