from typing import List, Dict, Tuple, Optional
import numpy as np
from .face_structures import FaceTrack, FaceObservation
from facekit.utils.geometry import compute_iou


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

    # -------------------
    # Internal Utilities
    # -------------------

    def _claim_matches(
        self,
        items: List,
        candidates: List,
        score_fn,
        threshold: float,
        skip_condition=None
    ) -> Dict[int, List[Tuple]]:
        """
        Compute match candidates (items -> candidates) with scores ≥ threshold.
        Returns dict keyed by candidate_id, values = list of (candidate, item, score).
        """
        claims = {}
        for item in items:
            for cand in candidates:
                if skip_condition and skip_condition(cand):
                    continue
                score = score_fn(cand, item)
                if score >= threshold:
                    if cand.track_id not in claims:
                        claims[cand.track_id] = []
                    claims[cand.track_id].append((cand, item, score))

        # Sort candidates for each track by descending score
        for k in claims:
            claims[k].sort(key=lambda x: x[2], reverse=True)
        return claims

    def _resolve_conflicts(self, claims: Dict[int, List[Tuple]]) -> Tuple[List[Tuple], List]:
        """
        Resolve conflicts where multiple observations claim the same track.
        Strategy: keep the highest-scoring observation per track, return losers for reallocation.
        """
        assignments = []
        losers = []
        used_items = []
        for _, candidates in claims.items():
            for idx, (cand, item, _) in enumerate(candidates):
                if idx == 0 and all(item is not u for u in used_items):
                    assignments.append((cand, item))
                    used_items.append(item)
                else:
                    losers.append(item)
        return assignments, losers
    
    def assign_track_ids(self, frame_idx: int, observations: List[FaceObservation]) -> List[int]:
        """
        Assigns track_ids to observations using the same logic as update_tracks_with_frame,
        but returns the assigned track_ids instead of updating internal state.

        Intended for use on detection frames to allow tracker init with known IDs.
        """
        assigned_ids = []
        temp_active_flags = [False for _ in self.tracks]
        unassigned_obs = observations.copy()

        for obs in observations:
            best_track = None
            best_iou = 0.0
            best_idx = -1

            for idx, track in enumerate(self.tracks):
                if track.is_closed():
                    continue
                last_bbox = track.get_last_bbox()
                if last_bbox is None:
                    continue
                iou = compute_iou(last_bbox, obs.bbox)
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_track = track
                    best_idx = idx

            if best_track:
                assigned_ids.append(best_track.track_id)
                temp_active_flags[best_idx] = True
                unassigned_obs.remove(obs)
            else:
                assigned_ids.append(None)  # Will replace later

        # Fill in None values for new tracks
        for i, tid in enumerate(assigned_ids):
            if tid is None:
                new_track_id = self.next_track_id
                self.next_track_id += 1
                assigned_ids[i] = new_track_id

        return assigned_ids


    # -------------------
    # Frame-Level Assignment
    # -------------------

    def update_tracks_with_frame(self, frame_idx: int, observations: List[FaceObservation]):
        if not observations:
            return  # No observations to process
        
        sources = set(obs.source for obs in observations)
        assert len(sources) == 1, f"Mixed observation sources in frame {frame_idx}: {sources}"
        source = sources.pop()

        if source == 'tracking':
            self.update_tracks_with_tracking_frame(frame_idx, observations)
        elif source == 'detection':
            self.update_tracks_with_detection_frame(frame_idx, observations)
        else:
            raise ValueError(f"Unknown source '{source}' in frame {frame_idx}")

    def update_tracks_with_detection_frame(self, frame_idx: int, observations: List[FaceObservation]):
        """
        Called when current frame contains detection-based observations.
        Performs IoU-based matching against open tracks, creates new tracks for unmatched detections.
        """

        for track in self.tracks:
            track.is_active = False

        unassigned_obs = observations.copy()

        # Match via IoU
        for obs in observations:
            best_track = None
            best_iou = 0.0

            for track in self.tracks:
                if track.is_closed():
                    continue
                last_bbox = track.get_last_bbox()
                if last_bbox is None:
                    continue
                iou = compute_iou(last_bbox, obs.bbox)
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_track = track

            if best_track:
                best_track.add_observation(obs)
                best_track.is_active = True
                unassigned_obs.remove(obs)

        # Create new tracks for unmatched
        for obs in unassigned_obs:
            new_track = FaceTrack(track_id=self.next_track_id, shot_id=self.shot_number)
            new_track.add_observation(obs)
            new_track.is_active = True
            self.tracks.append(new_track)
            self.next_track_id += 1

        for track in self.tracks:
            if not track.is_active and not track.is_closed():
                track.mark_closed()

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

            obs = FaceObservation(frame_idx=frame_idx, bbox=bbox, aligned_face=aligned_face)
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


    # def resolve_segment_ids(self, segment_id_counter: int, embedding_threshold: float = 0.6) -> int:
    #     """
    #     Assign segment IDs to tracks in a shot:
    #     - Reuse existing segment_ids where possible based on embedding similarity.
    #     - Ensure no temporal overlap in reuse.
    #     - Cluster remaining unassigned tracks by embedding similarity (and no overlap).
    #     """

    #     def similarity(emb1, emb2):
    #         emb1 = emb1 / np.linalg.norm(emb1)
    #         emb2 = emb2 / np.linalg.norm(emb2)
    #         return float(np.dot(emb1, emb2))

    #     def tracks_temporally_overlap(t1, t2):
    #         return not (t1.last_frame() < t2.first_frame() or t2.last_frame() < t1.first_frame())

    #     # Split into assigned and unassigned
    #     unassigned = [t for t in self.tracks if t.segment_id is None]
    #     existing = [t for t in self.tracks if t.segment_id is not None]


    #     # Pass 1: Reuse existing IDs
    #     for u in unassigned[:]:
    #         if not u.has_embedding():
    #             continue
    #         best_match, best_score = None, -1.0
    #         for e in existing:
    #             if not e.has_embedding():
    #                 continue
    #             if tracks_temporally_overlap(u, e):
    #                 continue
    #             score = similarity(u.compute_average_embedding(), e.compute_average_embedding())
    #             if score >= embedding_threshold and score > best_score:
    #                 best_match, best_score = e, score
    #         if best_match:
    #             u.segment_id = best_match.segment_id
    #             existing.append(u)  # Now it can help future matches
    #             unassigned.remove(u)
    #             print(f"[DEBUG] Reused ID {u.segment_id} for track {u.track_id} (match={best_match.track_id}, score={best_score:.3f})")

    #     # Pass 2: Assign new IDs (with grouping)
    #     while unassigned:
    #         base = unassigned.pop(0)
    #         base.segment_id = segment_id_counter
    #         group = [base]

    #         if base.has_embedding():
    #             # Group other similar, non-overlapping tracks
    #             candidates = []
    #             for t in unassigned:
    #                 if not t.has_embedding():
    #                     continue
    #                 if tracks_temporally_overlap(base, t):
    #                     continue
    #                 score = similarity(base.compute_average_embedding(), t.compute_average_embedding())
    #                 if score >= embedding_threshold:
    #                     candidates.append((t, score))

    #             # Sort candidates by similarity
    #             candidates.sort(key=lambda x: x[1], reverse=True)
    #             for t, _ in candidates:
    #                 t.segment_id = segment_id_counter
    #                 group.append(t)

    #             # Remove grouped tracks
    #             unassigned = [t for t in unassigned if t.segment_id is None]

    #         segment_id_counter += 1

    #     return segment_id_counter

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
            print("\n[DEBUG] Summary of tracks missing embeddings:")
            for t in missing_tracks:
                aligned_count = sum(1 for obs in t.observations if obs.aligned_face is not None)
                valid_embeds = sum(1 for e in t.embeddings if e is not None)
                print(f"  - Track {t.track_id}: duration={t.duration()}, "
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
                print(f"[DEBUG] Assigned existing segment_id {current.segment_id} to track {current.track_id} "
                    f"(similarity={best_score:.3f})")
            else:
                current.segment_id = segment_id_counter
                print(f"[DEBUG] Assigned new segment_id {segment_id_counter} to track {current.track_id}")
                segment_id_counter += 1

        return segment_id_counter

    def finalize_tracks(self) -> List[FaceTrack]:
        """
        Close all remaining open tracks and return them.
        Call prior to 
        """

        #DEBUG
        for t in self.tracks:
            if t.first_frame() in {14863, 14864}:
                print(f"[DEBUG] Final track {t.track_id} starts at frame {t.first_frame()} with frames {t.get_frame_indices()}")

        for t in self.tracks:
            if not t.is_closed():
                t.mark_closed()
        return self.tracks

    def get_tracks_in_frame(self, frame_idx: int) -> List[FaceTrack]:
        """
        Get all tracks that contain an observation for the given frame index.
        """
        return [t for t in self.tracks if frame_idx in t.get_frame_indices()]
