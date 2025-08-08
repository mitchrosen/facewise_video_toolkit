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
        - Persistent identity mapping with vchunk IDs
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
        Compute match candidates (items → candidates) with scores ≥ threshold.
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

    # -------------------
    # Frame-Level Assignment
    # -------------------

    def update_tracks_with_frame(self, frame_idx: int, observations: List[FaceObservation]):
        """
        Update existing tracks and/or create new tracks for the current frame.

        Workflow:
        1. Reset `is_active` for all tracks.
        2. Match observations to open tracks by IoU (previous frame bounding box).
        3. Add matched observations to corresponding tracks.
        4. Create new tracks for unmatched observations.
        5. Close tracks that were not updated in this frame.
        """

        # 1. Reset activity flags
        for track in self.tracks:
            track.is_active = False

        unassigned_obs = observations.copy()

        # 2. IoU-based matching with currently open tracks
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

        # 3. Create new tracks for unmatched observations
        for obs in unassigned_obs:
            new_track = FaceTrack(track_id=self.next_track_id, shot_id=self.shot_number)
            new_track.add_observation(obs)
            new_track.is_active = True
            self.tracks.append(new_track)
            self.next_track_id += 1

        # 4. Close tracks that were not updated
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
              - bbox:       (x1, y1, x2, y2), may arrive as list → coerced to tuple of ints
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

    # -------------------
    # Persistent Identity Assignment
    # -------------------

    # def resolve_vchunk_ids(self, vchunk_id_counter: int, embedding_threshold: float = 0.6) -> int:
    #     """
    #     Assign vchunk IDs within a single shot based on embedding similarity.
    #     """
    #     print("\n[DEBUG] Starting resolve_vchunk_ids()")
    #     unassigned_tracks = [t for t in self.tracks if t.vchunk_id is None]
    #     existing_tracks = [t for t in self.tracks if t.vchunk_id is not None]

    #     print(f"[DEBUG] Initial unassigned: {[t.track_id for t in unassigned_tracks]}")
    #     print(f"[DEBUG] Initial existing: {[t.track_id for t in existing_tracks]}")

    #     def similarity(e1, e2):
    #         return float(np.dot(e1 / np.linalg.norm(e1), e2 / np.linalg.norm(e2)))

    #     # Pass 1: Try to reuse IDs for similar embeddings
    #     for u in unassigned_tracks[:]:
    #         if not u.has_embedding():
    #             print(f"[DEBUG] Track {u.track_id} skipped (no embedding)")
    #             continue

    #         best_match, best_score = None, -1.0
    #         for e in existing_tracks:
    #             if not e.has_embedding():
    #                 continue
    #             score = similarity(u.compute_average_embedding(), e.compute_average_embedding())
    #             print(f"[DEBUG] Compare unassigned {u.track_id} to existing {e.track_id}: score={score:.3f}")
    #             if score > best_score and score >= embedding_threshold:
    #                 best_match, best_score = e, score

    #         if best_match:
    #             print(f"[DEBUG] Assigning {u.track_id} → vchunk_id {best_match.vchunk_id} (score={best_score:.3f})")
    #             u.vchunk_id = best_match.vchunk_id
    #             existing_tracks.remove(best_match)  # Prevent reuse
    #         else:
    #             print(f"[DEBUG] No match for {u.track_id}, assigning new vchunk_id {vchunk_id_counter}")
    #             u.vchunk_id = vchunk_id_counter
    #             vchunk_id_counter += 1

    #     # Pass 2: Assign IDs to any leftovers (should be none)
    #     for t in unassigned_tracks:
    #         if t.vchunk_id is None:
    #             print(f"[DEBUG] Final fallback assignment for {t.track_id}: {vchunk_id_counter}")
    #             t.vchunk_id = vchunk_id_counter
    #             vchunk_id_counter += 1

    #     print("[DEBUG] Final assignments:", [(t.track_id, t.vchunk_id) for t in self.tracks])
    #     return vchunk_id_counter

    def resolve_vchunk_ids(self, vchunk_id_counter: int, embedding_threshold: float = 0.6) -> int:
        """
        Assign vchunk IDs to tracks in a shot:
        - Reuse existing vchunk_ids where possible based on embedding similarity.
        - Ensure no temporal overlap in reuse.
        - Cluster remaining unassigned tracks by embedding similarity (and no overlap).
        """

        def similarity(emb1, emb2):
            emb1 = emb1 / np.linalg.norm(emb1)
            emb2 = emb2 / np.linalg.norm(emb2)
            return float(np.dot(emb1, emb2))

        def tracks_temporally_overlap(t1, t2):
            return not (t1.last_frame() < t2.first_frame() or t2.last_frame() < t1.first_frame())

        # Split into assigned and unassigned
        unassigned = [t for t in self.tracks if t.vchunk_id is None]
        existing = [t for t in self.tracks if t.vchunk_id is not None]


        # ✅ Pass 1: Reuse existing IDs
        for u in unassigned[:]:
            if not u.has_embedding():
                continue
            best_match, best_score = None, -1.0
            for e in existing:
                if not e.has_embedding():
                    continue
                if tracks_temporally_overlap(u, e):
                    continue
                score = similarity(u.compute_average_embedding(), e.compute_average_embedding())
                if score >= embedding_threshold and score > best_score:
                    best_match, best_score = e, score
            if best_match:
                u.vchunk_id = best_match.vchunk_id
                existing.append(u)  # Now it can help future matches
                unassigned.remove(u)
                print(f"[DEBUG] Reused ID {u.vchunk_id} for track {u.track_id} (match={best_match.track_id}, score={best_score:.3f})")

        # ✅ Pass 2: Assign new IDs (with grouping)
        while unassigned:
            base = unassigned.pop(0)
            base.vchunk_id = vchunk_id_counter
            group = [base]

            if base.has_embedding():
                # Group other similar, non-overlapping tracks
                candidates = []
                for t in unassigned:
                    if not t.has_embedding():
                        continue
                    if tracks_temporally_overlap(base, t):
                        continue
                    score = similarity(base.compute_average_embedding(), t.compute_average_embedding())
                    if score >= embedding_threshold:
                        candidates.append((t, score))

                # Sort candidates by similarity
                candidates.sort(key=lambda x: x[1], reverse=True)
                for t, _ in candidates:
                    t.vchunk_id = vchunk_id_counter
                    group.append(t)

                # Remove grouped tracks
                unassigned = [t for t in unassigned if t.vchunk_id is None]

            vchunk_id_counter += 1

        return vchunk_id_counter


    def finalize_tracks(self) -> List[FaceTrack]:
        """
        Close all remaining open tracks and return them.
        Call prior to 
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
