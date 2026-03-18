import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from facekit.tracking.face_structures import FaceTrack

def _first_abs_frame(track) -> int:
    # Track may expose helpers; else infer from observations.
    if hasattr(track, "first_frame") and callable(getattr(track, "first_frame")):
        return int(track.first_frame())
    observations = getattr(track, "observations", None) or []
    if not observations:
        return 1 << 30
    return int(getattr(observations[0], "frame_idx", 1 << 30))

@dataclass
class _Comp:
    tracks: list
    earliest: int

@dataclass
class _TrackGroup:
    members: List[FaceTrack]
    member_keys: List[Tuple[int, int, int, int, int, int, int, int]]
    group_embeddings: List[np.ndarray]
    shot_id: Optional[int]
    span_start: Optional[int]
    span_end: Optional[int]

class GlobalIdentityResolver:
    def __init__(self, embedding_threshold: float = 0.7, device: str = "auto"):
        """
        Resolves global IDs for FaceTracks using clustering based on embedding similarity.
        Supports GPU acceleration for similarity computation.

        Args:
            embedding_threshold (float): Cosine similarity threshold for linking tracks.
            device (str): "auto", "cpu", or "cuda".
                - "auto": Use GPU if available, else CPU.
                - "cpu": Force CPU.
                - "cuda": Require GPU (raise error if not available).
        """
        self.embedding_threshold = embedding_threshold
        self._threshold_tol = 1e-6

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available. Please check your environment.")
        else:
            self.device = device

        # self._audit = bool(int(os.environ.get("FACEKIT_GI_AUDIT", "0")))
        self._audit = 1
        logging.info(
            f"[INFO] GlobalIdentityResolver initialized on {self.device} (audit={self._audit})"
        )

    def resolve_global_ids(self, tracks: List[FaceTrack], start_id: int = 0) -> int:
        """
        Assign global_ids by clustering group representatives based on embedding similarity,
        groups consisting of tracks with same segment_id within a shot. A must-not-link
        constraint forbids merging tracks which overlap in time within the same shot.
        Uses a deterministic, best-first union-find so results are invariant to input order
        and cannot be 'bridged' across the constraint.

        Args:
            tracks (List[FaceTrack]): All FaceTracks (possibly across shots).
            start_id (int): Starting global_id counter.

        Returns:
            int: The next available global_id after assignment.
        """

        # ------------------------------
        # Helpers
        # ------------------------------

        def track_key(track: FaceTrack) -> Tuple[int, int, int, int, int, int, int, int]:
            """
            Stable key for determinism across cold/resume.
            Avoid using list indices and track_id.
            Use track content:
              - shot_id
              - segment_id (if present; else large sentinel)
              - first_frame, last_frame
              - first bbox (rounded to int pixels) as an additional stable tie-break
            """
            shot_id = int(getattr(track, "shot_id", -1))
            segment_id = getattr(track, "segment_id", None)
            segment_key = int(segment_id) if segment_id is not None else (1 << 30)

            try:
                first = int(track.first_frame())
            except Exception:
                observations = getattr(track, "observations", None) or []
                first = (
                    int(getattr(observations[0], "frame_idx", 1 << 30))
                    if observations
                    else (1 << 30)
                )

            try:
                last = int(track.last_frame())
            except Exception:
                observations = getattr(track, "observations", None) or []
                last = int(getattr(observations[-1], "frame_idx", -1)) if observations else -1

            x1 = y1 = x2 = y2 = (1 << 30)
            observations = getattr(track, "observations", None) or []
            if observations:
                bbox = getattr(observations[0], "bbox", None)
                if bbox is not None and len(bbox) == 4:
                    try:
                        x1, y1, x2, y2 = (
                            int(round(float(bbox[0]))),
                            int(round(float(bbox[1]))),
                            int(round(float(bbox[2]))),
                            int(round(float(bbox[3]))),
                        )
                    except Exception:
                        pass

            return (shot_id, segment_key, first, last, x1, y1, x2, y2)

        def robust_center(
            embedding_list: List[np.ndarray], cutoff: float = 0.30
        ) -> Optional[np.ndarray]:
            embedding_matrix = np.stack(embedding_list).astype(np.float32)

            norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embedding_matrix = embedding_matrix / norms

            center = embedding_matrix.mean(axis=0)
            center_norm = np.linalg.norm(center)
            if not np.isfinite(center_norm) or center_norm == 0.0:
                return None

            center_hat = center / center_norm
            cosine_distance = 1.0 - (embedding_matrix @ center_hat)
            keep_mask = cosine_distance <= cutoff

            trimmed_center = embedding_matrix[keep_mask].mean(axis=0) if keep_mask.any() else center_hat
            trimmed_center_norm = np.linalg.norm(trimmed_center)
            if not np.isfinite(trimmed_center_norm) or trimmed_center_norm == 0.0:
                return None

            return trimmed_center / trimmed_center_norm

        def first_frame(track: FaceTrack) -> int:
            return track.first_frame()

        def last_frame(track: FaceTrack) -> int:
            return track.last_frame()

        def _identity_embeddings_for_track(track: FaceTrack) -> List[np.ndarray]:
            """
            Return the embedding evidence this track should contribute to
            identity resolution.

            Contract:
            - If a stable representative_embedding exists, use that as the
              canonical identity signal for the track.
            - Otherwise fall back to the track's per-observation embeddings.
            """
            representative_embedding = getattr(track, "representative_embedding", None)
            is_stable = bool(getattr(track, "embedding_stable", False))

            if is_stable and representative_embedding is not None:
                representative_array = np.asarray(representative_embedding, dtype=np.float32)
                if representative_array.ndim == 1 and np.isfinite(representative_array).all():
                    return [representative_array]

            identity_embeddings: List[np.ndarray] = []
            for embedding in (getattr(track, "embeddings", None) or []):
                if embedding is None:
                    continue
                embedding_array = np.asarray(embedding, dtype=np.float32)
                if embedding_array.ndim == 1 and np.isfinite(embedding_array).all():
                    identity_embeddings.append(embedding_array)

            return identity_embeddings

        # ------------------------------
        # Audit helpers
        # ------------------------------

        def _obs_mix(track: FaceTrack):
            from facekit.common.obs_consts import Source

            detected_count = 0
            tracked_count = 0
            for observation in getattr(track, "observations", []) or []:
                source = getattr(observation, "source", None)
                if source == Source.DETECTED:
                    detected_count += 1
                elif source == Source.TRACKED:
                    tracked_count += 1
            return detected_count, tracked_count

        def _emb_stats(track: FaceTrack):
            embeddings = getattr(track, "embeddings", None) or []
            embedding_count = len(embeddings)
            if embedding_count == 0:
                return 0, 0, "0.000"

            nan_count = sum(
                0 if (embedding is not None and np.isfinite(embedding).all()) else 1
                for embedding in embeddings
            )

            norms = []
            for embedding in embeddings:
                if embedding is None:
                    continue
                norm_value = float(np.linalg.norm(np.asarray(embedding, dtype=np.float32)))
                if math.isfinite(norm_value):
                    norms.append(norm_value)

            average_norm = (sum(norms) / len(norms)) if norms else 0.0
            return embedding_count, nan_count, f"{average_norm:.3f}"

        def _tspan(track: FaceTrack):
            try:
                return int(track.first_frame()), int(track.last_frame())
            except Exception:
                observations = getattr(track, "observations", None) or []
                if not observations:
                    return (1 << 30, -1)
                return int(observations[0].frame_idx), int(observations[-1].frame_idx)

        # ------------------------------
        # Preconditions
        # ------------------------------

        missing_track_ids = [
            getattr(track, "track_id", None)
            for track in tracks
            if not getattr(track, "observations", None)
        ]
        if missing_track_ids:
            raise ValueError(
                "Track(s) have no observations; resolver requires >=1 observation per track. "
                f"Offenders track_id={missing_track_ids}"
            )

        if self._audit:
            logging.info("GI-AUDIT INPUT: count=%d", len(tracks))
            for track in sorted(tracks, key=track_key):
                shot_id = int(getattr(track, "shot_id", -1))
                segment_id = getattr(track, "segment_id", None)
                track_id = int(getattr(track, "track_id", -1))
                span_start, span_end = _tspan(track)
                observation_count = len(getattr(track, "observations", []) or [])
                detected_count, tracked_count = _obs_mix(track)
                embedding_count, nan_count, average_norm = _emb_stats(track)

                logging.info(
                    "GI-IN: shot=%d seg=%s tid=%d span=[%d..%d] n_obs=%d det=%d trk=%d emb=%d nan=%d avg_norm=%s",
                    shot_id,
                    (str(int(segment_id)) if segment_id is not None else "-"),
                    track_id,
                    span_start,
                    span_end,
                    observation_count,
                    detected_count,
                    tracked_count,
                    embedding_count,
                    nan_count,
                    average_norm,
                )

        # ------------------------------
        # Step 0: Build track groups
        # ------------------------------

        group_by_key: Dict[Any, _TrackGroup] = {}

        for track in tracks:
            shot_id = getattr(track, "shot_id", None)
            segment_id = getattr(track, "segment_id", None)

            if segment_id is None:
                group_key = ("__per_track__", track_key(track))
            else:
                group_key = (shot_id, segment_id)

            if group_key not in group_by_key:
                group_by_key[group_key] = _TrackGroup(
                    members=[],
                    member_keys=[],
                    group_embeddings=[],
                    shot_id=shot_id,
                    span_start=None,
                    span_end=None,
                )

            group = group_by_key[group_key]
            group.members.append(track)
            group.member_keys.append(track_key(track))

            for embedding in _identity_embeddings_for_track(track):
                group.group_embeddings.append(embedding)

            track_start = first_frame(track)
            track_end = last_frame(track)
            if track_start is not None and track_end is not None:
                group.span_start = (
                    track_start
                    if group.span_start is None
                    else min(group.span_start, track_start)
                )
                group.span_end = (
                    track_end
                    if group.span_end is None
                    else max(group.span_end, track_end)
                )

        if self._audit:

            def _group_sort_key(item):
                group_key, group = item
                if group_key[0] == "__per_track__":
                    return (1, group_key[1])

                shot_id, segment_id = group_key
                shot_sort = int(shot_id) if shot_id is not None else -1
                segment_sort = int(segment_id) if segment_id is not None else (1 << 30)
                return (0, shot_sort, segment_sort)

            for group_key, group in sorted(group_by_key.items(), key=_group_sort_key):
                if group_key[0] == "__per_track__":
                    logging.info(
                        "GI-GROUP: per_track key=%s members=%s",
                        group_key[1],
                        group.members,
                    )
                else:
                    logging.info(
                        "GI-GROUP: shot=%d seg=%s members=%s span=[%s..%s] embs=%d",
                        int(group.shot_id) if group.shot_id is not None else -1,
                        (str(int(group_key[1])) if group_key[1] is not None else "-"),
                        group.members,
                        str(group.span_start),
                        str(group.span_end),
                        len(group.group_embeddings),
                    )

        # ------------------------------
        # Step 1: Build group representatives
        # ------------------------------

        representative_group_keys = []
        representative_embeddings = []
        representative_shots = []
        representative_starts = []
        representative_ends = []

        def _rep_group_sort_key(item):
            group_key, group = item
            if group_key[0] == "__per_track__":
                return (1, group_key[1])

            shot_id, segment_id = group_key
            shot_sort = int(shot_id) if shot_id is not None else -1
            segment_sort = int(segment_id) if segment_id is not None else (1 << 30)
            min_member_key = min(group.member_keys) if group.member_keys else (1 << 30,) * 8
            return (0, shot_sort, segment_sort, min_member_key)

        for group_key, group in sorted(group_by_key.items(), key=_rep_group_sort_key):
            if group.group_embeddings:
                representative_embedding = robust_center(group.group_embeddings)
                if representative_embedding is None:
                    continue

                representative_group_keys.append(group_key)
                representative_embeddings.append(representative_embedding)
                representative_shots.append(
                    int(group.shot_id) if group.shot_id is not None else -1
                )
                representative_starts.append(
                    group.span_start if group.span_start is not None else -10**9
                )
                representative_ends.append(
                    group.span_end if group.span_end is not None else 10**9
                )

        if not representative_embeddings:
            for track in tracks:
                track.global_id = start_id
                start_id += 1
                if self._audit:
                    logging.info(
                        "GI-AUDIT: no reps -> assigned unique global_ids up to %d",
                        start_id - 1,
                    )
            return start_id

        group_members = [
            group_by_key[group_key].members for group_key in representative_group_keys
        ]
        group_member_keys = [
            group_by_key[group_key].member_keys for group_key in representative_group_keys
        ]

        # ------------------------------
        # Step 2: Compute similarities among group representatives
        # ------------------------------

        embedding_tensor = torch.tensor(
            np.stack(representative_embeddings),
            dtype=torch.float32,
            device=self.device,
        )
        embedding_tensor = F.normalize(embedding_tensor, p=2, dim=1)
        similarity_matrix = torch.mm(embedding_tensor, embedding_tensor.T)
        similarity_matrix_np = similarity_matrix.detach().cpu().numpy()

        embedding_threshold = float(self.embedding_threshold)
        threshold_tolerance = float(getattr(self, "_threshold_tol", 1e-6))
        group_count = len(representative_group_keys)

        candidate_edges = []
        for left_group_idx in range(group_count):
            similarity_row = similarity_matrix_np[left_group_idx]
            for right_group_idx in range(left_group_idx + 1, group_count):
                similarity = similarity_row[right_group_idx]
                if similarity + threshold_tolerance >= embedding_threshold:
                    left_tiebreak = (
                        min(group_member_keys[left_group_idx])
                        if group_member_keys[left_group_idx]
                        else (1 << 30,) * 8
                    )
                    right_tiebreak = (
                        min(group_member_keys[right_group_idx])
                        if group_member_keys[right_group_idx]
                        else (1 << 30,) * 8
                    )
                    if left_tiebreak > right_tiebreak:
                        left_tiebreak, right_tiebreak = right_tiebreak, left_tiebreak

                    candidate_edges.append(
                        (
                            similarity,
                            left_group_idx,
                            right_group_idx,
                            left_tiebreak,
                            right_tiebreak,
                        )
                    )

        candidate_edges.sort(key=lambda edge: (-float(edge[0]), edge[3], edge[4]))

        if self._audit:
            logging.info(
                "GI-EDGES: threshold=%.3f candidates=%d",
                embedding_threshold,
                len(candidate_edges),
            )
            for similarity, left_group_idx, right_group_idx, left_tiebreak, right_tiebreak in candidate_edges:
                logging.info(
                    "GI-EDGE: sim=%.4f i=%d j=%d tiebreak=(%s,%s)",
                    float(similarity),
                    left_group_idx,
                    right_group_idx,
                    left_tiebreak,
                    right_tiebreak,
                )

        # ------------------------------
        # Step 3: Union-Find with must-not-link constraint
        # ------------------------------

        parent = list(range(group_count))
        rank = [0] * group_count

        component_spans = [
            {representative_shots[group_idx]: [(representative_starts[group_idx], representative_ends[group_idx])]}
            for group_idx in range(group_count)
        ]

        def find(group_idx: int) -> int:
            while parent[group_idx] != group_idx:
                parent[group_idx] = parent[parent[group_idx]]
                group_idx = parent[group_idx]
            return group_idx

        def union(left_group_idx: int, right_group_idx: int) -> bool:
            """
            Union with union-by-rank.
            Returns True if merged, False otherwise.
            """
            left_root = find(left_group_idx)
            right_root = find(right_group_idx)
            if left_root == right_root:
                return False

            left_spans = component_spans[left_root]
            right_spans = component_spans[right_root]
            shared_shots = (set(left_spans.keys()) & set(right_spans.keys())) - {-1}

            if shared_shots:

                def overlaps(
                    left_start: int,
                    left_end: int,
                    right_start: int,
                    right_end: int,
                ) -> bool:
                    return not (left_end < right_start or right_end < left_start)

                for shot_id in shared_shots:
                    left_intervals = left_spans.get(shot_id, [])
                    right_intervals = right_spans.get(shot_id, [])

                    for left_start, left_end in left_intervals:
                        for right_start, right_end in right_intervals:
                            if overlaps(left_start, left_end, right_start, right_end):
                                if self._audit:
                                    logging.info(
                                        "GI-BLOCK: shot=%d A=[%d..%d] B=[%d..%d]",
                                        int(shot_id),
                                        int(left_start),
                                        int(left_end),
                                        int(right_start),
                                        int(right_end),
                                    )
                                return False

            if rank[left_root] < rank[right_root]:
                left_root, right_root = right_root, left_root

            parent[right_root] = left_root
            if rank[left_root] == rank[right_root]:
                rank[left_root] += 1

            merged_spans = dict(left_spans)
            for shot_id, interval_list in right_spans.items():
                merged_spans.setdefault(shot_id, []).extend(interval_list)

            component_spans[left_root] = merged_spans
            component_spans[right_root] = {}

            if self._audit:
                logging.info(
                    "GI-UNION: a=%d b=%d -> root=%d",
                    int(left_group_idx),
                    int(right_group_idx),
                    int(left_root),
                )

            return True

        merge_count = 0
        for similarity, left_group_idx, right_group_idx, _, _ in candidate_edges:
            if union(left_group_idx, right_group_idx):
                merge_count += 1

        if self._audit:
            logging.info(
                "GI-UNION-SUMMARY: merged=%d / candidates=%d / groups=%d",
                merge_count,
                len(candidate_edges),
                group_count,
            )

        # ------------------------------
        # Step 4: Gather components and assign global IDs
        # ------------------------------

        root_to_group_indices: Dict[int, List[int]] = {}
        for group_idx in range(group_count):
            root_idx = find(group_idx)
            root_to_group_indices.setdefault(root_idx, []).append(group_idx)

        def component_earliest_frame(group_indices: List[int]) -> int:
            earliest_frame = 10**9
            for group_idx in group_indices:
                earliest_frame = min(
                    earliest_frame,
                    representative_starts[group_idx]
                    if representative_starts[group_idx] is not None
                    else 10**9,
                )
            return earliest_frame

        def component_tiebreak(group_indices: List[int]) -> Tuple[int, int, int, int, int, int, int, int]:
            return min(
                min(group_member_keys[group_idx])
                for group_idx in group_indices
                if group_member_keys[group_idx]
            )

        components = sorted(
            root_to_group_indices.values(),
            key=lambda group_indices: (
                component_earliest_frame(group_indices),
                component_tiebreak(group_indices),
            ),
        )

        gid_assignments = []
        for component_group_indices in components:
            for group_idx in component_group_indices:
                for track in group_members[group_idx]:
                    track.global_id = start_id
                    gid_assignments.append(
                        (
                            start_id,
                            int(getattr(track, "shot_id", -1)),
                            int(
                                getattr(track, "segment_id", -1)
                                if getattr(track, "segment_id", None) is not None
                                else -1
                            ),
                            int(getattr(track, "track_id", -1)),
                            _tspan(track)[0],
                            _tspan(track)[1],
                        )
                    )
            start_id += 1

        if self._audit:
            for gid, shot_id, segment_id, track_id, span_start, span_end in sorted(
                gid_assignments, key=lambda row: (row[0], row[1], row[4])
            ):
                logging.info(
                    "GI-ASSIGN: gid=%d shot=%d seg=%s tid=%d span=[%d..%d]",
                    gid,
                    shot_id,
                    ("-" if segment_id < 0 else str(segment_id)),
                    track_id,
                    span_start,
                    span_end,
                )

        # ------------------------------
        # Step 5: Assign unique IDs to tracks without embeddings
        # ------------------------------

        assigned_track_ids = {
            id(track)
            for member_list in group_members
            for track in member_list
        }

        leftovers = [track for track in tracks if id(track) not in assigned_track_ids]

        def _leftover_sort_key(track: FaceTrack):
            try:
                first = int(track.first_frame())
            except Exception:
                observations = getattr(track, "observations", None) or []
                first = (
                    int(getattr(observations[0], "frame_idx", 1 << 30))
                    if observations
                    else (1 << 30)
                )
            return (first, track_key(track))

        leftovers.sort(key=_leftover_sort_key)

        for track in leftovers:
            track.global_id = start_id
            start_id += 1

        if self._audit and leftovers:
            logging.info(
                "GI-LEFTOVERS: assigned %d unique ids up to %d",
                len(leftovers),
                start_id - 1,
            )

        return start_id