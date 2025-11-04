import json
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple, Any, Union
from pathlib import Path
import logging

from facekit.tracking.face_structures import FaceTrack


class GlobalIdentityResolver:
    def __init__(
        self,
        embedding_threshold: float = 0.7,
        device: str = "auto",
        *,
        debug_dump_path: Optional[Union[str, Path]] = None,
        seed_map: Optional[Dict[Tuple[int, int], int]] = None,
        max_logged_edges: int = 200,
    ):
        """
        Resolves global IDs by clustering (shot_id, segment_id) groups using cosine similarity,
        with a must-not-link constraint for overlapping same-shot intervals.

        Args:
            embedding_threshold: Cosine similarity threshold to connect groups.
            device: "auto" | "cpu" | "cuda".
            debug_dump_path: Optional JSON file to write a compact assignment report.
            seed_map: Optional {(shot_id, segment_id) -> fixed_global_id}.
            max_logged_edges: Cap similarity-edge logs to avoid giant logs.
        """
        self.embedding_threshold = float(embedding_threshold)
        self._threshold_tol = 1e-6
        self.debug_dump_path = str(debug_dump_path) if debug_dump_path else None
        self.seed_map = dict(seed_map) if seed_map else {}
        self.max_logged_edges = int(max_logged_edges)

        # Device selection
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        else:
            self.device = device

        self.logger = logging.getLogger("facekit.resolver")
        self.logger.info(
            "[resolver] init device=%s embed_thresh=%.3f dump=%s",
            self.device, self.embedding_threshold, self.debug_dump_path
        )

    def resolve_global_ids(self, tracks: List[FaceTrack], start_id: int = 0) -> int:
        """
        Assign global_ids to tracks via union-find clustering of per-(shot,segment) groups.
        Returns the next available global_id after assignment.
        """
        # ---------- helpers ----------
        def robust_center(emb_list: List[np.ndarray], cutoff: float = 0.30) -> np.ndarray:
            E = np.stack(emb_list).astype(np.float32)
            E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
            c = E.mean(axis=0)
            d = 1.0 - (E @ (c / (np.linalg.norm(c) + 1e-9)))
            keep = d <= cutoff
            return (E[keep].mean(axis=0) if keep.any() else c)

        def _to_int_or_none(x):
            try:
                if x is None:
                    return None
                # Protect against inf/NaN coming from some fakes
                if isinstance(x, float):
                    import math
                    if math.isinf(x) or math.isnan(x):
                        return None
                return int(x)
            except (OverflowError, ValueError, TypeError):
                return None

        def first_frame(t: FaceTrack) -> Optional[int]:
            return _to_int_or_none(getattr(t, "first_frame")() if hasattr(t, "first_frame") else None)

        def last_frame(t: FaceTrack) -> Optional[int]:
            return _to_int_or_none(getattr(t, "last_frame")() if hasattr(t, "last_frame") else None)


        # ---- Step 0: group by (shot_id, segment_id) or fall back per-track ----
        groups: Dict[Any, Dict[str, Any]] = {}
        for i, t in enumerate(tracks):
            # Invariant: tracks must have at least one observation
            obs = getattr(t, "observations", None)
            if (not obs) or (hasattr(obs, "__len__") and len(obs) == 0):
                msg = (f"[resolver] Track idx={i} has no observations "
                       f"(shot_id={getattr(t,'shot_id',None)}, "
                       f"segment_id={getattr(t,'segment_id',None)}, "
                       f"track_id={getattr(t,'track_id',None)}). "
                       f"This violates the pipeline invariant.")
                self.logger.error(msg)
                raise ValueError(msg)

            shot_id = getattr(t, "shot_id", None)
            shot_face_id = getattr(t, "segment_id", None)

            key = (shot_id, shot_face_id) if shot_face_id is not None else ("__per_track__", i)
            g = groups.setdefault(key, {
                "members": [], "all_embs": [], "shot_id": shot_id,
                "span_start": None, "span_end": None
            })
            g["members"].append(i)

            embs = getattr(t, "embeddings", None)
            if embs:
                g["all_embs"].extend([np.asarray(e, dtype=np.float32) for e in embs])

            s0, s1 = first_frame(t), last_frame(t)
            g["span_start"] = s0 if g["span_start"] is None else min(g["span_start"], s0)
            g["span_end"]   = s1 if g["span_end"]   is None else max(g["span_end"],   s1)

        for key, g in groups.items():
            self.logger.info(
                "[groups] key=%s members=%s span=[%s..%s] n_embs=%d shot=%s",
                key, g["members"], g["span_start"], g["span_end"], len(g["all_embs"]), g["shot_id"]
            )

        # ---- Step 1: build representatives ----
        group_keys: List[Any] = []
        reps: List[np.ndarray] = []
        shots: List[int] = []
        starts: List[int] = []
        ends: List[int] = []

        for key, g in groups.items():
            if g["all_embs"]:
                rep = robust_center(g["all_embs"])
                group_keys.append(key)
                reps.append(rep)
                shots.append(int(g["shot_id"]) if g["shot_id"] is not None else 10**9)
                # Use very wide sentinels but keep them integers
                starts.append(int(g["span_start"]) if g["span_start"] is not None else -10**9)
                ends.append(int(g["span_end"])     if g["span_end"]   is not None else  10**9)

        # No reps → no embeddings anywhere → assign uniques and return
        if not reps:
            for t in tracks:
                t.global_id = start_id
                start_id += 1
            return start_id

        # Map: group index -> member track indices
        group_members = [groups[k]["members"] for k in group_keys]

        # ---- Step 2: similarity matrix over reps ----
        emb_tensor = torch.tensor(np.stack(reps), dtype=torch.float32, device=self.device)
        emb_tensor = F.normalize(emb_tensor, p=2, dim=1)
        sim_matrix = torch.mm(emb_tensor, emb_tensor.T).detach().cpu().numpy()

        emb_threshold = self.embedding_threshold
        tol = float(self._threshold_tol)
        m = len(group_keys)

        edges: List[Tuple[float, int, int, int, int]] = []
        for i in range(m):
            row = sim_matrix[i]
            for j in range(i + 1, m):
                s = float(row[j])
                if s + tol >= emb_threshold:
                    lo = min(group_members[i]) if group_members[i] else 10**9
                    hi = min(group_members[j]) if group_members[j] else 10**9
                    if lo > hi:
                        lo, hi = hi, lo
                    edges.append((s, i, j, lo, hi))
        edges.sort(key=lambda x: (-x[0], x[3], x[4]))
        for k, (s, i, j, _, _) in enumerate(edges[: self.max_logged_edges]):
            self.logger.info("[edges] s=%.3f i=%d key_i=%s j=%d key_j=%s",
                             s, i, group_keys[i], j, group_keys[j])

        # ---- Step 3: union-find with must-not-link (overlap in same shot) ----
        parent = list(range(m))
        rank = [0] * m
        comp_spans = [{shots[i]: [(starts[i], ends[i])]} for i in range(m)]

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def overlaps(a0, a1, b0, b1) -> bool:
            return not (a1 < b0 or b1 < a0)

        def union(a: int, b: int) -> bool:
            ra, rb = find(a), find(b)
            if ra == rb:
                return False
            spans_a = comp_spans[ra]
            spans_b = comp_spans[rb]
            for sh in set(spans_a.keys()) & set(spans_b.keys()):
                for (sa0, sa1) in spans_a.get(sh, []):
                    for (sb0, sb1) in spans_b.get(sh, []):
                        if overlaps(sa0, sa1, sb0, sb1):
                            return False  # must-not-link
            # union by rank
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1
            # merge span maps
            merged = dict(spans_a)
            for sh, lst in comp_spans[rb].items():
                merged.setdefault(sh, []).extend(lst)
            comp_spans[ra] = merged
            comp_spans[rb] = {}
            return True

        merges = 0
        for s, i, j, _, _ in edges:
            if union(i, j):
                merges += 1

        # ---- Step 4: components and assignment order ----
        root_to_nodes: Dict[int, List[int]] = {}
        for i in range(m):
            r = find(i)
            root_to_nodes.setdefault(r, []).append(i)

        def comp_min_track(nodes: List[int]) -> int:
            mins = []
            for gi in nodes:
                mins.extend(group_members[gi])
            return min(mins) if mins else 10**9

        def comp_min_time(nodes: List[int]) -> int:
            mins = []
            for gi in nodes:
                mins.extend(group_members[gi])
            if not mins:
                return 10**9
            t0s = [tracks[idx].first_frame() for idx in mins]
            return min(t0s) if t0s else 10**9

        components = sorted(root_to_nodes.values(), key=lambda nodes: (comp_min_time(nodes), comp_min_track(nodes)))

        self.logger.info(
            "[assign-order] %s",
            [{"min_time": comp_min_time(n), "min_track": comp_min_track(n)} for n in components]
        )

        # Seeded components first (if component entirely matches a single seeded gid)
        assigned_components = set()
        if self.seed_map:
            for ci, comp in enumerate(components):
                keys = set()
                for local_idx in comp:
                    k = group_keys[local_idx]
                    if isinstance(k, tuple) and len(k) == 2 and k[0] != "__per_track__":
                        keys.add((int(k[0]), int(k[1])))
                if keys and all(k in self.seed_map for k in keys):
                    seeded_ids = {self.seed_map[k] for k in keys}
                    if len(seeded_ids) == 1:
                        gid = list(seeded_ids)[0]
                        for local_idx in comp:
                            for tr_idx in group_members[local_idx]:
                                tracks[tr_idx].global_id = gid
                        assigned_components.add(ci)
                        self.logger.info("[assign] seeded gid=%d keys=%s", gid, sorted(keys))

        # Assign remaining components in the chosen order
        next_id = start_id
        for ci, comp in enumerate(components):
            if ci in assigned_components:
                continue
            for local_idx in comp:
                for tr_idx in group_members[local_idx]:
                    tracks[tr_idx].global_id = next_id
            _mt = comp_min_time(comp)
            # Use %s to avoid OverflowError when _mt is inf/None
            self.logger.info(
                "[assign] gid=%d comp_min_time=%s members=%s",
                next_id, _mt, [group_keys[x] for x in comp]
            )

            next_id += 1

        assigned = {idx for members in group_members for idx in members}

        def _min_time_for_track_idx(i: int) -> int:
            t = tracks[i]
            # Prefer first_frame; if None, try last_frame; else push to end
            f0 = first_frame(t)
            if f0 is not None:
                return int(f0)
            f1 = last_frame(t)
            return int(f1) if f1 is not None else 10**9

        leftovers = [i for i in range(len(tracks)) if i not in assigned]
        leftovers.sort(key=lambda i: (_min_time_for_track_idx(i), i))  # stable tiebreak by original index
        for i in leftovers:
            tracks[i].global_id = next_id
            self.logger.info("[assign] gid=%d (no-emb track idx=%d, min_time=%s)",
                             next_id, i, _min_time_for_track_idx(i))
            next_id += 1


        # Optional JSON debug dump
        if self.debug_dump_path:
            out = []
            gid_to_keys: Dict[int, List[Any]] = {}
            for local_idx, key in enumerate(group_keys):
                for tr_idx in group_members[local_idx]:
                    gid = int(getattr(tracks[tr_idx], "global_id", -1))
                    gid_to_keys.setdefault(gid, []).append(key)
            gid_to_keys = {gid: sorted({tuple(k) if isinstance(k, (list, tuple)) else k for k in v})
                           for gid, v in gid_to_keys.items()}
            for gid, keys in sorted(gid_to_keys.items()):
                mins = []
                for k in keys:
                    for li, gk in enumerate(group_keys):
                        if tuple(gk) == tuple(k):
                            mins.extend(group_members[li])
                min_time = min((tracks[idx].first_frame() for idx in mins), default=10**9)
                out.append({"global_id": gid, "group_keys": keys, "min_time": int(min_time)})
            try:
                Path(self.debug_dump_path).parent.mkdir(parents=True, exist_ok=True)
                with open(self.debug_dump_path, "w") as f:
                    json.dump(out, f, indent=2)
                self.logger.info("[dump] wrote resolver debug JSON to %s", self.debug_dump_path)
            except Exception as e:
                self.logger.exception("[dump] failed writing resolver debug JSON: %s", e)

        return next_id
