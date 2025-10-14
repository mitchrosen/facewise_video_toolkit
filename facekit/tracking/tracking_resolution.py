import numpy as np
import torch
import torch.nn.functional as F
from typing import List
from facekit.tracking.face_structures import FaceTrack
import logging


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

        # Device selection logic
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available. Please check your environment.")
        else:
            self.device = device

        logging.info(f"[INFO] GlobalIdentityResolver initialized on {self.device}")

    def resolve_global_ids(self, tracks: List[FaceTrack], start_id: int = 0) -> int:
        """
        Assign global_ids by clustering group representatives based on embedding similarity, 
        groups consisting of tracks with same segment_id within a shot.  A must-not-link 
        constraint  forbids merging tracks which overlap in time within the same shot. 
        Uses a deterministic, best-first union-find so results are invariant to input order 
        and cannot be 'bridged' across the constraint.

        Args:
            tracks (List[FaceTrack]): All FaceTracks (possibly across shots).
            start_id (int): Starting global_id counter.

        Returns:
            int: The next available global_id after assignment.
        """
        # ---------- helpers ----------
        def robust_center(emb_list: List[np.ndarray], cutoff: float = 0.30) -> np.ndarray:
            E = np.stack(emb_list).astype(np.float32)
            E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
            c = E.mean(axis=0)
            d = 1.0 - (E @ (c / (np.linalg.norm(c) + 1e-9)))
            keep = d <= cutoff
            return (E[keep].mean(axis=0) if keep.any() else c)

        def first_frame(t: FaceTrack) -> int:
            return t.first_frame()

        def last_frame(t: FaceTrack) -> int:
            return t.last_frame()

        # Step 0: Group tracks by (shot_id, segment_id)
        # Fall back to per-track grouping when no segment_id present
        groups = {}  # key -> dict with: members (indices), all_embs, shot_id, span_start, span_end
        for i, t in enumerate(tracks):
            shot_id = getattr(t, "shot_id", None)
            shot_face_id = getattr(t, "segment_id", None)

            if shot_face_id is None:
                # fallback to per-track behavior - track missing segment_id
                key = ("__per_track__", i)
            else:
                key = (shot_id, shot_face_id)

            g = groups.setdefault(key, {
                "members": [], "all_embs": [], "shot_id": shot_id,
                "span_start": None, "span_end": None
            })
            g["members"].append(i)

            embs = getattr(t, "embeddings", None)
            if embs:
                # extend all embeddings for robust averaging
                g["all_embs"].extend([np.asarray(e, dtype=np.float32) for e in embs])

            # accumulate span
            s0 = first_frame(t); s1 = last_frame(t)
            if s0 is not None and s1 is not None:
                g["span_start"] = s0 if g["span_start"] is None else min(g["span_start"], s0)
                g["span_end"]   = s1 if g["span_end"]   is None else max(g["span_end"],   s1)
            
        # Step 1: Build group representatives
        group_keys = []
        reps = []
        shots = []
        starts = []
        ends = []

        for key, g in groups.items():
            if g["all_embs"]:
                rep = robust_center(g["all_embs"])
                group_keys.append(key)
                reps.append(rep)
                shots.append(g["shot_id"])
                starts.append(g["span_start"] if g["span_start"] is not None else -10**9)
                ends.append(g["span_end"]   if g["span_end"]   is not None else  10**9)
        
        # Edge case: no representatives calculated (no embeddings at all) -> assign unique IDs
        if not reps:
           for t in tracks:
                t.global_id = start_id
                start_id += 1
           return start_id
        
        # Map: group index -> member track indices
        group_members = [groups[k]["members"] for k in group_keys]

        # Step 2: Compute similarities among group reps
        emb_tensor = torch.tensor(np.stack(reps), dtype=torch.float32, device=self.device)
        emb_tensor = F.normalize(emb_tensor, p=2, dim=1)               # (n, d)
        sim_matrix = torch.mm(emb_tensor, emb_tensor.T)                # cosine similarity matrix, (n, n)
        sim_matrix_np = sim_matrix.detach().cpu().numpy()              # Pull to CPU numpy once to avoid many device hops

        emb_threshold = float(self.embedding_threshold)
        tol = float(getattr(self, "_threshold_tol", 1e-6))
        m = len(group_keys)

        edges = []
        for i in range(m):
            # Diagonal ignored; only i < j
            row = sim_matrix_np[i]
            for j in range(i + 1, m):
                s = row[j]
                if s + tol >= emb_threshold:
                    # stable ordering by original smallest track index in each group, then by other
                    # (helps determinism in ID numbering)                    
                    lo = min(group_members[i]) if group_members[i] else 10**9
                    hi = min(group_members[j]) if group_members[j] else 10**9
                    if lo > hi:
                        lo, hi = hi, lo
                    edges.append((s, i, j, lo, hi))

        edges.sort(key=lambda x: (-x[0], x[3], x[4]))  # similarity desc, then stable tiebreak


        # Step 3: Union-Find (Disjoint Set) with must-not-link constraint
        parent = list(range(m))
        rank = [0] * m

        # For each component we keep a map: shot_id -> list of (start,end) intervals
        comp_spans = [{shots[i]: [(starts[i], ends[i])]} for i in range(m)]

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> bool:
            """ Union with union-by-rank. Returns True if merged, False otherwise. """
            ra, rb = find(a), find(b)
            if ra == rb:
                return False

            # ---- must-not-link: forbid merging if any same-shot intervals overlap ----
            spans_a = comp_spans[ra]
            spans_b = comp_spans[rb]
            shared_shots = set(spans_a.keys()) & set(spans_b.keys())

            if shared_shots:

                def overlaps(a0, a1, b0, b1):
                    # inclusive overlap
                    return not (a1 < b0 or b1 < a0)

                for sh in shared_shots:
                    la = spans_a.get(sh, [])
                    lb = spans_b.get(sh, [])
                    # If any interval pair overlaps, merging is forbidden
                    for (sa0, sa1) in la:
                        for (sb0, sb1) in lb:
                            if overlaps(sa0, sa1, sb0, sb1):
                                return False  # must-not-link violation

            # If constraint passes, merge by rank and merge span maps
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra  # ensure ra is the new root
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1

            # Merge span maps: concatenate interval lists per shot
            merged = dict(spans_a)
            for sh, lst in spans_b.items():
                merged.setdefault(sh, []).extend(lst)
            comp_spans[ra] = merged
            comp_spans[rb] = {}  # not used anymore
            return True

        merges = 0
        for s, i, j, _, _ in edges:
            if union(i, j):
                merges += 1

        # Step 4: Gather components and assign global IDs
        root_to_nodes = {}
        for i in range(m):
            r = find(i)
            root_to_nodes.setdefault(r, []).append(i)

        # Deterministic component order: by smallest original track index inside each component
        def comp_min_track(nodes: List[int]) -> int:
            mins = []
            for gi in nodes:
                mins.extend(group_members[gi])
            return min(mins) if mins else 10**9

        components = sorted(root_to_nodes.values(), key=comp_min_track)

        for comp in components:
            for local_idx in comp:
                for tr_idx in group_members[local_idx]:   
                    tracks[tr_idx].global_id = start_id
            start_id += 1

        # Step 5: Assign unique IDs to tracks without embeddings
        # (Rare: groups existed but had zero embs; or tracks with no embs at all)
        assigned = {idx for members in group_members for idx in members}
        for i, t in enumerate(tracks):
            if i not in assigned:
                t.global_id = start_id
                start_id += 1

        return start_id
