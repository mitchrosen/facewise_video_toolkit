import numpy as np
from typing import Optional, List
from dataclasses import dataclass
from typing import Tuple, Dict, Any
import math
import torch
import torch.nn.functional as F
import logging

from facekit.tracking.face_structures import FaceTrack

def _first_abs_frame(track) -> int:
    # Track may expose helpers; else infer from observations.
    if hasattr(track, "first_frame") and callable(getattr(track, "first_frame")):
        return int(track.first_frame())
    obs = getattr(track, "observations", None) or []
    if not obs:
        return 1 << 30
    return int(getattr(obs[0], "frame_idx", 1 << 30))

@dataclass
class _Comp:
    tracks: list
    earliest: int

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

        # self._audit = bool(int(os.environ.get("FACEKIT_GI_AUDIT", "0")))
        self._audit = 1
        logging.info(f"[INFO] GlobalIdentityResolver initialized on {self.device} (audit={self._audit})")

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
        def track_key(t: FaceTrack) -> Tuple[int, int, int, int, int]:
            """
            Stable key for determinism across cold/resume.
            Avoid using list indices and track_id.
            Use track content:
              - shot_id
              - segment_id (if present; else large sentinel)
              - first_frame, last_frame
              - first bbox (rounded to int pixels) as an additional stable tie-break
            """
            shot = int(getattr(t, "shot_id", -1))
            seg  = getattr(t, "segment_id", None)
            segk = int(seg) if seg is not None else (1 << 30)
            try:
                f0 = int(t.first_frame())
            except Exception:
                obs = getattr(t, "observations", None) or []
                f0 = int(getattr(obs[0], "frame_idx", 1 << 30)) if obs else (1 << 30)
            try:
                f1 = int(t.last_frame())
            except Exception:
                obs = getattr(t, "observations", None) or []
                f1 = int(getattr(obs[-1], "frame_idx", -1)) if obs else -1

            # geometry tie-break: first bbox (int pixels), if available
            x1 = y1 = x2 = y2 = (1 << 30)
            obs = getattr(t, "observations", None) or []
            if obs:
                bb = getattr(obs[0], "bbox", None)
                if bb is not None and len(bb) == 4:
                    try:
                        x1, y1, x2, y2 = (int(round(float(bb[0]))),
                                          int(round(float(bb[1]))),
                                          int(round(float(bb[2]))),
                                          int(round(float(bb[3]))))
                    except Exception:
                        pass

            return (shot, segk, f0, f1, x1, y1, x2, y2)

        def robust_center(emb_list: List[np.ndarray], cutoff: float = 0.30) -> Optional[np.ndarray]:
            E = np.stack(emb_list).astype(np.float32)
            # row-normalize to unit length
            norms = np.linalg.norm(E, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            E = E / norms
            c = E.mean(axis=0)
            cn = np.linalg.norm(c)
            if not np.isfinite(cn) or cn == 0.0:
                return None
            # trim outliers by cosine distance to the provisional center
            c_hat = c / cn
            d = 1.0 - (E @ c_hat)
            keep = d <= cutoff
            c2 = (E[keep].mean(axis=0) if keep.any() else c_hat)
            cn2 = np.linalg.norm(c2)
            if not np.isfinite(cn2) or cn2 == 0.0:
                return None
            return (c2 / cn2)

        def first_frame(t: FaceTrack) -> int:
            return t.first_frame()

        def last_frame(t: FaceTrack) -> int:
            return t.last_frame()
        
        # ----------- audit helpers -------------
        def _obs_mix(t: FaceTrack):
            from facekit.common.obs_consts import Source
            n_det = n_trk = 0
            for o in getattr(t, "observations", []) or []:
                src = getattr(o, "source", None)
                if src == Source.DETECTED: n_det += 1
                elif src == Source.TRACKED: n_trk += 1
            return n_det, n_trk

        def _emb_stats(t: FaceTrack):
            embs = getattr(t, "embeddings", None) or []
            n = len(embs)
            if n == 0:
                return 0, 0, "0.000"
            nan_cnt = sum(0 if (e is not None and np.isfinite(e).all()) else 1 for e in embs)
            norms = []
            for e in embs:
                if e is None: continue
                v = float(np.linalg.norm(np.asarray(e, dtype=np.float32)))
                if math.isfinite(v): norms.append(v)
            avg_norm = (sum(norms)/len(norms) if norms else 0.0)
            return n, nan_cnt, f"{avg_norm:.3f}"

        def _tspan(t: FaceTrack):
            try:
                return int(t.first_frame()), int(t.last_frame())
            except Exception:
                obs = getattr(t, "observations", None) or []
                if not obs: return (1<<30, -1)
                return int(obs[0].frame_idx), int(obs[-1].frame_idx)

        # --- Preconditions: every track must have >= 1 observation ---
        # Treat falsy (None or []) as invalid. Raise ValueError (as tests expect).
        missing = [getattr(t, "track_id", None) for t in tracks
                if not getattr(t, "observations", None)]
        if missing:
            raise ValueError(
                "Track(s) have no observations; resolver requires >=1 observation per track. "
                f"Offenders track_id={missing}"
            )
        
        if self._audit:
            logging.info("GI-AUDIT INPUT: count=%d", len(tracks))
            # Deterministic order for diffs
            for t in sorted(tracks, key=track_key):
                s  = int(getattr(t, "shot_id", -1))
                sg = getattr(t, "segment_id", None)
                tid= int(getattr(t, "track_id", -1))
                f0,f1 = _tspan(t)
                n_obs = len(getattr(t, "observations", []) or [])
                n_det,n_trk = _obs_mix(t)
                n_emb, n_nan, avg_norm = _emb_stats(t)
                logging.info("GI-IN: shot=%d seg=%s tid=%d span=[%d..%d] n_obs=%d det=%d trk=%d emb=%d nan=%d avg_norm=%s",
                             s, (str(int(sg)) if sg is not None else "-"), tid, f0, f1,
                             n_obs, n_det, n_trk, n_emb, n_nan, avg_norm)

        # Step 0: Group tracks by (shot_id, segment_id)
        # Fall back to per-track grouping when no segment_id present
        groups: Dict[Any, dict] = {}  # key -> dict with: members (tracks), member_keys, all_embs, shot_id, span_start, span_end
        for t in tracks:
            shot_id = getattr(t, "shot_id", None)
            shot_face_id = getattr(t, "segment_id", None)

            if shot_face_id is None:
                # fallback to per-track behavior - use stable key so it is deterministic
                key = ("__per_track__", track_key(t))
            else:
                key = (shot_id, shot_face_id)

            g = groups.setdefault(key, {
                "members": [], "member_keys": [], "all_embs": [], "shot_id": shot_id,
                "span_start": None, "span_end": None
            })
            g["members"].append(t)
            g["member_keys"].append(track_key(t))

            embs = getattr(t, "embeddings", None)
            if embs:
                # extend all embeddings for robust averaging
                for e in embs:
                    if e is None:
                        continue
                    arr = np.asarray(e, dtype=np.float32)
                    if arr.ndim != 1:
                        continue
                    if not np.isfinite(arr).all():
                        continue
                    g["all_embs"].append(arr)

            # accumulate span
            s0 = first_frame(t); s1 = last_frame(t)
            if s0 is not None and s1 is not None:
                g["span_start"] = s0 if g["span_start"] is None else min(g["span_start"], s0)
                g["span_end"]   = s1 if g["span_end"]   is None else max(g["span_end"],   s1)
            
        if self._audit:
            def _group_sort_key(item):
                key, g = item
                if key[0] == "__per_track__":
                    # put per-track groups after real (shot, seg) groups; stable by the embedded track_key tuple
                    return (1, key[1])
                shot_id, seg_id = key
                shot_sort = int(shot_id) if shot_id is not None else -1
                seg_sort  = int(seg_id)  if seg_id  is not None else (1 << 30)
                return (0, shot_sort, seg_sort)

            for key, g in sorted(groups.items(), key=_group_sort_key):
                if key[0] == "__per_track__":
                    logging.info("GI-GROUP: per_track key=%s members=%s", key[1], g["members"])
                else:
                    logging.info("GI-GROUP: shot=%d seg=%s members=%s span=[%s..%s] embs=%d",
                                 int(g["shot_id"]) if g["shot_id"] is not None else -1,
                                 (str(int(key[1])) if key[1] is not None else "-"),
                                 g["members"], str(g["span_start"]), str(g["span_end"]), len(g["all_embs"]))


        # Step 1: Build group representatives
        group_keys = []
        reps = []
        shots = []
        starts = []
        ends = []

        def _rep_group_sort_key(item):
            key, g = item
            if key[0] == "__per_track__":
                # key[1] is already a stable track_key tuple
                return (1, key[1])
            shot_id, seg_id = key
            shot_sort = int(shot_id) if shot_id is not None else -1
            seg_sort  = int(seg_id)  if seg_id  is not None else (1 << 30)
            # add min member key for extra determinism if you ever get duplicate (shot, seg) keys by mistake
            min_member = min(g["member_keys"]) if g["member_keys"] else (1<<30,)*5
            return (0, shot_sort, seg_sort, min_member)

        for key, g in sorted(groups.items(), key=_rep_group_sort_key):
            if g["all_embs"]:
                rep = robust_center(g["all_embs"])
                if rep is None:
                    # Pathological embeddings: skip this group (falls back to leftover assignment)
                    continue
                group_keys.append(key)
                reps.append(rep)
                # Default unknown shot to -1 so must-not-link only triggers on real matches
                shots.append(int(g["shot_id"]) if g["shot_id"] is not None else -1)
                starts.append(g["span_start"] if g["span_start"] is not None else -10**9)
                ends.append(g["span_end"]   if g["span_end"]   is not None else  10**9)
        
        # Edge case: no representatives calculated (no embeddings at all) -> assign unique IDs
        if not reps:
           for t in tracks:
                t.global_id = start_id
                start_id += 1
                if self._audit:
                    logging.info("GI-AUDIT: no reps -> assigned unique global_ids up to %d", start_id-1)

           return start_id
        
        # Map: group index -> member track indices
        group_members = [groups[k]["members"] for k in group_keys]          # List[List[FaceTrack]]
        group_member_keys = [groups[k]["member_keys"] for k in group_keys]  # List[List[track_key]]

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
                    # stable ordering by smallest stable track_key in each group
                    lo = min(group_member_keys[i]) if group_member_keys[i] else (1<<30,)*8
                    hi = min(group_member_keys[j]) if group_member_keys[j] else (1<<30,)*8
                    if lo > hi: lo, hi = hi, lo
                    edges.append((s, i, j, lo, hi))

        edges.sort(key=lambda x: (-float(x[0]), x[3], x[4]))  # similarity desc, then stable tiebreak
        if self._audit:
            logging.info("GI-EDGES: threshold=%.3f candidates=%d", emb_threshold, len(edges))
            for s, i, j, lo, hi in edges:
                logging.info("GI-EDGE: sim=%.4f i=%d j=%d tiebreak=(%s,%s)", float(s), i, j, lo, hi)

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
            shared_shots = (set(spans_a.keys()) & set(spans_b.keys())) - {-1}

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
                                if self._audit:
                                    logging.info("GI-BLOCK: shot=%d A=[%d..%d] B=[%d..%d]", int(sh), int(sa0), int(sa1), int(sb0), int(sb1))
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
            if self._audit:
                logging.info("GI-UNION: a=%d b=%d -> root=%d", int(a), int(b), int(ra))
            return True

        merges = 0
        for s, i, j, _, _ in edges:
            if union(i, j):
                merges += 1
        if self._audit:
            logging.info("GI-UNION-SUMMARY: merged=%d / candidates=%d / groups=%d", merges, len(edges), m)

        # Step 4: Gather components and assign global IDs
        root_to_nodes = {}
        for i in range(m):
            r = find(i)
            root_to_nodes.setdefault(r, []).append(i)

        # Deterministic component order: by earliest absolute frame among a component's members
        def comp_earliest_frame(nodes: List[int]) -> int:
            earliest = 10**9
            for gi in nodes:
                # each group idx -> its span_start in `starts`
                earliest = min(earliest, starts[gi] if starts[gi] is not None else 10**9)
            return earliest

        # Stable ordering: earliest frame, then smallest member track index in the component
        def comp_tiebreak(nodes: List[int]) -> Tuple[int, int, int, int, int]:
            # choose the smallest stable key present in this component
            # (now independent of transient track_id)
            return min(min(group_member_keys[gi]) for gi in nodes if group_member_keys[gi])
        components = sorted(
            root_to_nodes.values(),
            key=lambda nodes: (comp_earliest_frame(nodes), comp_tiebreak(nodes))
        )
 
        gid_assignments = []  # for audit dump
        for comp in components:
            for local_idx in comp:
                for tr in group_members[local_idx]:
                    tr.global_id = start_id
                    gid_assignments.append((start_id,
                        int(getattr(tr, "shot_id", -1)),
                        int(getattr(tr, "segment_id", -1) if getattr(tr, "segment_id", None) is not None else -1),
                        int(getattr(tr, "track_id", -1)),
                        _tspan(tr)[0],
                        _tspan(tr)[1],
                    ))
            start_id += 1
        if self._audit:
            for gid, shot, seg, tid, f0, f1 in sorted(gid_assignments, key=lambda r: (r[0], r[1], r[4])):
                logging.info("GI-ASSIGN: gid=%d shot=%d seg=%s tid=%d span=[%d..%d]",
                             gid, shot, ("-" if seg < 0 else str(seg)), tid, f0, f1)


        # Step 5: Assign unique IDs to tracks without embeddings
        # (Rare: groups existed but had zero embs; or tracks with no embs at all)
        assigned = {id(t) for members in group_members for t in members}
        # Assign leftovers (tracks without any group rep / embeddings) AFTER components, by earliest frame
        leftovers = [t for t in tracks if id(t) not in assigned]
        def _leftover_sort_key(t: FaceTrack):
            try:
                f0 = int(t.first_frame())
            except Exception:
                obs = getattr(t, "observations", None) or []
                f0 = int(getattr(obs[0], "frame_idx", 1 << 30)) if obs else (1 << 30)
            return (f0, track_key(t))
        leftovers.sort(key=_leftover_sort_key)
        for t in leftovers:
            t.global_id = start_id
            start_id += 1

        if self._audit and leftovers:
            logging.info("GI-LEFTOVERS: assigned %d unique ids up to %d", len(leftovers), start_id-1)

        return start_id