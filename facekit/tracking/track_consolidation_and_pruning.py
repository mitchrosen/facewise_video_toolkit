# track_consolidation_and_pruning.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import logging
from collections import deque

from facekit.tracking.face_structures import FaceTrack, FaceObservation
from facekit.utils.geometry import compute_iou
from facekit.common.obs_consts import Source

logger = logging.getLogger(__name__)

BBox = Tuple[int, int, int, int]


# =============================================================================
# Public entrypoint (the story)
# =============================================================================

def apply_track_consolidation_and_pruning(
    tracks: list[FaceTrack],
    *,
    min_gap_len: int,
    min_track_len: int,
    iou_threshold: float = 0.2,
) -> list[FaceTrack]:
    """
    Mutates tracks in-place (global_id changes + interpolated observations, plus merges)
    and returns the surviving tracks.
    """
    by_shot = group_tracks_by_shot(tracks)
    out: list[FaceTrack] = []

    for shot_id in sorted(by_shot):
        shot_tracks = by_shot[shot_id]

        assignments = propose_gid_reassignments_for_short_tracks(
            shot_tracks,
            min_gap_len=min_gap_len,
            min_track_len=min_track_len,
            iou_threshold=iou_threshold,
        )
        apply_assigned_global_ids(shot_tracks, assignments)

        shot_tracks = prune_short_tracks(shot_tracks, min_track_len=min_track_len)

        # NOTE: this may merge tracks (reduce count) and inject observations
        fill_short_gaps_within_shot_and_merge(
            shot_tracks,
            min_gap_len=min_gap_len,
            iou_threshold=iou_threshold,
        )

        out.extend(shot_tracks)

    return out


# =============================================================================
# Phase 0: grouping
# =============================================================================

def group_tracks_by_shot(tracks: Iterable[FaceTrack]) -> Dict[int, List[FaceTrack]]:
    by: Dict[int, List[FaceTrack]] = {}
    for t in tracks:
        sid = int(getattr(t, "shot_id", -1))
        by.setdefault(sid, []).append(t)
    return by


def group_tracks_by_gid(tracks: Iterable[FaceTrack]) -> Dict[Optional[int], List[FaceTrack]]:
    by: Dict[Optional[int], List[FaceTrack]] = {}
    for t in tracks:
        by.setdefault(getattr(t, "global_id", None), []).append(t)
    return by


# =============================================================================
# Phase 1: propose + apply global_id reassignment for short tracks (Option 1b)
# =============================================================================

@dataclass(frozen=True)
class GidAssignment:
    track_id: int
    new_global_id: int


@dataclass(frozen=True)
class RankedCandidate:
    gid: int
    score: float


@dataclass
class _ShortTrackPlan:
    track: FaceTrack
    candidates: list[RankedCandidate]
    next_idx: int = 0

    def next_candidate(self) -> Optional[RankedCandidate]:
        if self.next_idx >= len(self.candidates):
            return None
        c = self.candidates[self.next_idx]
        self.next_idx += 1
        return c


def propose_gid_reassignments_for_short_tracks(
    shot_tracks: list[FaceTrack],
    *,
    min_gap_len: int,
    min_track_len: int,
    iou_threshold: float,
) -> list[GidAssignment]:
    """
    Decide global_id reassignment for short tracks (< min_track_len) within this shot.

    Option 1b conflict handling:
      - We do NOT assign a gid that would overlap-in-time an existing *fixed* track with that gid.
      - If two short tracks compete for the same gid and would overlap in time:
          winner keeps the gid, loser is pushed to try its next-best gid.
      - Only after exhausting candidates does a short track remain unassigned (then Phase 2 may prune it).
    """
    if not shot_tracks:
        return []

    ordered = sorted(shot_tracks, key=lambda t: (t.first_frame(), t.track_id))
    short_tracks = [t for t in ordered if track_len(t) < int(min_track_len)]
    if not short_tracks:
        return []

    # "Fixed" tracks are those we will not change gids for in this phase.
    # (i.e., non-short tracks)
    short_ids = {int(t.track_id) for t in short_tracks}
    fixed_tracks = [t for t in ordered if int(t.track_id) not in short_ids]

    fixed_by_gid: Dict[int, list[FaceTrack]] = {}
    for t in fixed_tracks:
        gid = getattr(t, "global_id", None)
        if gid is None:
            continue
        fixed_by_gid.setdefault(int(gid), []).append(t)

    # Precompute candidates for each short track
    plans: Dict[int, _ShortTrackPlan] = {}
    for st in sorted(short_tracks, key=lambda t: (track_len(t), t.first_frame(), t.track_id)):
        cands = rank_candidate_gids_for_short_track(
            st,
            ordered,
            min_gap_len=min_gap_len,
            iou_threshold=iou_threshold,
        )
        plans[int(st.track_id)] = _ShortTrackPlan(track=st, candidates=cands, next_idx=0)

    # State: which short track currently "owns" a gid in this phase
    claimed_by_gid: Dict[int, int] = {}  # gid -> track_id
    assigned_gid_by_tid: Dict[int, int] = {}

    # Work queue: short tracks to attempt assignment (deterministic order)
    q = deque(sorted(short_tracks, key=lambda t: (track_len(t), t.first_frame(), t.track_id)))

    # Score lookup for deterministic winner selection
    score_cache: Dict[Tuple[int, int], float] = {}  # (tid,gid) -> score

    def _score(tid: int, gid: int) -> float:
        key = (int(tid), int(gid))
        if key in score_cache:
            return score_cache[key]
        plan = plans.get(int(tid))
        if not plan:
            score_cache[key] = float("-inf")
            return score_cache[key]
        for c in plan.candidates:
            if int(c.gid) == int(gid):
                score_cache[key] = float(c.score)
                return score_cache[key]
        score_cache[key] = float("-inf")
        return score_cache[key]

    def _fixed_overlap_blocks(t: FaceTrack, gid: int) -> bool:
        for fx in fixed_by_gid.get(int(gid), []):
            if ranges_overlap(*span(t), *span(fx)):
                return True
        return False

    def _short_overlap_conflict(t: FaceTrack, other_tid: int, gid: int) -> bool:
        other = plans[int(other_tid)].track
        return ranges_overlap(*span(t), *span(other))

    def _winner_tid(tid_a: int, tid_b: int, gid: int) -> int:
        """
        Deterministic:
          - higher score wins
          - if tie, longer track wins
          - if tie, earlier start wins
          - if tie, lower tid wins
        """
        sa = _score(tid_a, gid)
        sb = _score(tid_b, gid)
        if sa != sb:
            return tid_a if sa > sb else tid_b

        ta = plans[int(tid_a)].track
        tb = plans[int(tid_b)].track
        la = track_len(ta)
        lb = track_len(tb)
        if la != lb:
            return tid_a if la > lb else tid_b

        fa = int(ta.first_frame())
        fb = int(tb.first_frame())
        if fa != fb:
            return tid_a if fa < fb else tid_b

        return tid_a if int(tid_a) < int(tid_b) else tid_b

    # Safety to avoid infinite loops if something goes sideways
    max_iters = 50_000
    iters = 0

    while q and iters < max_iters:
        iters += 1
        t = q.popleft()
        tid = int(t.track_id)

        # If already assigned (e.g., came back in queue and got fixed), skip
        if tid in assigned_gid_by_tid:
            continue

        plan = plans.get(tid)
        if plan is None:
            continue

        while True:
            cand = plan.next_candidate()
            if cand is None:
                # no assignment possible
                break

            gid = int(cand.gid)

            # Block if it would overlap any fixed track with this gid
            if _fixed_overlap_blocks(t, gid):
                continue

            # If gid not claimed yet, claim it
            other_tid = claimed_by_gid.get(gid)
            if other_tid is None:
                claimed_by_gid[gid] = tid
                assigned_gid_by_tid[tid] = gid
                break

            # If claimed, check overlap; if no overlap, allow both to share gid in shot? NO.
            # Your constraint is: no two tracks in a shot with same gid can overlap in time.
            if not _short_overlap_conflict(t, other_tid, gid):
                # They do not overlap in time; allow both to have gid.
                # Keep claimed_by_gid as-is (it just records "one owner"), but we can still assign.
                assigned_gid_by_tid[tid] = gid
                break

            # Conflict: choose winner; loser tries its next best
            win = _winner_tid(tid, other_tid, gid)
            lose = other_tid if win == tid else tid

            if win == other_tid:
                # Current track loses; try another candidate
                continue

            # Current track wins: it gets gid, other gets unassigned and re-queued
            claimed_by_gid[gid] = tid
            assigned_gid_by_tid[tid] = gid

            # Remove old assignment (if any) for loser
            if int(lose) in assigned_gid_by_tid:
                del assigned_gid_by_tid[int(lose)]
            # Re-queue loser to find next best gid
            q.append(plans[int(lose)].track)
            break

    if iters >= max_iters:
        logger.warning("gid reassignment: hit max_iters=%d in shot=%s; partial assignments may result.", max_iters, getattr(shot_tracks[0], "shot_id", None))

    # Emit assignments only when it changes gid
    assignments: list[GidAssignment] = []
    by_tid = {int(t.track_id): t for t in shot_tracks}
    for tid, new_gid in assigned_gid_by_tid.items():
        tr = by_tid.get(int(tid))
        if tr is None:
            continue
        old_gid = getattr(tr, "global_id", None)
        if old_gid is None or int(old_gid) != int(new_gid):
            assignments.append(GidAssignment(track_id=int(tid), new_global_id=int(new_gid)))

    # deterministic order
    assignments.sort(key=lambda a: int(a.track_id))
    return assignments


def rank_candidate_gids_for_short_track(
    short_t: FaceTrack,
    ordered_shot_tracks: list[FaceTrack],
    *,
    min_gap_len: int,
    iou_threshold: float,
) -> list[RankedCandidate]:
    """
    Returns candidates sorted best-first, deterministic on ties.
    Score is mainly IoU evidence, plus a small sandwich bonus if applicable.
    """
    idx = ordered_shot_tracks.index(short_t)
    lefts = ordered_shot_tracks[:idx]
    rights = ordered_shot_tracks[idx + 1 :]

    left_caps = nearby_left_caps(short_t, lefts, min_gap_len=min_gap_len)
    right_caps = nearby_right_caps(short_t, rights, min_gap_len=min_gap_len)

    scores: Dict[int, float] = {}

    def bump(gid: Optional[int], score: Optional[float]) -> None:
        if gid is None or score is None:
            return
        gid = int(gid)
        scores[gid] = max(scores.get(gid, float("-inf")), float(score))

    # one-sided evidence
    for lt in left_caps:
        bump(getattr(lt, "global_id", None), score_left_cap(lt, short_t, iou_threshold=iou_threshold))
    for rt in right_caps:
        bump(getattr(rt, "global_id", None), score_right_cap(short_t, rt, iou_threshold=iou_threshold))

    # sandwich: same gid appears on both sides -> small bonus, but not dominant
    sandwich = encompassing_sandwich_candidates(left_caps, right_caps)
    for gid, (lt, rt) in sandwich.items():
        iou_l = safe_iou(lt.get_last_bbox(), short_t.get_first_bbox())
        iou_r = safe_iou(short_t.get_last_bbox(), rt.get_first_bbox())
        if iou_l is None or iou_r is None:
            continue
        if iou_l < iou_threshold or iou_r < iou_threshold:
            continue
        base = scores.get(int(gid), 0.0)
        bonus = 0.05 + 0.5 * min(iou_l, iou_r)
        scores[int(gid)] = max(scores.get(int(gid), float("-inf")), base + bonus)

    ranked = [RankedCandidate(gid=k, score=v) for k, v in scores.items()]
    ranked.sort(key=lambda c: (-c.score, int(c.gid)))
    return ranked


def apply_assigned_global_ids(
    shot_tracks: list[FaceTrack],
    assignments: list[GidAssignment],
) -> None:
    if not assignments:
        return

    by_tid = {int(t.track_id): t for t in shot_tracks}
    for a in assignments:
        tr = by_tid.get(int(a.track_id))
        if tr is None:
            continue
        old = getattr(tr, "global_id", None)
        tr.global_id = int(a.new_global_id)
        logger.debug("gid reassignment: shot=%s tid=%s %r -> %r", tr.shot_id, tr.track_id, old, tr.global_id)


# =============================================================================
# Phase 2: prune short tracks
# =============================================================================

def prune_short_tracks(
    shot_tracks: list[FaceTrack],
    *,
    min_track_len: int,
) -> list[FaceTrack]:
    """
    Keep:
      - any track len >= min_track_len
      - short tracks that have a non-None global_id
    Drop:
      - short tracks with global_id None
    """
    out: list[FaceTrack] = []
    for t in shot_tracks:
        if track_len(t) >= int(min_track_len):
            out.append(t)
            continue
        if getattr(t, "global_id", None) is None:
            logger.debug("prune short track: shot=%s tid=%s len=%s", t.shot_id, t.track_id, track_len(t))
            continue
        out.append(t)
    return out


# =============================================================================
# Phase 3: fill short gaps between same gid tracks (and MERGE)
# =============================================================================

def fill_short_gaps_within_shot_and_merge(
    shot_tracks: list[FaceTrack],
    *,
    min_gap_len: int,
    iou_threshold: float,
) -> None:
    """
    For each global_id group within this shot:
      - sort tracks by time
      - for consecutive tracks (A then B):
          gap = B.first - A.last - 1
          if 0 < gap < min_gap_len:
             require boundary IoU(A.last_bbox, B.first_bbox) >= iou_threshold
             require NO occupancy in any gap frame by any track in this shot
             interpolate bboxes into A for all gap frames
             merge B into A, remove B from shot_tracks
    """
    if not shot_tracks or int(min_gap_len) <= 0:
        return

    occupied = build_occupied_frames(shot_tracks)

    # We will mutate shot_tracks (remove merged rights). So loop carefully.
    by_gid = group_tracks_by_gid(shot_tracks)
    for gid, group in list(by_gid.items()):
        if gid is None:
            continue

        ordered = sorted(group, key=lambda t: (t.first_frame(), t.track_id))
        i = 0
        while i < len(ordered) - 1:
            left = ordered[i]
            right = ordered[i + 1]

            gap_start, gap_end = compute_gap(left, right)
            if gap_start is None or gap_end is None:
                i += 1
                continue

            gap_len = gap_end - gap_start + 1
            if gap_len <= 0:
                i += 1
                continue
            if gap_len >= int(min_gap_len):
                i += 1
                continue

            if not boundary_iou_passes(left, right, iou_threshold=iou_threshold):
                i += 1
                continue

            if gap_has_any_occupancy(gap_start, gap_end, occupied):
                # No skipping. If blocked, do nothing.
                i += 1
                continue

            # Interpolate + merge
            inject_interpolated_gap_no_skips(
                left=left,
                right=right,
                gap_start=gap_start,
                gap_end=gap_end,
            )
            merge_right_into_left(left, right)

            # Remove right from the main shot list
            try:
                shot_tracks.remove(right)
            except ValueError:
                pass

            # Also remove from our local ordered list; keep left at same index
            ordered.pop(i + 1)

            # Update occupancy for the newly injected gap frames so later merges
            # cannot "double-inject" into the same frames.
            for f in range(int(gap_start), int(gap_end) + 1):
                occupied[int(f)] = max(occupied.get(int(f), 0), 1)

            # Do not i += 1; there may be another consecutive track to merge into the same left
            continue

        # end while


# =============================================================================
# Low-level helpers (geometry / time / occupancy)
# =============================================================================

def track_len(t: FaceTrack) -> int:
    try:
        return int(t.duration())
    except Exception:
        a = int(t.first_frame())
        b = t.last_frame()
        b = int(b) if b is not None else a
        return max(0, b - a + 1)


def compute_gap(left: FaceTrack, right: FaceTrack) -> Tuple[Optional[int], Optional[int]]:
    l_last = left.last_frame()
    if l_last is None:
        return None, None
    r_first = int(right.first_frame())
    gap_start = int(l_last) + 1
    gap_end = int(r_first) - 1
    return gap_start, gap_end


def build_occupied_frames(shot_tracks: list[FaceTrack]) -> Dict[int, int]:
    """
    frame -> number of tracks having an observation at that frame (any source).
    """
    occ: Dict[int, int] = {}
    for t in shot_tracks:
        for f in t.get_frame_indices():
            occ[int(f)] = occ.get(int(f), 0) + 1
    return occ


def gap_has_any_occupancy(gap_start: int, gap_end: int, occupied: Dict[int, int]) -> bool:
    for f in range(int(gap_start), int(gap_end) + 1):
        if occupied.get(int(f), 0) > 0:
            return True
    return False


def boundary_iou_passes(left: FaceTrack, right: FaceTrack, *, iou_threshold: float) -> bool:
    a = left.get_last_bbox()
    b = right.get_first_bbox()
    iou = safe_iou(a, b)
    return (iou is not None) and (float(iou) >= float(iou_threshold))


def safe_iou(a: Optional[BBox], b: Optional[BBox]) -> Optional[float]:
    if a is None or b is None:
        return None
    try:
        return float(compute_iou(a, b))
    except Exception:
        return None


def inject_interpolated_gap_no_skips(
    *,
    left: FaceTrack,
    right: FaceTrack,
    gap_start: int,
    gap_end: int,
) -> None:
    """
    Adds INTERPOLATED observations into `left` for ALL frames gap_start..gap_end.
    No skipping; caller must have already verified occupancy is clear.
    """
    a_bb = left.get_last_bbox()
    b_bb = right.get_first_bbox()
    if a_bb is None or b_bb is None:
        return

    frames = list(range(int(gap_start), int(gap_end) + 1))
    bboxes = interpolate_bboxes(a_bb, b_bb, len(frames))

    for f, bb in zip(frames, bboxes):
        f = int(f)

        # Defensive: do not overwrite
        if left.get_bbox_by_frame(f) is not None:
            continue

        obs = FaceObservation(
            frame_idx=f,
            bbox=bb,
            source=Source.INTERPOLATED,
            track_id=int(left.track_id),
            confidence=None,
            embedding=None,
            aligned_face=None,
            landmarks=None,
        )
        try:
            obs.shot_id = int(left.shot_id)
        except Exception:
            pass

        left.add_observation(obs, allow_closed=True)

    _normalize_track_order(left)


def merge_right_into_left(left: FaceTrack, right: FaceTrack) -> None:
    """
    Merge `right` track into `left` track:
      - move all observations from right into left (preserving frame indices)
      - keep left.track_id as the survivor
      - mark right closed (optional) but primarily caller removes it from list
    """
    # Ensure right starts after left ends (should be true for a "gap then right" case)
    for obs in (right.observations or []):
        # do not allow frame collisions
        if left.get_bbox_by_frame(int(obs.frame_idx)) is not None:
            # If this happens, caller's invariants were violated; refuse to merge silently.
            raise ValueError(
                f"merge would collide at frame={obs.frame_idx} "
                f"(left tid={left.track_id}, right tid={right.track_id})"
            )

        # mutate obs to reflect merged identity
        obs.track_id = int(left.track_id)
        try:
            obs.shot_id = int(left.shot_id)
        except Exception:
            pass

        left.add_observation(obs, allow_closed=True)

    _normalize_track_order(left)

    # close the right (defensive)
    try:
        right.mark_closed()
    except Exception:
        pass


def _normalize_track_order(t: FaceTrack) -> None:
    """
    Ensure observations are in ascending frame order.
    Your FaceTrack maintains a _frame_index_map; we rebuild it deterministically.
    """
    obs = sorted(list(t.observations or []), key=lambda o: int(o.frame_idx))
    t.observations = obs

    # Rebuild the frame map if present
    if hasattr(t, "_frame_index_map"):
        m = {}
        for o in obs:
            m[int(o.frame_idx)] = o
        t._frame_index_map = m


def interpolate_bboxes(a: BBox, b: BBox, n: int) -> list[BBox]:
    """
    n missing frames. For missing frame k=1..n, t=k/(n+1)
    """
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)

    out: list[BBox] = []
    for k in range(1, n + 1):
        t = k / float(n + 1)
        x1 = ax1 + (bx1 - ax1) * t
        y1 = ay1 + (by1 - ay1) * t
        x2 = ax2 + (bx2 - ax2) * t
        y2 = ay2 + (by2 - ay2) * t
        out.append((int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))))
    return out


# =============================================================================
# Candidate neighborhood helpers
# =============================================================================

def nearby_left_caps(short_t: FaceTrack, lefts: list[FaceTrack], *, min_gap_len: int) -> list[FaceTrack]:
    """
    Left candidates that end within min_gap_len of short_t.first_frame().
    """
    s0 = int(short_t.first_frame())
    out: list[FaceTrack] = []
    for t in reversed(lefts):
        t1 = t.last_frame()
        if t1 is None:
            continue
        gap = s0 - int(t1) - 1
        if gap < 0:
            continue
        if gap < int(min_gap_len):
            out.append(t)
        else:
            break
    return out


def nearby_right_caps(short_t: FaceTrack, rights: list[FaceTrack], *, min_gap_len: int) -> list[FaceTrack]:
    """
    Right candidates that start within min_gap_len of short_t.last_frame().
    """
    s1 = short_t.last_frame()
    if s1 is None:
        return []
    s1 = int(s1)
    out: list[FaceTrack] = []
    for t in rights:
        t0 = int(t.first_frame())
        gap = t0 - s1 - 1
        if gap < 0:
            continue
        if gap < int(min_gap_len):
            out.append(t)
        else:
            break
    return out


def encompassing_sandwich_candidates(
    left_caps: list[FaceTrack],
    right_caps: list[FaceTrack],
) -> Dict[int, Tuple[FaceTrack, FaceTrack]]:
    """
    gid -> (closest_left, closest_right) when gid appears on both sides.
    """
    left_by_gid: Dict[int, FaceTrack] = {}
    for t in left_caps:
        gid = getattr(t, "global_id", None)
        if gid is None:
            continue
        gid = int(gid)
        if gid not in left_by_gid or int(t.last_frame() or -1) > int(left_by_gid[gid].last_frame() or -1):
            left_by_gid[gid] = t

    right_by_gid: Dict[int, FaceTrack] = {}
    for t in right_caps:
        gid = getattr(t, "global_id", None)
        if gid is None:
            continue
        gid = int(gid)
        if gid not in right_by_gid or int(t.first_frame()) < int(right_by_gid[gid].first_frame()):
            right_by_gid[gid] = t

    out: Dict[int, Tuple[FaceTrack, FaceTrack]] = {}
    for gid, l in left_by_gid.items():
        r = right_by_gid.get(gid)
        if r is not None:
            out[gid] = (l, r)
    return out


def score_left_cap(left: FaceTrack, short_t: FaceTrack, *, iou_threshold: float) -> Optional[float]:
    iou = safe_iou(left.get_last_bbox(), short_t.get_first_bbox())
    if iou is None or iou < float(iou_threshold):
        return None
    return float(iou)


def score_right_cap(short_t: FaceTrack, right: FaceTrack, *, iou_threshold: float) -> Optional[float]:
    iou = safe_iou(short_t.get_last_bbox(), right.get_first_bbox())
    if iou is None or iou < float(iou_threshold):
        return None
    return float(iou)


# =============================================================================
# Overlap helpers (time overlap constraint)
# =============================================================================

def span(t: FaceTrack) -> Tuple[int, int]:
    a = int(t.first_frame())
    b = t.last_frame()
    b = int(b) if b is not None else a
    return a, b


def ranges_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 < b0 or b1 < a0)
