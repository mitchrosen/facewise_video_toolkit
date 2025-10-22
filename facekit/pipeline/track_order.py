from __future__ import annotations
from typing import Dict, Tuple, List, Any

ShotTrack = Tuple[int, int]  # (shot, track_id)
ShotTrackOrderDict = Dict[ShotTrack, int]
ShotTrackOrderList = List[Dict[str, int]]

class TrackOrderError(ValueError):
    pass

def track_order_dict_to_list(
    shot_track_to_order: ShotTrackOrderDict
) -> ShotTrackOrderList:
    """
    Convert {(shot, track_id): order, ...} -> [{"shot": s, "track_id": t, "order": o}, ...]
    Sorted deterministically by (order, shot, track_id).
    """
    items = [
        {"shot": int(s), "track_id": int(t), "order": int(o)}
        for (s, t), o in shot_track_to_order.items()
    ]
    items.sort(key=lambda x: (x["order"], x["shot"], x["track_id"]))
    return items

def track_order_list_to_dict(
    track_order_list: Any, *, strict: bool = True
) -> Tuple[ShotTrackOrderDict, int]:
    """
    Convert list back to dict and compute next_order.
    Validates shape & duplicates. If strict=False, best-effort sanitize.
    Returns: (dict, next_order)
    """
    d: ShotTrackOrderDict = {}
    max_order = -1

    if not isinstance(track_order_list, list):
        if strict:
            raise TrackOrderError("track_order must be a list.")
        return {}, 0

    seen_pairs: set[ShotTrack] = set()
    seen_orders: set[int] = set()

    for i, entry in enumerate(track_order_list):
        if not isinstance(entry, dict):
            if strict:
                raise TrackOrderError(f"track_order[{i}] is not an object.")
            else:
                continue

        try:
            s = int(entry["shot"])
            t = int(entry["track_id"])
            o = int(entry["order"])
        except Exception as e:
            if strict:
                raise TrackOrderError(f"track_order[{i}] missing/invalid fields: {e}")
            else:
                continue

        if s < 1 or t < 0 or o < 0:
            if strict:
                raise TrackOrderError(f"track_order[{i}] out-of-range values: {entry}")
            else:
                continue

        key = (s, t)
        if key in seen_pairs:
            if strict:
                raise TrackOrderError(f"Duplicate (shot,track_id) in track_order: {key}")
            else:
                # keep the earliest occurrence, skip duplicates
                continue

        # order uniqueness is optional; we warn/skip on strict=True
        if o in seen_orders and strict:
            raise TrackOrderError(f"Duplicate order value in track_order: {o}")

        seen_pairs.add(key)
        seen_orders.add(o)
        d[key] = o
        if o > max_order:
            max_order = o

    next_order = max_order + 1
    return d, next_order

def track_order_add(
    shot_track_to_order: ShotTrackOrderDict, *, shot: int, track_id: int, next_order: int
) -> int:
    """
    Add (shot, track_id) -> next_order if absent; return (possibly updated) next_order.
    """
    key = (int(shot), int(track_id))
    if key not in shot_track_to_order:
        shot_track_to_order[key] = int(next_order)
        return next_order + 1
    return next_order

def track_order_summary(shot_track_to_order: ShotTrackOrderDict) -> str:
    """
    Human-friendly summary for logs.
    """
    if not shot_track_to_order:
        return "entries=0"
    shots = sorted({s for (s, _) in shot_track_to_order})
    return f"entries={len(shot_track_to_order)} shots={shots} next={max(shot_track_to_order.values())+1}"
