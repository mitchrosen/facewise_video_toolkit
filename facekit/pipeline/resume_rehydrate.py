from typing import Optional, List, Tuple
import numpy as np
from math import isfinite
import logging 
from facekit.tracking.face_structures import FaceTrack, FaceObservation
from facekit.common.obs_consts import Source
from facekit.errors import ResumeSafetyError

def _as_int_bbox(bb) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = bb
    vals = (float(x1), float(y1), float(x2), float(y2))
    if not all(isfinite(v) for v in vals):
        return None
    x1, y1, x2, y2 = vals
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if (x2 - x1) <= 0 or (y2 - y1) <= 0:
        return None
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

def _sanitize_bbox(x1, y1, x2, y2) -> Optional[Tuple[int,int,int,int]]:
    if not all(np.isfinite([x1, y1, x2, y2])):
        return None
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    if x2 == x1 or y2 == y1:
        return None
    return x1, y1, x2, y2

def _row_to_faceobs(r, tid: int) -> Optional[FaceObservation]:
    bb = _as_int_bbox(r["bbox_xyxy"])
    if bb is None:
        return None
    x1, y1, x2, y2 = bb
    return FaceObservation(
        frame_idx=int(r["f"]),
        track_id=int(tid),  # <- from group key
        bbox=(x1, y1, x2, y2),
        embedding=None,
        confidence=(float(r["conf"]) if "conf" in r and r["conf"] is not None else None),
        aligned_face=None,
        source=Source.DETECTED if str(r.get("src", "detected")).lower() == "detected" else Source.TRACKED,
    )

def rehydrate_tracks_from_observations(
    collector,
    *,
    frame_max: Optional[int],
    track_order: dict[tuple[int, int], int],
) -> List[FaceTrack]:
    tracks: List[FaceTrack] = []

    groups = list(collector.iter_tracks(frame_max=frame_max))

    def _debug_track_order_lookup(track_order: dict[tuple[int, int], int], shot: int, track_id: int) -> None:
        key = (int(shot), int(track_id))
        ko = track_order

        # Nearby candidates to catch off-by-one/indexing errors:
        neighbors = []
        for dshot in (-2, -1, 0, +1, +2):
            k = (key[0] + dshot, key[1])
            if k in ko:
                neighbors.append((k, ko[k]))

        # Other track_ids in the same shot:
        same_shot = sorted([(k, v) for (k, v) in ko.items() if k[0] == key[0]], key=lambda kv: kv[1])[:20]
        # Same track_id across all shots:
        same_tid = sorted([(k, v) for (k, v) in ko.items() if k[1] == key[1]], key=lambda kv: kv[0])[:20]

        logging.error(
            "ckpt:track_order missing for key=%s. "
            "neighbors(+/-2 shots, same track_id)=%s "
            "same_shot(first20 by order)=%s "
            "same_track_id(first20 by shot)=%s",
            key, neighbors, same_shot, same_tid
        )

    def key_of(group):
        shot, track_id, _ = group
        k = (int(shot), int(track_id))
        if k not in track_order:

            keys = list(track_order.keys())
            shots_present = sorted({s for (s, _) in keys})
            same_shot = sorted([(k, o) for (k, o) in track_order.items() if k[0] == shot], key=lambda x: x[1])[:20]
            same_tid  = sorted([(k, o) for (k, o) in track_order.items() if k[1] == track_id], key=lambda x: (x[0][0], x[1]) )[:20]

            logging.error(
                "ckpt:track_order missing for key=%s. "
                "neighbors(+/-2 shots, same track_id)=%s "
                "same_shot(first20 by order)=%s "
                "same_track_id(first20 by shot)=%s "
                "[shots_present=%s, len(keys)=%d, sample=%s]",
                k,
                [m for (m, _) in same_tid if abs(k[0] - shot) <= 2],
                same_shot,
                same_tid,
                shots_present,
                len(keys),
                keys[:10],
            )

            # helpful hints for index/number confusion:
            cand1 = (shot + 1, track_id)
            cand2 = (shot - 1, track_id)
            if cand1 in track_order:
                logging.error("hint: (shot+1,tid) exists -> %s has order %s", cand1, track_order[cand1])
            if cand2 in track_order:
                logging.error("hint: (shot-1,tid) exists -> %s has order %s", cand2, track_order[cand2])

            _debug_track_order_lookup(track_order, shot, track_id)
            raise ResumeSafetyError(
                f"Missing track_order for (shot={shot}, track_id={track_id}). "
                "Likely a shot-number vs shot-index mismatch."
            )
        return track_order[k]

    groups.sort(key=key_of)

    for shot, track_id, rows in groups:
        obs_list = []
        for r in rows:
            o = _row_to_faceobs(r, track_id)
            if o is not None:
                obs_list.append(o)
        if not obs_list:
            continue
        t = FaceTrack(shot_id=int(shot), track_id=int(track_id))
        for o in obs_list:
            t.add_observation(o)
        tracks.append(t)
    
    return tracks

def build_prior_tracks_from_collector(obs_collector, *, frame_max: int | None):
    prior = []
    for shot, tid, obs_list in (obs_collector.iter_tracks(frame_max=frame_max) or []):
        tr = SimpleTrack(shot_id=int(shot), track_id=int(tid))
        for d in obs_list:
            x1, y1, x2, y2 = d["bbox_xyxy"]
            tr.observations.append(SimpleObs(
                frame_idx=int(d["f"]),
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                source=str(d.get("src", "detected")),
                confidence=(float(d["conf"]) if "conf" in d else None),
            ))
        if tr.observations:
            prior.append(tr)
    return prior