from typing import Optional, List, Tuple, Callable, Iterable
import numpy as np
from math import isfinite
import logging
from facekit.tracking.face_structures import FaceTrack, FaceObservation
from facekit.common.obs_consts import Source
from facekit.errors import ResumeSafetyError


# ---------- bbox helpers (unchanged) ----------
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


def _row_to_faceobs(r, tid: int) -> Optional[FaceObservation]:
    bb = _as_int_bbox(r["bbox_xyxy"])
    if bb is None:
        return None
    x1, y1, x2, y2 = bb
    src = str(r.get("src", "detected")).lower()
    return FaceObservation(
        frame_idx=int(r["f"]),
        track_id=int(tid),
        bbox=(x1, y1, x2, y2),
        embedding=None,             # will be filled in attach step
        confidence=(float(r["conf"]) if "conf" in r and r["conf"] is not None else None),
        aligned_face=None,          # not persisted/rehydrated
        source=Source.DETECTED if src == "detected" else Source.TRACKED,
    )


# ---------- Phase 1: observations-only rehydration ----------
def rehydrate_observation_tracks(
    collector,
    *,
    frame_max: Optional[int],
    track_order: dict[tuple[int, int], int],
) -> List[FaceTrack]:
    """
    Create FaceTrack objects from persisted observation rows only (no embeddings yet),
    sorted deterministically using the persisted `track_order`.
    """
    groups = list(collector.iter_tracks(frame_max=frame_max))
    if not groups:
        logging.info("rehydrate_observation_tracks: no groups from collector (frame_max=%r)", frame_max)
        return []

    def _debug_track_order_lookup(track_order: dict[tuple[int, int], int], shot: int, track_id: int) -> None:
        key = (int(shot), int(track_id))
        ko = track_order
        neighbors = []
        for dshot in (-2, -1, 0, +1, +2):
            k = (key[0] + dshot, key[1])
            if k in ko:
                neighbors.append((k, ko[k]))
        same_shot = sorted([(k, v) for (k, v) in ko.items() if k[0] == key[0]], key=lambda kv: kv[1])[:20]
        same_tid  = sorted([(k, v) for (k, v) in ko.items() if k[1] == key[1]], key=lambda kv: kv[0])[:20]
        logging.error(
            "ckpt:track_order missing for key=%s. neighbors(+/-2 shots, same tid)=%s "
            "same_shot(first20 by order)=%s same_track_id(first20 by shot)=%s",
            key, neighbors, same_shot, same_tid
        )

    def _order_key(group):
        shot, track_id, _ = group
        k = (int(shot), int(track_id))
        if k not in track_order:
            # rich diagnostics + fail hard; mixing shot_number vs shot_index is dangerous
            keys = list(track_order.keys())
            shots_present = sorted({s for (s, _) in keys})
            same_shot = sorted([(kk, o) for (kk, o) in track_order.items() if kk[0] == shot], key=lambda x: x[1])[:20]
            same_tid  = sorted([(kk, o) for (kk, o) in track_order.items() if kk[1] == track_id], key=lambda x: (x[0][0], x[1]))[:20]
            logging.error(
                "ckpt:track_order missing for key=%s. neighbors/samples: same_shot=%s same_tid=%s "
                "[shots_present=%s, len(keys)=%d, sample=%s]",
                k, same_shot, same_tid, shots_present, len(keys), keys[:10],
            )
            _debug_track_order_lookup(track_order, shot, track_id)
            raise ResumeSafetyError(
                f"Missing track_order for (shot={shot}, track_id={track_id}). "
                "Likely a shot-number vs shot-index mismatch."
            )
        return track_order[k]

    groups.sort(key=_order_key)

    tracks: List[FaceTrack] = []
    obs_count = 0
    first_span = last_span = None

    for shot, track_id, rows in groups:
        obs_list: List[FaceObservation] = []
        for r in rows:
            o = _row_to_faceobs(r, track_id)
            if o is not None:
                obs_list.append(o)

        if not obs_list:
            continue

        # enforce chronological ordering inside the track
        obs_list.sort(key=lambda o: o.frame_idx)

        t = FaceTrack(shot_id=int(shot), track_id=int(track_id))
        for o in obs_list:
            t.add_observation(o)

        # track spans for debugging
        if first_span is None:
            first_span = (t.shot_id, t.observations[0].frame_idx, t.track_id)
        last_span = (t.shot_id, t.observations[-1].frame_idx, t.track_id)

        obs_count += len(obs_list)
        tracks.append(t)

    logging.info(
        "rehydrate_observation_tracks: built %d tracks, %d observations; span first=%s last=%s",
        len(tracks), obs_count, first_span, last_span
    )
    return tracks


# ---------- Phase 2: embeddings attachment ----------
# Flexible lookups: supply whichever your checkpoint/collector supports.
EmbLookup = Callable[[int, int], Optional[Tuple[Iterable[int], np.ndarray]]]       # -> (frame_indices, embs)
EmbArrayLookup = Callable[[int, int], Optional[np.ndarray]]                        # -> embs only


def attach_embeddings_to_tracks(
    tracks: List[FaceTrack],
    *,
    emb_lookup: EmbLookup | None = None,
    emb_array_lookup: EmbArrayLookup | None = None,
    strict: bool = True,
    log_prefix: str = "rehydrate",
) -> None:
    """
    Attach embeddings to DET observations for each track.

    Matching policy:
      - If emb_lookup returns (frames, embs): we align by chronological order of DET
        observations; if len matches (or both are >0), we attach in order.
      - Else if emb_array_lookup returns embs only: we align by count to DET obs (same ordering).
      - If we cannot attach any embedding to a track that appears to have had DETs, we log,
        and if `strict` we raise ResumeSafetyError.

    Notes:
      - We do not set `aligned_face` (not persisted).
      - We attach embeddings only to DET observations (mirrors write-time behavior).
    """
    tot_det = 0
    tot_attached = 0
    for tr in tracks:
        det_idxs = [i for i, o in enumerate(tr.observations) if o.source == Source.DETECTED]
        if not det_idxs:
            continue

        det_obs = [tr.observations[i] for i in det_idxs]
        tot_det += len(det_obs)

        embs = None

        if emb_lookup is not None:
            res = emb_lookup(int(tr.shot_id), int(tr.track_id))
            if res is not None:
                _, embs = res

        if embs is None and emb_array_lookup is not None:
            embs = emb_array_lookup(int(tr.shot_id), int(tr.track_id))

        if embs is None:
            # couldn’t find any embeddings for this track
            if strict:
                raise ResumeSafetyError(
                    f"{log_prefix}: no embeddings found for (shot={tr.shot_id}, tid={tr.track_id}) "
                    f"with {len(det_obs)} DET observations"
                )
            else:
                logging.info(
                    "%s: missing embeddings for (shot=%d, tid=%d); allowed (strict=False)",
                    log_prefix, int(tr.shot_id), int(tr.track_id),
                )
                continue

        embs = np.asarray(embs)
        if embs.ndim != 2 or embs.shape[1] != 512:
            raise ResumeSafetyError(
                f"{log_prefix}: invalid embedding shape {embs.shape} for (shot={tr.shot_id}, tid={tr.track_id}); "
                "expected (K,512)"
            )

        # Align by chronological order of DET obs. Best-effort policy:
        # - If counts match: 1:1 assignment.
        # - If counts differ but both >0: assign min(counts) in order, log the discrepancy.
        k = min(len(det_obs), embs.shape[0])
        if k == 0:
            if strict:
                raise ResumeSafetyError(
                    f"{log_prefix}: zero alignment between DET obs and embeddings for (shot={tr.shot_id}, tid={tr.track_id})"
                )
            else:
                logging.info(
                    "%s: zero alignment between DET obs and embeddings for (shot=%d, tid=%d); allowed",
                    log_prefix, int(tr.shot_id), int(tr.track_id),
                )
                continue

        if k < len(det_obs) or k < embs.shape[0]:
            logging.warning(
                "%s: partial embedding alignment for (shot=%d, tid=%d): det_obs=%d, emb_rows=%d -> attaching %d",
                log_prefix, int(tr.shot_id), int(tr.track_id), len(det_obs), int(embs.shape[0]), k
            )

        # assign in-order
        for i in range(k):
            det_obs[i].embedding = np.asarray(embs[i], dtype=np.float32, order="C")

        tot_attached += k

    pct = (100.0 * tot_attached / max(1, tot_det))
    logging.info(
        "%s: DET obs=%d, with_embeddings=%d (%.1f%%)", log_prefix, tot_det, tot_attached, pct
    )


# ---------- Phase 3: one-call rehydration (observations + embeddings) ----------
def rehydrate_tracks(
    collector,
    *,
    frame_max: Optional[int],
    track_order: dict[tuple[int, int], int],
    # Flexible providers; pass lambdas that call your checkpoint/collector/emb_store:
    emb_lookup: EmbLookup | None = None,
    emb_array_lookup: EmbArrayLookup | None = None,
) -> List[FaceTrack]:
    """
    Full rehydration used by the pipeline at resume-time:
      1) Rebuild tracks from observation rows (strictly before anchor).
      2) Attach embeddings to DET observations from checkpoint/collector.
    """
    tracks = rehydrate_observation_tracks(collector, frame_max=frame_max, track_order=track_order)
    attach_embeddings_to_tracks(
        tracks,
        emb_lookup=emb_lookup,
        emb_array_lookup=emb_array_lookup,
        strict=False,
        log_prefix="rehydrate",
    )
    return tracks
