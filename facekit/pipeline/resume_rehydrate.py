from typing import Optional, List, Tuple, Callable, Iterable, Dict
import numpy as np
from math import isfinite
from pathlib import Path
from dataclasses import dataclass
import logging

from facekit.tracking.face_structures import FaceTrack, FaceObservation
from facekit.common.obs_consts import Source, code_to_src, SRC_TO_CODE
from facekit.errors import ResumeSafetyError
from facekit.pipeline.checkpoint import TrackingCheckpoint

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

def _normalize_src(raw_src) -> Source:
    """
    Normalize a persisted 'src' field into a Source enum.

    Accepts:
      - Source enum          -> returned as-is
      - int / np.integer     -> treated as numeric code via code_to_src(...)
      - numeric str ("0")    -> same as above
      - name str ("detected"/"tracked"/"flow") -> mapped via Source(...)
      - numpy scalar variants -> unboxed then re-normalized

    Raises ResumeSafetyError on anything we cannot interpret safely.
    """
    # Already a Source enum
    if isinstance(raw_src, Source):
        return raw_src

    # Numpy scalar or plain int
    if isinstance(raw_src, (int, np.integer)):
        return code_to_src(int(raw_src))

    # Plain string (could be code or name)
    if isinstance(raw_src, str):
        s = raw_src.strip()
        # Try as numeric code first
        try:
            return code_to_src(int(s))
        except ValueError:
            # Then as enum value ("detected", "tracked", "flow")
            try:
                return Source(s.lower())
            except Exception as e:
                raise ResumeSafetyError(f"rehydrate: unknown string src={raw_src!r}") from e

    # Numpy scalar string or other scalar-like
    if hasattr(raw_src, "item"):
        try:
            return _normalize_src(raw_src.item())
        except Exception as e:
            raise ResumeSafetyError(f"rehydrate: unsupported src scalar {raw_src!r}") from e

    raise ResumeSafetyError(
        f"rehydrate: unsupported src type {type(raw_src)!r} value={raw_src!r}"
    )

def _landmarks_from_flat10(has_landmarks, flat10) -> np.ndarray | None:
    """
    Convert persisted (has_landmarks, flat10) into a (5,2) float32 array.

    Contract:
      - If has_landmarks == 0 → return None
      - If has_landmarks == 1 → flat10 MUST be length 10, all finite
      - Anything else → ResumeSafetyError
    """
    if not has_landmarks:
        return None

    if flat10 is None:
        raise ResumeSafetyError("rehydrate: has_landmarks=1 but landmarks_flat10 is None")

    try:
        arr = np.asarray(flat10, dtype=np.float32)
    except Exception as e:
        raise ResumeSafetyError(f"rehydrate: invalid landmarks_flat10: {e}")

    if arr.shape != (10,):
        raise ResumeSafetyError(
            f"rehydrate: landmarks_flat10 must have shape (10,), got {arr.shape}"
        )

    if not np.all(np.isfinite(arr)):
        raise ResumeSafetyError(
            "rehydrate: landmarks_flat10 contains NaN/Inf but has_landmarks=1"
        )

    return arr.reshape((5, 2))

def _row_to_faceobs(row: dict, *, shot_id: int, track_id: int) -> FaceObservation | None:
    """
    Convert a persisted observation row into a FaceObservation.

    IMPORTANT:
      - Some collectors omit 'shot' inside each row because shot is implied by grouping.
        We therefore accept shot_id explicitly and only *read* row['shot'] if present.
    """
    bb = _as_int_bbox(row["bbox_xyxy"])
    if bb is None:
        return None
    x1, y1, x2, y2 = bb

    raw_src = row.get("src")
    try:
        src = _normalize_src(raw_src)
    except Exception as e:
        raise ResumeSafetyError(
            f"rehydrate: invalid src={raw_src!r} in obs row "
            f"(shot={shot_id}, f={row.get('f')}, tid={track_id})"
        ) from e

    # landmarks payload (strict/soft policy is handled in B; for A keep strict call)
    has_landmarks = int(row.get("has_landmarks", 0))
    landmarks_flat10 = row.get("landmarks_flat10")
    landmarks = _landmarks_from_flat10(has_landmarks, landmarks_flat10)

    # confidence (optional)
    conf = None
    if "conf" in row and row["conf"] is not None:
        try:
            v = float(row["conf"])
            conf = None if np.isnan(v) else v
        except Exception:
            conf = None

    obs = FaceObservation(
        frame_idx=int(row["f"]),
        track_id=int(track_id) if track_id is not None and int(track_id) >= 0 else None,
        bbox=(x1, y1, x2, y2),
        embedding=None,
        confidence=conf,
        aligned_face=None,
        landmarks=landmarks,
        source=src,
    )

    # Debug logging (optional, safe)
    try:
        logging.info(
            "rehydrate DEBUG obs row: shot=%r tid=%r f=%r src=%r has_landmarks=%r flat10_len=%r",
            int(shot_id),
            track_id,
            row.get("f"),
            raw_src,
            has_landmarks,
            (len(landmarks_flat10) if landmarks_flat10 is not None else None),
        )
    except Exception:
        pass

    return obs

#
# ---------- Embedding sidecar helpers ----------
#
def make_emb_lookup_from_sidecars(
    *,
    obs_collector,
    emb_array: np.ndarray,
) -> "EmbLookup":
    """
    Build an EmbLookup that uses:
      - obs_collector.to_array() with fields: shot, track_id, f, src, emb_idx
      - emb_array: flat (N, 512) embeddings matrix, where emb_idx is the row index.

    For each (shot, track_id), it:
      - finds DET rows with emb_idx>=0
      - sorts them by frame, and
      - returns (frames, embs) where embs is stacked in that same order.

    If *any* DET row for that track lacks an embedding (emb_idx < 0), strict callers
    will treat that as a mismatch when attach_embeddings_to_tracks runs.
    """
    if emb_array is None or getattr(emb_array, "ndim", 0) != 2:
        raise ResumeSafetyError(
            f"rehydrate: emb_array has invalid shape {getattr(emb_array, 'shape', None)}; expected (N, 512)"
        )

    try:
        arr = obs_collector.to_array()
    except Exception as e:
        raise ResumeSafetyError(f"rehydrate: obs_collector.to_array() failed: {e}") from e

    if getattr(arr, "size", 0) == 0:
        # no observations at all → no embeddings to attach.
        def _empty_lookup(_shot: int, _tid: int):
            return None
        return _empty_lookup

    names = arr.dtype.names or ()
    required = {"shot", "track_id", "f", "src", "emb_idx"}
    missing = required - set(names)
    if missing:
        raise ResumeSafetyError(
            f"rehydrate: obs sidecar missing required fields {sorted(missing)}; "
            f"present={list(names)}"
        )

    det_src = Source.DETECTED

    # Pre-index rows by (shot, track_id)
    by_key: Dict[tuple[int, int], list[tuple[int, int]]] = {}
    for idx, row in enumerate(arr):
        try:
            row_src = _normalize_src(row["src"])
        except ResumeSafetyError as e:
            raise ResumeSafetyError(
                f"rehydrate: invalid src={row['src']!r} in obs sidecar row idx={idx}"
            ) from e

        if row_src != det_src:
            continue

        emb_idx = int(row["emb_idx"])
        if emb_idx < 0:
            # DET row without embedding; we'll let the strictness check in
            # attach_embeddings_to_tracks decide whether this is acceptable.
            continue

        key = (int(row["shot"]), int(row["track_id"]))
        frame = int(row["f"])
        by_key.setdefault(key, []).append((frame, emb_idx))

    def _lookup(shot: int, tid: int) -> Optional[Tuple[Iterable[int], np.ndarray]]:
        key = (int(shot), int(tid))
        rows = by_key.get(key)
        if not rows:
            return None
        # sort by frame to stay in chronological order
        rows_sorted = sorted(rows, key=lambda t: t[0])
        frames: list[int] = []
        vecs: list[np.ndarray] = []
        n_rows = emb_array.shape[0]
        for f, ei in rows_sorted:
            if ei < 0 or ei >= n_rows:
                raise ResumeSafetyError(
                    f"rehydrate: emb_idx {ei} out of bounds for embeddings array of size {n_rows}"
                )
            frames.append(int(f))
            vecs.append(np.asarray(emb_array[ei], dtype=np.float32, order="C"))

        if not vecs:
            return None

        return frames, np.stack(vecs, axis=0)

    return _lookup

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
            # Some collectors omit 'shot' in row dicts. Ensure shot is always known.
            # Also avoid mutating collector-owned objects by copying to a plain dict.
            row = dict(r)
            row.setdefault("shot", int(shot))  # ensures debug/log consistency and downstream safety

            o = _row_to_faceobs(row, shot_id=int(shot), track_id=int(track_id))
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
            try:
                # extra context for debugging lookup mismatch
                det_frames = [int(o.frame_idx) for o in det_obs]
                logging.error("%s: missing embs for (shot=%d, tid=%d) det_frames=%s",
                          log_prefix, int(tr.shot_id), int(tr.track_id), det_frames[:10])
            except Exception:
                pass

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

        # Align by chronological order of DET obs. For resume pre-anchor we expect
        # a 1:1 match (strict). Fail fast if counts differ.
        k_det  = len(det_obs)
        k_embs = int(embs.shape[0])
        if strict and k_embs != k_det:
            raise ResumeSafetyError(
                f"{log_prefix}: embedding count mismatch for (shot={tr.shot_id}, tid={tr.track_id}): "
                f"DET={k_det} vs EMB={k_embs}"
            )
        k = min(k_det, k_embs)
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

        # assign in-order AND publish to the track-level list
        if getattr(tr, "embeddings", None) is None:
            tr.embeddings = []

        for i in range(k):
            vec = np.asarray(embs[i], dtype=np.float32, order="C")
            det_obs[i].embedding = vec
            tr.embeddings.append(vec)

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
    anchor_shot_id: int,
    strict: bool = True,
) -> List[FaceTrack]:
    """
    Full rehydration used by the pipeline at resume-time:
      1) Rebuild tracks from observation rows (strictly before anchor).
      2) Attach embeddings to DET observations from checkpoint/collector.
    """
    completed_tracks: list[FaceTrack] = []
    anchor_tracks: list[FaceTrack] = []
    future_tracks: list[FaceTrack] = []

    tracks = rehydrate_observation_tracks(
        collector,
        frame_max=frame_max,
        track_order=track_order,
    )
    
    for t in tracks:
        s = int(getattr(t, "shot_id", -1))
        if s == anchor_shot_id:
            anchor_tracks.append(t)
        elif s < anchor_shot_id:
            completed_tracks.append(t)
        else:
            future_tracks.append(t)

    if future_tracks:
        # This should never happen if shots are well-formed and frame_max is pre-anchor.
        raise ResumeSafetyError(
            f"rehydrate: found pre-anchor tracks from future shot(s) > anchor_shot_id={anchor_shot_id}"
        )

    # 1) completed shots: obey strict flag
    if completed_tracks:
        attach_embeddings_to_tracks(
            completed_tracks,
            emb_lookup=emb_lookup,
            emb_array_lookup=emb_array_lookup,
            strict=bool(strict),
            log_prefix="rehydrate-completed",
        )

    # 2) anchor shot: always non-strict on embeddings (by design will be mid-shot with no embs yet)
    if anchor_tracks:
        attach_embeddings_to_tracks(
            anchor_tracks,
            emb_lookup=emb_lookup,
            emb_array_lookup=emb_array_lookup,
            strict=False,
            log_prefix="rehydrate-anchor",
        )

    # 3) Stitch back together in the original order
    return completed_tracks + anchor_tracks

@dataclass
class ResumePlan:
    """
    Immutable snapshot of all resume-related state needed by track_across_segments.

    Fields
    ------
    anchor_frame :
        Absolute frame index at which new work must begin. All frames strictly
        before this are considered "pre-anchor" and must already be persisted.
    is_resume :
        True if this run is resuming from a previous checkpoint (anchor_frame > 0),
        False for a cold start.
    first_processed_shot_number :
        The shot_number of the first shot that will be processed in this run
        (i.e. the shot containing anchor_frame, after trimming the shot list).
    segment_id_seed_by_shot :
        Per-shot seed for segment_id assignment. For shots that were already
        completed pre-anchor, this is the count of their pre-existing segments;
        for the anchor shot it is the number of pre-anchor segments.
    trackid_seed_by_shot :
        Per-shot seed for track_id allocation. For each shot, this is
        max(pre-anchor track_id) + 1, so new tracks continue numbering cleanly.
    prior_tracks_anchor :
        Tracks rehydrated for the anchor-containing shot, limited to frames
        strictly before anchor_frame. These are used to seed the first shot's
        aggregator so labels and embeddings remain consistent across runs.
    reuse_tid_for_first_shot :
        If not None, the specific track_id that should be reused for the first
        *new* detection track in the anchor shot, to preserve exact parity of
        labels at the resume boundary.
    """
    anchor_frame: int
    is_resume: bool
    first_processed_shot_number: int
    segment_id_seed_by_shot: Dict[int, int]
    trackid_seed_by_shot: Dict[int, int]
    prior_tracks_anchor: List[FaceTrack]
    reuse_tid_for_first_shot: Optional[int]

def _resolve_anchor(
    checkpoint: TrackingCheckpoint | None,
    resume_enabled: bool,
) -> int:
    """
    Decide the global resume anchor frame for this run.

    Precedence (highest to lowest):
      1. checkpoint.get_resume_anchor()[0]               # explicit tuple
      2. checkpoint.read_status()['last_detection_frame']
      3. status.json on disk via checkpoint.status_path
      4. checkpoint.last_detection_frame                 # legacy attr/callable
      5. max frame index from checkpoint.obs_collector
      6. Fallback to 0 if resume is disabled or no info exists.

    Parameters
    ----------
    checkpoint :
        Optional TrackingCheckpoint providing status and observation sidecars.
    resume_enabled :
        If False, all resume sources are ignored and 0 is returned.

    Returns
    -------
    int
        The absolute frame index where new work must begin. 0 means cold start.
    """
    if not (resume_enabled and checkpoint):
        logging.info("resume: disabled or no checkpoint -> anchor=0")
        return 0

    # (1) explicit tuple from get_resume_anchor()
    try:
        anchors = checkpoint.get_resume_anchor()
        if anchors is not None and len(anchors) >= 1:
            logging.info("resume: using get_resume_anchor() -> %r", anchors)
            return int(anchors[0])
    except Exception:
        pass

    # (2) prefer read_status() → status.json
    try:
        rs = getattr(checkpoint, "read_status", None)
        if callable(rs):
            status = rs() or {}
            val = status.get("last_detection_frame")
            if val is not None:
                logging.info("resume: using read_status()['last_detection_frame'] -> %r", val)
                return int(val)
    except Exception:
        pass

    # (3) direct status.json path if exposed
    try:
        status_path = getattr(checkpoint, "status_path", None)
        if status_path:
            import os as _os, json as _json
            if _os.path.exists(status_path):
                with open(status_path, "r") as f:
                    status = _json.load(f) or {}
                val = status.get("last_detection_frame")
                if val is not None:
                    logging.info("resume: using status.json file -> %r", val)
                    return int(val)
    except Exception:
        pass

    # (4) legacy callable/attr last_detection_frame
    try:
        candidate = getattr(checkpoint, "last_detection_frame", None)
        val = candidate() if callable(candidate) else candidate
        if val is not None:
            logging.info("resume: using legacy last_detection_frame -> %r", val)
            return int(val)
    except Exception:
        pass

    # (5) max observed frame from the obs collector
    try:
        collector = getattr(checkpoint, "obs_collector", None)
        if collector is not None and hasattr(collector, "get_all_frame_indices"):
            frames_seen = collector.get_all_frame_indices()
            if frames_seen is not None and len(frames_seen) > 0:
                max_frame = int(np.max(frames_seen))
                logging.info("resume: using obs_collector max frame -> %d", max_frame)
                return max_frame
    except Exception:
        pass

    logging.info("resume: no anchor inputs -> anchor=0")
    return 0

def _shot_idx_by_abs_frame(shots, abs_frame_idx: int) -> int:
    """
    Return the **shot index** i such that shots[i]["last_frame"] >= abs_frame_idx,
    choosing the leftmost such i. If no such shot exists (abs_frame_idx is after
    all shots), return len(shots).

    Notes:
    - Shots must be sorted, non-overlapping, ascending by frame range.
    - If abs_frame_idx < shots[0]["first_frame"], this returns 0. The caller
      could clamp the starting frame i.e:
          start_at = max(shots[i]["first_frame"], abs_frame_idx)
    """
    # First shot whose last_frame >= resume_abs_frame
    from bisect import bisect_left
    last_frames = [s["last_frame"] for s in shots]
    return bisect_left(last_frames, abs_frame_idx)

def _assign_segment_ids_for_rehydrated(
    prior_tracks: List[FaceTrack],
    track_order: dict[tuple[int, int], int],
) -> Dict[int, int]:
    """
    Assign deterministic per-shot segment_id values to rehydrated tracks.

    Primary rule:
      - If (shot_id, track_id) exists in `track_order`, use that order index as segment_id.
        This makes segment labeling stable across resumes, independent of tracking dynamics.

    Fallback rule (only if track_order is missing for a particular track):
      - Use track_id as a deterministic stand-in segment_id.

    Returns:
      segment_id_seed_by_shot: {shot_id: next_segment_id_seed}
        where next_seed is max_assigned_segment_id + 1 for that shot (or 0 if none).
    """
    seed_by_shot: Dict[int, int] = {}
    max_seg_by_shot: Dict[int, int] = {}

    for tr in prior_tracks or []:
        shot = int(getattr(tr, "shot_id", -1))
        tid  = int(getattr(tr, "track_id", -1))
        if shot < 0 or tid < 0:
            continue

        key = (shot, tid)
        if key in track_order:
            seg = int(track_order[key])
        else:
            # Deterministic fallback (keeps resume working even if track_order is partial)
            seg = tid
            logging.warning(
                "resume: track_order missing for (shot=%d, tid=%d); "
                "using fallback segment_id=%d",
                shot, tid, seg
            )

        # Mutate the track in place (expected by downstream segment labeling)
        setattr(tr, "segment_id", seg)

        prev = max_seg_by_shot.get(shot, -1)
        if seg > prev:
            max_seg_by_shot[shot] = seg

    for shot, max_seg in max_seg_by_shot.items():
        seed_by_shot[int(shot)] = int(max_seg) + 1 if int(max_seg) >= 0 else 0

    return seed_by_shot

def _build_resume_plan(
    shots: list,
    *,
    checkpoint: TrackingCheckpoint | None,
    resume_enabled: bool,
    all_tracks: List[FaceTrack],
) -> tuple[ResumePlan, list]:
    """
    Construct a ResumePlan and trim the shot list to the anchor-containing shot.

    This function encapsulates all the heavy lifting that used to live inline
    in track_across_segments:
      * Resolve the anchor frame.
      * Audit pre-anchor DET rows for embedding/landmarks parity.
      * Rehydrate pre-anchor tracks from sidecars.
      * Split pre-anchor tracks into completed vs anchor-containing shots.
      * Push completed-shot tracks into all_tracks immediately.
      * Assign segment_id seeds and track_id seeds per shot.
      * Decide which tid to reuse at the resume boundary.
      * Trim `shots` so the first processed shot always contains the anchor.

    Parameters
    ----------
    shots :
        Original list of shot dicts from shot JSON (untrimmed).
    checkpoint :
        Optional TrackingCheckpoint for accessing sidecars and status.
    resume_enabled :
        Whether resume is allowed. If False, this behaves like a cold start.
    all_tracks :
        The list that will ultimately hold all tracks; pre-anchor completed
        tracks are appended here immediately.

    Returns
    -------
    plan, trimmed_shots :
        plan : ResumePlan
            Immutable resume state struct used by the main tracking loop.
        trimmed_shots : list
            The subset of shots starting from the anchor-containing shot.
    """
    segment_id_seed_by_shot: Dict[int, int] = {}
    trackid_seed_by_shot: Dict[int, int] = {}
    prior_tracks: List[FaceTrack] = []

    resume_abs_frame: int = _resolve_anchor(checkpoint, resume_enabled)

    # === AUDIT FENCE #0: record collector rows BEFORE any new work ===
    try:
        if checkpoint and hasattr(checkpoint, "obs_collector"):
            rows_before = int(checkpoint.obs_collector.count())
            logging.info("resume: obs_collector rows BEFORE processing=%d", rows_before)
    except Exception:
        logging.exception("resume: failed to read obs_collector.count() before")

    # Identify anchor-containing shot BEFORE any trimming (used for logging)
    anchor_shot_idx = _shot_idx_by_abs_frame(shots, resume_abs_frame) if resume_abs_frame > 0 else None
    anchor_shot_num = (
        shots[anchor_shot_idx]["shot_number"]
        if (anchor_shot_idx is not None and anchor_shot_idx < len(shots))
        else None
    )

    # --- AUDIT: verify pre-anchor DET rows have embeddings in sidecar ---
    _audit_preanchor_embedding_parity(
        checkpoint,
        shots=shots,
        anchor_frame=resume_abs_frame,
        anchor_shot=anchor_shot_num,
    )

    # Compute completed shots (fully before anchor) BEFORE trimming
    completed_shot_nums = (
        {s["shot_number"] for s in shots}
        if resume_abs_frame == 0
        else {s["shot_number"] for s in shots if s["last_frame"] < resume_abs_frame}
    )

    # Rehydrate BEFORE the anchor (if any)
    obs_for_rehydrate = getattr(checkpoint, "obs_collector", None)
    if obs_for_rehydrate is not None and resume_abs_frame > 0:
        emb_lookup, emb_array_lookup = _build_emb_lookups_for_checkpoint(
            checkpoint,
            anchor_frame=resume_abs_frame,
        )
        prior_tracks = rehydrate_tracks(
            obs_for_rehydrate,
            frame_max=resume_abs_frame - 1,
            track_order=(checkpoint.get_track_order() or {}) if checkpoint else {},
            emb_lookup=emb_lookup,
            emb_array_lookup=emb_array_lookup,
            anchor_shot_id=anchor_shot_num,
            strict=True,
        )
    elif obs_for_rehydrate is not None:
        logging.info("resume: anchor=0 -> cold start; skipping pre-anchor rehydration")
        prior_tracks = []
    else:
        logging.info("resume: no obs_collector on checkpoint; skipping pre-anchor rehydration")
        prior_tracks = []

    # Hard guard: enforce DET↔EMB parity and landmarks presence on pre-anchor DETs
    for t in prior_tracks or []:
        shot_id = int(getattr(t, "shot_id", -1))
        is_completed_shot = shot_id in completed_shot_nums
        is_anchor_shot = (anchor_shot_num is not None and shot_id == int(anchor_shot_num))

        if not (is_completed_shot or is_anchor_shot):
            continue

        det_cnt = sum(
            1
            for o in (getattr(t, "observations", []) or [])
            if getattr(o, "source", None) == Source.DETECTED
            and int(o.frame_idx) <= int(resume_abs_frame - 1)
        )
        emb_cnt = len(getattr(t, "embeddings", []) or [])

        if is_completed_shot and det_cnt > 0 and emb_cnt != det_cnt:
            raise ResumeSafetyError(
                f"rehydrate: pre-anchor embedding parity failed for (shot={shot_id}, "
                f"tid={int(getattr(t,'track_id',-1))}): DET={det_cnt} vs EMB={emb_cnt}"
            )

        for o in getattr(t, "observations", []) or []:
            if (
                getattr(o, "source", None) == Source.DETECTED
                and int(o.frame_idx) <= int(resume_abs_frame - 1)
            ):
                lm = getattr(o, "landmarks", None)
                if lm is None:
                    raise ResumeSafetyError(
                        "rehydrate: missing landmarks for pre-anchor DET "
                        f"(shot={shot_id}, tid={int(getattr(t,'track_id',-1))}, frame={int(o.frame_idx)})"
                    )
                arr = np.asarray(lm, dtype=np.float32)
                if arr.shape != (5, 2):
                    raise ResumeSafetyError(
                        f"rehydrate: landmarks must be (5,2) on pre-anchor DET "
                        f"(shot={shot_id}, tid={int(getattr(t,'track_id',-1))}, frame={int(o.frame_idx)}), got {arr.shape}"
                    )
                if not np.all(np.isfinite(arr)):
                    raise ResumeSafetyError(
                        f"rehydrate: landmarks contain NaN/Inf on pre-anchor DET "
                        f"(shot={shot_id}, tid={int(getattr(t,'track_id',-1))}, frame={int(o.frame_idx)})"
                    )

    # Split rehydrated tracks into fully completed vs anchor shot
    prior_tracks_completed: List[FaceTrack] = []
    prior_tracks_anchor: List[FaceTrack] = []
    for t in prior_tracks or []:
        s = int(getattr(t, "shot_id", -1))
        if s in completed_shot_nums:
            prior_tracks_completed.append(t)
        elif s == (anchor_shot_num if anchor_shot_num is not None else -1):
            prior_tracks_anchor.append(t)
        else:
            logging.debug("rehydrate: ignoring pre-anchor track from unexpected shot=%s", s)

    # Completed shots become outputs immediately
    if prior_tracks_completed:
        logging.info(
            "resume: adding %d completed pre-anchor tracks to outputs",
            len(prior_tracks_completed),
        )
        all_tracks.extend(prior_tracks_completed)

    # Logging for rehydrated counts
    try:
        by_shot_counts = {}
        for track in prior_tracks:
            shot = int(getattr(track, "shot_id", -1))
            by_shot_counts[shot] = by_shot_counts.get(shot, 0) + 1
        logging.info("resume: rehydrated counts by shot: %r", by_shot_counts)
        logging.info(
            "resume: segment_id_seed_by_shot: %r",
            {int(k): int(v) for k, v in (segment_id_seed_by_shot or {}).items()},
        )
        logging.info(
            "resume: trackid_seed_by_shot: %r",
            {int(k): int(v) for k, v in (trackid_seed_by_shot or {}).items()},
        )
    except Exception:
        logging.exception("resume: failed summarizing seeds")

    # Assign stable segment_ids to rehydrated tracks and compute per-shot seeds
    last_tid_by_shot: Dict[int, int] = {}
    try:
        track_order_map = (
            checkpoint.get_track_order()
            if (checkpoint and hasattr(checkpoint, "get_track_order"))
            else {}
        )
        segment_id_seed_by_shot = _assign_segment_ids_for_rehydrated(
            prior_tracks, track_order_map or {}
        )

        tmp: Dict[int, int] = {}
        last_frame_by_shot_tid: Dict[Tuple[int, int], int] = {}

        for track in prior_tracks:
            shot = int(getattr(track, "shot_id", 0))
            tid = int(getattr(track, "track_id", -1))
            if tid >= 0:
                tmp[shot] = max(tmp.get(shot, -1), tid)
                if getattr(track, "observations", None):
                    last_frame = max(o.frame_idx for o in track.observations)
                    key = (shot, tid)
                    if key not in last_frame_by_shot_tid or last_frame > last_frame_by_shot_tid[key]:
                        last_frame_by_shot_tid[key] = last_frame

        trackid_seed_by_shot = {shot: (max_tid + 1) for shot, max_tid in tmp.items()}
        for (shot, tid), last_frame in last_frame_by_shot_tid.items():
            if (shot not in last_tid_by_shot) or last_frame > last_frame_by_shot_tid.get(
                (shot, last_tid_by_shot[shot]), -1
            ):
                last_tid_by_shot[shot] = tid

        logging.info(
            "resume: assigned segment_ids to %d rehydrated tracks; seeds=%s",
            len(prior_tracks),
            {k: int(v) for k, v in segment_id_seed_by_shot.items()},
        )
    except Exception:
        logging.exception(
            "resume: failed to assign segment_ids to rehydrated tracks; continuing with empty seeds"
        )
        segment_id_seed_by_shot = {}
        trackid_seed_by_shot = {}
        last_tid_by_shot = {}

    if checkpoint and hasattr(checkpoint, "_validate_resume_embeddings"):
        checkpoint._validate_resume_embeddings(anchor_shot=anchor_shot_num)

    logging.info(
        "resume: prior_tracks strictness OK (anchor_shot=%s); rehydrated=%d (completed=%d, anchor-shot=%d)",
        anchor_shot_num,
        len(prior_tracks or []),
        len(prior_tracks_completed),
        len(prior_tracks_anchor),
    )

    # Trim shots so the first processed shot is the one containing the anchor
    start_shot_idx = _shot_idx_by_abs_frame(shots, resume_abs_frame)
    if start_shot_idx >= len(shots):
        raise ResumeSafetyError("resume anchor beyond last shot; aborting for safety.")
    shots_trimmed = shots[start_shot_idx:]

    if not shots_trimmed:
        first_processed_shot_number = 0
    else:
        first_processed_shot_number = shots_trimmed[0]["shot_number"]

    is_resume = bool(resume_abs_frame > 0)

    reuse_tid_for_first_shot: Optional[int] = None
    if checkpoint and resume_abs_frame > 0:
        try:
            reuse_tid_for_first_shot = last_tid_by_shot.get(int(first_processed_shot_number))
            if reuse_tid_for_first_shot is not None:
                logging.info(
                    "resume: will reuse tid=%d for first detection in shot=%d",
                    int(reuse_tid_for_first_shot),
                    int(first_processed_shot_number),
                )
        except Exception:
            reuse_tid_for_first_shot = None

        logging.info(
            "resume: anchor=%d anchor_shot_num=%s first_processed_shot_number=%s reuse_tid_for_first_shot=%s",
            resume_abs_frame,
            anchor_shot_num,
            first_processed_shot_number,
            reuse_tid_for_first_shot,
        )

    plan = ResumePlan(
        anchor_frame=int(resume_abs_frame),
        is_resume=is_resume,
        first_processed_shot_number=int(first_processed_shot_number),
        segment_id_seed_by_shot=segment_id_seed_by_shot,
        trackid_seed_by_shot=trackid_seed_by_shot,
        prior_tracks_anchor=prior_tracks_anchor,
        reuse_tid_for_first_shot=reuse_tid_for_first_shot,
    )
    return plan, shots_trimmed

def _audit_preanchor_embedding_parity(checkpoint, *, shots: list, anchor_frame: int, anchor_shot: Optional[int]):
    """
    Resume-only audit: verify that every DET row with frame <= anchor-1 has an embedding.
    Uses rows_for_frame(shot, frame) if available. Falls back to iter_track_frames(...) to
    at least count DETs and flag likely gaps. Logs which frames are missing and whether a
    landmarks are present (when available).
    """
    if not checkpoint or not hasattr(checkpoint, "obs_collector"):
        logging.info("RESUME-AUDIT: no checkpoint/collector; skipping.")
        return

    oc = checkpoint.obs_collector
    has_rows_for_frame = hasattr(oc, "rows_for_frame") and callable(getattr(oc, "rows_for_frame"))
    has_iter_track_frames = hasattr(oc, "iter_track_frames") and callable(getattr(oc, "iter_track_frames"))

    if not has_rows_for_frame and not has_iter_track_frames:
        logging.info("RESUME-AUDIT: collector lacks rows_for_frame/iter_track_frames; skipping.")
        return

    # Helper: record a missing embedding at (shot, tid, frame) with landmarks presence if we can see it.
    def _record_missing(
        shot: int,
        tid: int,
        frame: int,
        *,
        has_landmarks: Optional[int],
        det_by_key: Dict[tuple[int, int], list],
    ):
        det_by_key.setdefault((shot, tid), []).append(
            {"f": int(frame), "has_landmarks": int(bool(has_landmarks))}
        )

    det_by_key: Dict[tuple[int,int], list] = {}
    total_missing = 0

    # Iterate only up to anchor-1 and (optionally) only up to anchor_shot.
    for s in shots:
        shot_num = int(s["shot_number"])
        if anchor_shot is not None and shot_num > int(anchor_shot):
            break  # nothing pre-anchor beyond the anchor shot
        first_f = int(s["first_frame"])
        last_f  = int(s["last_frame"])
        limit_last = min(last_f, (anchor_frame - 1)) if anchor_frame > 0 else last_f
        if limit_last < first_f:
            continue

        if has_rows_for_frame:
            # Strong path: examine persisted rows per frame.
            for f in range(first_f, limit_last + 1):
                try:
                    rows = list(oc.rows_for_frame(shot_num, f))  # iterable of dict-ish
                except Exception:
                    logging.exception("RESUME-AUDIT: rows_for_frame failed at shot=%d frame=%d", shot_num, f)
                    continue
                for r in rows:
                    try:
                        det_code = int(SRC_TO_CODE[Source.DETECTED])
                        if int(r.get("src", -1)) != det_code:
                            continue
                        tid = int(r.get("track_id", -1))
                        emb_idx = int(r.get("emb_idx", -1))
                        if emb_idx < 0:
                            _record_missing(
                                shot_num,
                                tid,
                                f,
                                has_landmarks=r.get("has_landmarks", 0),
                                det_by_key=det_by_key,
                            )
                            total_missing += 1
                    except Exception:
                        # tolerate malformed rows
                        continue
        elif has_iter_track_frames:
            # We can at least detect DET frames per (shot, tid) and infer that missing embs likely exist.
            # We won’t know landmarks on this path unless rows_for_frame exists, so mark has_landmarks=0.
            # This is a coarse safety net.
            # Iterate known tids by scanning iter_track_frames for small tid range until it yields nothing.
            # If your collector exposes a way to list tids per shot, prefer that.
            seen_any = False
            for tid_guess in range(0, 2048):  # generous upper bound; adjust if you know a tighter cap
                try:
                    frames_src = list(oc.iter_track_frames(shot_num, tid_guess))
                except Exception:
                    frames_src = []
                if not frames_src:
                    # Heuristic break: after a few consecutive empty tids, stop probing.
                    # (Assumes dense, small tid space per shot.)
                    if tid_guess > 32:
                        break
                    continue
                seen_any = True
                for f, src_code in frames_src:
                    try:
                        f = int(f)
                        if f > limit_last:
                            continue
                        if int(src_code) != int(Source.DETECTED.value):
                            continue
                        # We don't know emb_idx here, but strict parity expects one vector per DET.
                        # Flag as "missing/unknown"; later rows_for_frame (if implemented) will be authoritative.
                        _record_missing(
                            shot_num,
                            int(tid_guess),
                            f,
                            has_landmarks=None,   # unknown on this path
                            det_by_key=det_by_key,
                        )
                        total_missing += 1
                    except Exception:
                        continue
            if not seen_any:
                logging.debug("RESUME-AUDIT: no tracks observed via iter_track_frames for shot=%d", shot_num)

    # Summarize
    if not det_by_key:
        logging.info("RESUME-AUDIT: no pre-anchor DET rows to audit (or none missing).")
        return

    for (shot, tid), misses in sorted(det_by_key.items()):
        misses.sort(key=lambda r: int(r["f"]))
        frames_str = ",".join(f'{m["f"]}({m["has_landmarks"]})' for m in misses)
        logging.error(
            "RESUME-AUDIT shot=%d tid=%d missing_preanchor=%d frames=[%s]  (paren=has_landmarks)",
            shot, tid, len(misses), frames_str
        )

    if total_missing == 0:
        logging.info("RESUME-AUDIT OK: all pre-anchor DET rows have embeddings.")
    else:
        logging.error("RESUME-AUDIT FAIL: total missing pre-anchor DET embeddings=%d", total_missing)

def _build_emb_lookups_for_checkpoint(
    checkpoint: TrackingCheckpoint | None,
    *,
    anchor_frame: int,
):
    """
    Build emb_lookup/emb_array_lookup callables for resume_rehydrate.rehydrate_tracks.

    emb_lookup(shot, tid) returns (frames, embs) for DET observations strictly
    before the anchor frame, using the checkpoint's sidecar accessors:
      - get_det_frames_for_track(shot, tid, frame_max=anchor-1)
      - get_embeddings_by_frames(shot, frames)
    """
    if checkpoint is None or anchor_frame <= 0:
        return None, None

    def emb_lookup(shot: int, tid: int):
        try:
            frames = checkpoint.get_det_frames_for_track(
                int(shot), int(tid), frame_max=int(anchor_frame - 1)
            )
        except Exception:
            return None

        if not frames:
            return None

        try:
            embs = checkpoint.get_embeddings_by_frames(int(shot), frames)
        except Exception:
            return None

        if embs is None or len(embs) != len(frames):
            # Let resume_rehydrate.attach_embeddings enforce strict parity.
            return None

        return frames, embs

    # We intentionally disable emb_array_lookup to avoid any "count only" matching.
    return emb_lookup, None

def _validate_shots_are_absolute_and_increasing(shots):
    if not shots:
        return
    # Absolute: shot 0 must start at 0 (or ≥ 0) and later shots must NOT restart at 0
    later_all_zero = len(shots) > 1 and all(s.get("first_frame", None) == 0 for s in shots[1:])
    # Strictly increasing windows
    strictly_increasing = all(
        int(shots[i]["first_frame"]) > int(shots[i-1]["last_frame"])
        for i in range(1, len(shots))
    )

    if later_all_zero or not strictly_increasing:
        # Build a short diagnostic of the first few shots
        diag = ", ".join(
            f'#{i}[{int(s["first_frame"])},{int(s["last_frame"])}]'
            for i, s in enumerate(shots[:5])
        )
        raise ResumeSafetyError(
            "Shots must be in ABSOLUTE frame space and strictly increasing. "
            "Detected per-shot relative indexing and/or non-monotone ranges. "
            f"Example windows: {diag}. "
            "Ensure the shot detector writes global first_frame/last_frame (no reset to 0 between shots)."
        )
    