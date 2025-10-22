from typing import Optional, List, Tuple, Callable, Iterable, Dict
import numpy as np
from math import isfinite
from pathlib import Path
import logging
from facekit.tracking.face_structures import FaceTrack, FaceObservation
from facekit.common.obs_consts import Source, code_to_src, SRC_TO_CODE
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

def _row_to_faceobs(row: dict, track_id: int) -> FaceObservation | None:
    bb = _as_int_bbox(row["bbox_xyxy"])
    if bb is None:
        return None
    x1, y1, x2, y2 = bb

    raw_src = row.get("src")
    try:
        # Sidecars may hold enum, int code, numeric str, or legacy name ("detected")
        src = _normalize_src(raw_src)
    except Exception as e:
        raise ResumeSafetyError(
            f"rehydrate: invalid src={raw_src!r} in obs row "
            f"(shot={row.get('shot')}, f={row.get('f')}, tid={track_id})"
        ) from e

    obs = FaceObservation(
        frame_idx=int(row["f"]),
        track_id=int(track_id) if track_id is not None and int(track_id) >= 0 else None,
        bbox=(x1, y1, x2, y2),
        embedding=None,
        confidence=(
            float(row["conf"])
            if "conf" in row and row["conf"] is not None and not np.isnan(row["conf"])
            else None
        ),
        aligned_face=None,
        source=src,
    )

    crop_ref = row.get("crop_ref") or row.get("crop_path") or ""
    if crop_ref:
        setattr(obs, "crop_ref", str(crop_ref))

    try:
        logging.info(
            "rehydrate DEBUG FaceObservation: shot=%r track_id=%r frame=%r src=%r crop_ref=%r",
            row.get("shot"),
            track_id,
            obs.frame_idx,
            obs.source,
            getattr(obs, "crop_ref", None),
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

        if row_src is not det_src:
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

        # DEBUG: dump raw rows for the specific test case (shot=2, tid=1)
        try:
            if int(shot) == 2 and int(track_id) == 1:
                for r in rows:
                    # r is expected to be a dict-like object
                    try:
                        logging.info(
                            "rehydrate DEBUG raw row: "
                            "shot=%r tid=%r f=%r src=%r crop_ref=%r has_crop=%r",
                            r.get("shot"),
                            r.get("track_id", r.get("tid")),
                            r.get("f"),
                            r.get("src"),
                            r.get("crop_ref", None),
                            r.get("has_crop", None),
                        )
                    except AttributeError:
                        # If r is a numpy record instead of dict
                        logging.info(
                            "rehydrate DEBUG raw row (non-dict): type=%s fields=%s",
                            type(r),
                            getattr(r, "dtype", None),
                        )
        except Exception:
            logging.exception("rehydrate DEBUG: logging raw rows failed")

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
