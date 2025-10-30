from pathlib import Path
import json
from typing import List, Tuple, Dict, Union, Optional, Iterable
import numpy as np
from contextlib import ExitStack
from bisect import bisect_left
import logging

from facekit.tracking.aggregator import ShotFaceTrackAggregator
from facekit.tracking.face_structures import FaceTrack, FaceObservation
from facekit.tracking.face_tracker import FaceTracker
from facekit.tracking.tracker_validator import TrackerValidator, ValidatorParams
from facekit.embedding.embedder import FaceEmbedder
from facekit.embedding.alignment import align_face_for_arcface
from facekit.detection.face_detector import FaceDetector
from facekit.io.frame_provider import FrameProvider, ReaderCoordinator
from facekit.pipeline.checkpoint import TrackingCheckpoint
from facekit.common.obs_consts import Source
from facekit.pipeline.resume_rehydrate import rehydrate_tracks
from facekit.utils.geometry import compute_iou
from facekit.errors import ResumeSafetyError

# --- Logging helpers (diagnostics only) ---------------------------------------

def _last_det_bbox_and_frame(tr: FaceTrack) -> tuple[Optional[tuple[int,int,int,int]], int]:
    last_det_frame = -1
    last_det_bbox = None
    for o in getattr(tr, "observations", []) or []:
        if getattr(o, "source", None) == Source.DETECTED and o.bbox is not None:
            if o.frame_idx > last_det_frame:
                last_det_frame = int(o.frame_idx)
                last_det_bbox = tuple(int(v) for v in o.bbox[:4])
    return last_det_bbox, last_det_frame

def _track_summary(tr: FaceTrack) -> dict:
    dets = [o for o in (tr.observations or []) if getattr(o, "source", None) == Source.DETECTED]
    trks = [o for o in (tr.observations or []) if getattr(o, "source", None) == Source.TRACKED]
    last_det_bbox, last_det_frame = _last_det_bbox_and_frame(tr)
    return {
        "shot_id":        int(getattr(tr, "shot_id", -1)),
        "track_id":       int(getattr(tr, "track_id", -1)),
        "segment_id":     int(getattr(tr, "segment_id", -1)) if getattr(tr, "segment_id", None) is not None else None,
        "open":           (not tr.is_closed()),
        "frames":         tr.get_frame_indices() if hasattr(tr, "get_frame_indices") else [],
        "first_frame":    tr.first_frame() if hasattr(tr, "first_frame") else None,
        "last_frame":     tr.last_frame() if hasattr(tr, "last_frame") else None,
        "len_obs":        len(getattr(tr, "observations", []) or []),
        "len_det":        len(dets),
        "len_trk":        len(trks),
        "has_embedding":  bool(getattr(tr, "has_embedding", lambda: False)()),
        "avg_emb_norm":   float(np.linalg.norm(tr.compute_average_embedding())) if getattr(tr, "has_embedding", lambda: False)() else None,
        "last_det_bbox":  last_det_bbox,
        "last_det_frame": last_det_frame,
    }

def _log_tracks_state(msg: str, tracks: Iterable[FaceTrack], level: int=logging.INFO) -> None:
    logging.log(level, "%s (count=%d)", msg, sum(1 for _ in tracks))
    for tr in tracks:
        logging.log(level, "  track: %s", _track_summary(tr))


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
    last_frames = [s["last_frame"] for s in shots]
    return bisect_left(last_frames, abs_frame_idx)

def _assign_segment_ids_for_rehydrated(prior_tracks, track_order: dict[tuple[int,int], int]) -> dict[int, int]:
    """
    Deterministically assign segment_id to already rehydrated tracks per shot using persisted track_order.
    Returns: {shot_number: next_segment_seed} where the seed equals the count of assigned segments in that shot.
    """
    by_shot: dict[int, list] = {}
    for t in prior_tracks or []:
        s = int(getattr(t, "shot_id", 0))
        by_shot.setdefault(s, []).append(t)

    seeds: dict[int, int] = {}
    for s, tracks in by_shot.items():
        # Sort tracks by the persisted first-seen order
        tracks.sort(key=lambda tr: track_order.get((s, int(getattr(tr, "track_id", -1))), 1 << 30))
        # Assign deterministic segment ids
        for idx, tr in enumerate(tracks):
            setattr(tr, "segment_id", idx)
        seeds[s] = len(tracks)
    return seeds

def track_across_segments(
    frame_source: Union[str, Path, FrameProvider],
    shot_json_path: str,
    detector: FaceDetector,
    embedder: FaceEmbedder,
    iou_thresh: float = 0.5,
    embedding_thresh: float = 0.7,
    detect_interval: int = 10,
    embedding_batch_size_max: int = 32,
    *,
    checkpoint: TrackingCheckpoint | None = None,
    resume_enabled: bool = True,
) -> List[FaceTrack]:
    """
    Track faces across precomputed shot segments, attach embeddings, and return per-shot tracks.

    This function is the main, **shot-aware** tracker. For each shot [first_frame..last_frame]:
      1) **Streams frames sequentially** via a `FrameProvider` (`ReaderCoordinator` by default).
         We jump directly to the desired starting frame with `reset_to_frame(start_at)` and
         then consume frames with `next()`.
      2) Performs **periodic face detection** every `detect_interval` frames (and always on the first frame
         of the shot or after tracker resets). Detection observations are handed to the per-shot
         `ShotFaceTrackAggregator`, which assigns/continues **shot-local segment IDs**.
      3) Between detection frames, it uses a `FaceTracker` (CSRT by default) to propagate existing boxes
         and emits **tracking** observations for currently active tracks.
      4) If any tracker update fails, all open tracks are marked closed, and the pipeline **falls back to detection**
         on that frame to reboot tracking cleanly.
      5) At shot end, batches all aligned detection crops through the `embedder` and attaches 512-D embeddings
         to each `FaceTrack`. The aggregator is then finalized, and per-shot **segment IDs** are resolved.

    Parameters
    ----------
    frame_source : Union[str, Path, FrameProvider]
        Either:
          - a filesystem path to a video file (str or Path), in which case a `ReaderCoordinator`
            will be constructed and managed internally, OR
          - a preconstructed `FrameProvider` instance, in which case this function will
            use it but **not close it** (caller retains ownership).
    shot_json_path : str
        Path to a JSON produced by shot segmentation, containing
        `{"shots": [{"shot_number": int, "first_frame": int, "last_frame": int}, ...]}`.
    detector : FaceDetector
        Face detector used on scheduled detection frames. Its `detect_faces_in_frame(frame)` is expected
        to return `(boxes_xyxy, landmarks, confidences)` or `None` when no faces are present.
    embedder : FaceEmbedder
        Embeds aligned detection crops at the end of each shot; must return an `np.ndarray` of shape (K, 512), float32.
    iou_thresh : float, default 0.5
        IOU threshold inside the per-shot `ShotFaceTrackAggregator` for associating detections to tracks.
    embedding_thresh : float, default 0.7
        Cosine similarity threshold used **within the shot** when resolving/merging segment IDs.
        (Global identity resolution happens later, outside this function.)
    detect_interval : int, default 10
        Run the detector every `detect_interval` frames (and on the first frame of a shot or after tracker reset).
    embedding_batch_size_max : int, default 32
        Maximum batch size for `embedder.get_embedding_batch`.
    checkpoint : TrackingCheckpoint | None, optional
        Checkpoint interface used for sidecar persistence (observations/embeddings),
        status.json updates, and rehydration of prior observations/tracks.
    resume_enabled : bool, default True
        When True, this function resolves a resume anchor from the `checkpoint`
        (preferring `get_resume_anchor()`, then `last_detection_frame` from status,
        then the maximum observed frame in the observations collector). The tracker
        will start at `max(anchor, shot_first)` within the containing shot and will
        rehydrate prior observations strictly *before* the anchor. When False,
        the function ignores any resume state and starts from the first shot frame.

    Returns
    -------
    List[FaceTrack]
        All finalized `FaceTrack` objects across all shots, each with:
          - `shot_id`, `track_id` (per-shot), `segment_id` (shot-local face ID),
          - `observations` (detection + tracking),
          - attached 512-D embeddings for detection observations (if any crops existed).

    Notes
    -----
    - **Detector scheduling:** Frames with detections produce aligned crops; tracking-only frames intentionally **do not**
      create embeddings to keep the hot path fast. (A later second-pass can compute extra embeddings if needed.)
    - **Lifecycle:** When a `FrameProvider` is supplied, this function does not close it. When a path is supplied,
      the internally constructed provider is closed automatically via `ExitStack()`.
    - **Performance knobs:** `detect_interval` trades accuracy for speed; increasing it reduces detector calls.
      `embedding_batch_size_max` controls memory/throughput on the embedder.

    Resume semantics & label continuity
    -----------------------------------
    When resuming, we rehydrate prior tracks (up to the resume anchor) from the persisted
    observations sidecar. We then assign stable per-shot `segment_id`s **immediately** to the
    rehydrated tracks using the checkpoint's `track_order` mapping, so labels match the
    pre-interruption numbering. For the active shot (and any subsequent shots), we seed the
    segment-id counter with the number of already-assigned tracks for that shot so new tracks
    continue numbering where the previous run left off (no gaps or renumbering).
    """

    shot_json_path = Path(shot_json_path)
    with open(shot_json_path, "r") as f:
        _shot_data = json.load(f)
    shots = _shot_data["shots"]

    with ExitStack() as stack:
        if isinstance(frame_source, (str, Path)):
            frame_provider = stack.enter_context(ReaderCoordinator(str(frame_source)))
        else:
            # Require the canonical FrameProvider interface:
            # props: fps, size, total_frames; methods: reset_to_frame(i), next()
            has_core = all(hasattr(frame_source, m) for m in ("fps", "size", "total_frames"))
            has_iter = hasattr(frame_source, "reset_to_frame") and hasattr(frame_source, "next")
            _ok = isinstance(frame_source, FrameProvider) or (has_core and has_iter)
            if not _ok:
                raise TypeError(
                    "frame_source must be a str/Path or a FrameProvider-like object with "
                    "fps, size, total_frames, reset_to_frame(i), and next(). "
                    f"Got {type(frame_source)!r}"
                )
            frame_provider = frame_source

        all_tracks: List[FaceTrack] = []
        segment_id_seed_by_shot: Dict[int, int] = {}
        trackid_seed_by_shot: Dict[int, int] = {}
        prior_tracks: List[FaceTrack] = []
        # For exact resume parity: reuse the last pre-anchor track id on the
        # first detection inside the anchor-containing shot.
        reuse_tid_for_first_shot: Optional[int] = None
        reuse_binding_used: bool = False

        # ---------- Single, explicit resume arbiter ----------
        def _resolve_anchor() -> int:
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
                    import os, json
                    if os.path.exists(status_path):
                        with open(status_path, "r") as f:
                            status = json.load(f) or {}
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

        resume_abs_frame: int = _resolve_anchor()

        # Identify anchor-containing shot BEFORE any trimming (used for logging)
        anchor_shot_idx = _shot_idx_by_abs_frame(shots, resume_abs_frame) if resume_abs_frame > 0 else None
        anchor_shot_num = (
            shots[anchor_shot_idx]["shot_number"]
            if (anchor_shot_idx is not None and anchor_shot_idx < len(shots))
            else None
        )

        # Compute completed shots (fully before anchor) BEFORE trimming
        completed_shot_nums = {
            s["shot_number"] for s in shots
        } if resume_abs_frame == 0 else {
            s["shot_number"] for s in shots if s["last_frame"] < resume_abs_frame
        }

        # Rehydrate BEFORE the anchor (if any)
        if checkpoint and resume_abs_frame > 0:
            def _emb_lookup(shot: int, tid: int):
                # Preferred: return (frame_indices, embs)
                if hasattr(checkpoint, "emb_collector") and hasattr(checkpoint.emb_collector, "get_embeddings"):
                    return checkpoint.emb_collector.get_embeddings(shot, tid)  # -> (frames, np.ndarray) or None
                if hasattr(checkpoint, "obs_collector") and hasattr(checkpoint.obs_collector, "get_embeddings"):
                    return checkpoint.obs_collector.get_embeddings(shot, tid)
                return None

            def _emb_array_lookup(shot: int, tid: int):
                # Fallback: return np.ndarray or None
                if hasattr(checkpoint, "get_embeddings"):
                    return checkpoint.get_embeddings(shot, tid)
                if hasattr(checkpoint, "emb_collector") and hasattr(checkpoint.emb_collector, "get_embeddings_array"):
                    return checkpoint.emb_collector.get_embeddings_array(shot, tid)
                return None

            prior_tracks = rehydrate_tracks(
                checkpoint.obs_collector,
                frame_max=resume_abs_frame - 1,
                track_order=(checkpoint.get_track_order() or {}),
                emb_lookup=_emb_lookup,
                emb_array_lookup=_emb_array_lookup,
            )

            _log_tracks_state("resume: rehydrated tracks (pre-anchor)", prior_tracks, level=logging.INFO)

            # Also log by-shot counts & seeds we computed
            try:
                by_shot_counts = {}
                for tr in prior_tracks:
                    s = int(getattr(tr, "shot_id", -1))
                    by_shot_counts[s] = by_shot_counts.get(s, 0) + 1
                logging.info("resume: rehydrated counts by shot: %r", by_shot_counts)
                logging.info("resume: segment_id_seed_by_shot: %r", {int(k): int(v) for k,v in (segment_id_seed_by_shot or {}).items()})
                logging.info("resume: trackid_seed_by_shot: %r", {int(k): int(v) for k,v in (trackid_seed_by_shot or {}).items()})
            except Exception:
                logging.exception("resume: failed summarizing seeds")

            # Assign stable segment_ids to the rehydrated tracks and compute per-shot seeds
            try:
                track_order_map = checkpoint.get_track_order() if hasattr(checkpoint, "get_track_order") else {}
                segment_id_seed_by_shot = _assign_segment_ids_for_rehydrated(prior_tracks, track_order_map or {})
                # Compute next track_id seed per shot = max(track_id)+1 seen in that shot
                tmp: Dict[int, int] = {}
                # Also compute, per shot, the *last* observed track (by last frame) so we
                # can reuse that exact track_id on the first post-resume detection.
                last_tid_by_shot: Dict[int, int] = {}
                last_frame_by_shot_tid: Dict[Tuple[int, int], int] = {}

                for tr in prior_tracks:
                    s = int(getattr(tr, "shot_id", 0))
                    tid = int(getattr(tr, "track_id", -1))
                    if tid >= 0:
                        tmp[s] = max(tmp.get(s, -1), tid)
                        # Track the last frame seen for (shot, tid) and pick the tid with the max last frame.
                        if getattr(tr, "observations", None):
                            lf = max(o.frame_idx for o in tr.observations)
                            key = (s, tid)
                            if key not in last_frame_by_shot_tid or lf > last_frame_by_shot_tid[key]:
                                last_frame_by_shot_tid[key] = lf
                trackid_seed_by_shot = {s: (mx + 1) for s, mx in tmp.items()}
                # Reduce last_frame_by_shot_tid -> last_tid_by_shot (argmax over tid per shot)
                for (s, tid), lf in last_frame_by_shot_tid.items():
                    if (s not in last_tid_by_shot) or lf > last_frame_by_shot_tid.get((s, last_tid_by_shot[s]), -1):
                        last_tid_by_shot[s] = tid                
                logging.info("resume: assigned segment_ids to %d rehydrated tracks; seeds=%s",
                             len(prior_tracks), {k: int(v) for k, v in segment_id_seed_by_shot.items()})
            except Exception:
                logging.exception("resume: failed to assign segment_ids to rehydrated tracks; continuing with empty seeds")
                segment_id_seed_by_shot = {}
                trackid_seed_by_shot = {}

            # --- Resume integrity checks (for shots before anchor_shot_frame_idx embeddings must be finite) ---
            missing_by_shot: dict[int, int] = {}
            for tr in prior_tracks or []:
                shot_num = int(getattr(tr, "shot_id", -1))
                if shot_num in completed_shot_nums: 
                    for obs in getattr(tr, "observations", []) or []:
                        if obs.source == Source.DETECTED:
                            ok = (obs.embedding is not None) and np.isfinite(obs.embedding).all()
                            if not ok:
                                missing_by_shot[shot_num] = missing_by_shot.get(shot_num, 0) + 1
            if missing_by_shot:
                raise ResumeSafetyError(
                    f"rehydrate: missing DET embeddings in completed shots before anchor_shot_num: {missing_by_shot}"
                )
                
            logging.info("resume: prior_tracks strictness OK (anchor_shot_num=%s); %d tracks rehydrated",
                anchor_shot_num, len(prior_tracks or []))

            all_tracks.extend(prior_tracks)
        else:
            segment_id_seed_by_shot = {}
            trackid_seed_by_shot = {}

        # Trim shots so the first processed shot is the one containing the anchor (always)
        start_shot_idx = _shot_idx_by_abs_frame(shots, resume_abs_frame)
        if start_shot_idx >= len(shots):
            raise ResumeSafetyError("resume anchor beyond last shot; aborting for safety.")
        shots = shots[start_shot_idx:]

        # Track whether this invocation is a true resume (anchor > shot_first)
        _is_resume = bool(resume_abs_frame > 0)

        # Determine which shot contains the anchor and pre-compute its reuse tid.
        first_processed_shot_number = shots[0]["shot_number"]
        if checkpoint and resume_abs_frame > 0:
            try:
                # Prefer the tid whose last observation is closest to the anchor (already computed above).
                reuse_tid_for_first_shot = last_tid_by_shot.get(int(first_processed_shot_number))  # type: ignore[name-defined]
                if reuse_tid_for_first_shot is not None:
                    logging.info("resume: will reuse tid=%d for first detection in shot=%d",
                                 int(reuse_tid_for_first_shot), int(first_processed_shot_number))
            except Exception:
                reuse_tid_for_first_shot = None

            logging.info(
                "resume: anchor=%d anchor_shot_num=%s first_processed_shot_number=%s reuse_tid_for_first_shot=%s",
                resume_abs_frame, anchor_shot_num, first_processed_shot_number, reuse_tid_for_first_shot
            )

        for shot_idx, shot_num in enumerate(shots):
            shot_number = shot_num["shot_number"]
            first = shot_num["first_frame"]
            last = shot_num["last_frame"]
            # If resuming, start from the anchor frame inside the first shot; otherwise from shot start.
 
            if shot_idx == 0:
                start_at = max(first, resume_abs_frame)
            else:
                start_at = first

            if shot_idx == 0:
                logging.info("resume: first_new_frame=%d (shot=[%d..%d]) detect_interval=%d mod=%d",
                    start_at, first, last, detect_interval, (start_at % detect_interval))

            # Determine seeded prior tracks only when resuming and 
            # for the first processed shot (anchor-containing shot)
            seeded_tracks = None
            if (shot_idx == 0) and (resume_abs_frame > 0) and prior_tracks:
                seeded_tracks = [
                    t for t in prior_tracks
                    if int(getattr(t, "shot_id", -1)) == int(shot_number)
                ]

            # Readable two-branch constructor
            if seeded_tracks:
                aggregator = ShotFaceTrackAggregator(
                    shot_number=shot_number,
                    iou_threshold=iou_thresh,
                    embedding_threshold=embedding_thresh,
                    prior_tracks=seeded_tracks,
                    resume_abs_frame=start_at,
                    next_tid_seed=int(trackid_seed_by_shot.get(int(shot_number), 0)),
                )
            else:
                aggregator = ShotFaceTrackAggregator(
                    shot_number=shot_number,
                    iou_threshold=iou_thresh,
                    embedding_threshold=embedding_thresh,
                )

            if seeded_tracks:
                logging.info("RESUME: aggregator seeded with %d tracks", len(aggregator.tracks))
                for tr in aggregator.tracks:
                    last_bbox = getattr(tr, "get_last_bbox", lambda: None)()
                    logging.info(
                        "RESUME TRACK: tid=%s closed=%s last_frame=%s last_bbox=%s",
                        tr.track_id,
                        tr.is_closed() if hasattr(tr, "is_closed") else None,
                        tr.last_frame() if hasattr(tr, "last_frame") else None,
                        last_bbox
        )

            seed_tid = int(trackid_seed_by_shot.get(int(shot_number), 0))
            seg_seed = int(segment_id_seed_by_shot.get(int(shot_number), 0)) if segment_id_seed_by_shot else 0

            logging.info(
                "shot=%d init: aggregator.next_track_id=%d, seed_tid=%d, seg_seed=%d",
                int(shot_number),
                int(getattr(aggregator, "next_track_id", -1)),
                seed_tid,
                seg_seed,
            )

            # If this is the first processed shot, dump prior_tracks *for that shot* so we know the reference
            if shot_idx == 0:
                prior_for_shot = [tr for tr in prior_tracks if int(getattr(tr, "shot_id", -1)) == int(shot_number)]
                _log_tracks_state(f"shot={shot_number}: prior_tracks for this shot (pre-anchor)", prior_for_shot, level=logging.INFO)

            # With warmstart, aggregator may already contain seeded tracks
            logging.info("shot=%d init: aggregator has %d tracks (0 when cold, >0 when warmstart)",
                        int(shot_number), len(aggregator.tracks))

            # Seed the aggregator’s next track_id only if we did not already pass next_tid_seed.
            if not seeded_tracks and seed_tid > 0:
                if hasattr(aggregator, "set_track_id_seed") and callable(getattr(aggregator, "set_track_id_seed")):
                    aggregator.set_track_id_seed(seed_tid)
                elif hasattr(aggregator, "next_track_id"):
                    setattr(aggregator, "next_track_id", seed_tid)
                elif hasattr(aggregator, "_next_track_id"):
                    setattr(aggregator, "_next_track_id", seed_tid)

            face_tracker = FaceTracker(tracker_type="CSRT")
            tracker_active = False

            shot_frames: List[np.ndarray] = []
            validator = TrackerValidator(
                frames=shot_frames, 
                first_frame_idx=start_at, 
                params=ValidatorParams(iou_thresh=iou_thresh))

            frame_provider.reset_to_frame(start_at)

            for frame_idx in range(start_at, last + 1):
                frame = frame_provider.next()
                if frame is None:
                    break

                # Guardrail: never emit pre-anchor frames
                if resume_abs_frame and frame_idx < resume_abs_frame:
                    raise ResumeSafetyError(f"Illegal rewind: got frame_idx={frame_idx} < anchor={resume_abs_frame}")

                shot_frames.append(frame)
                observations: List[FaceObservation] = []

                detect_interval_hit = (frame_idx % detect_interval == 0)

                no_tracker   = (not tracker_active)
                open_tracks = [t for t in aggregator.tracks if not t.is_closed()]
                no_open_tracks = (len(open_tracks) == 0)
   
                # Determine if this is a scheduled detection frame
                need_detect = detect_interval_hit or no_tracker or no_open_tracks

                logging.debug(
                    "frame=%d need_detect=%s [det_interval_hit=%s no_tracker=%s no_open_tracks=%s] open_tracks=%d total_tracks=%d",
                    frame_idx, need_detect, detect_interval_hit, no_tracker, no_open_tracks, len(open_tracks),
                    len(aggregator.tracks),
                )

                if not need_detect:
                    # Try tracking all existing tracks
                    tracked_boxes: Dict[int, Tuple[float, float, float, float]] = face_tracker.update_trackers(frame)
                    basic_fail = (not tracked_boxes) or any(b is None for b in tracked_boxes.values())

                    logging.debug("TRACK: frame=%d (tracker_active=%s, basic_fail=%s)", frame_idx, tracker_active, basic_fail)

                    if (not basic_fail) and validator.validate(tracked_boxes, frame_idx, verbose=True):
                        # Track-only observations will lack aligned_face until optical flow landmarks are added
                        for track_id, tb in tracked_boxes.items():
                            if tb is None:
                                continue
                            x, y, w, h = tb
                            bbox = (int(x), int(y), int(x + w), int(y + h))
                            observations.append(FaceObservation(
                                frame_idx=frame_idx,
                                track_id = track_id,
                                bbox=bbox,
                                embedding=None,
                                confidence=None,
                                aligned_face=None,
                                source=Source.TRACKED
                            ))
                    else:
                        # validator rejected or tracker failed → close & force detection reboot
                        logging.debug("TRACK - validator reject or tracker fail: frame=%d (basic_fail=%s)", frame_idx, basic_fail)
                        aggregator.finalize_tracks()
                        tracker_active = False
                        need_detect = True                       

                if need_detect:  # Run detection if needed
                    logging.info(
                        "shot=%d first=%d frame=%d (mod=%d) tracker_active=%s, detect_interval=%d",
                        shot_number, first, frame_idx, (frame_idx % detect_interval), tracker_active, detect_interval
                    )
                    if checkpoint:
                        checkpoint.checkpoint_now(
                            frame_idx=frame_idx, 
                            shot_number=shot_number,
                            shot_first_frame=first)
                    
                    detections = detector.detect_faces_in_frame(frame)
                    
                    if detections:
                        boxes, landmark_lists, confidences = detections
                        for box, landmarks, confidence in zip(boxes, landmark_lists, confidences):
                            bbox = tuple(int(v) for v in box[:4])  # detector returns XYXY
                            aligned_face = align_face_for_arcface(frame, landmarks, frame_idx, source="detect")
                            observations.append(FaceObservation(
                                frame_idx=frame_idx,
                                bbox=bbox,
                                embedding=None,
                                confidence=float(confidence),
                                aligned_face=aligned_face,
                                source=Source.DETECTED
                            ))
                    else:
                        # no faces this frame; close shot-local tracks
                        aggregator.finalize_tracks()          

                # If this is the anchor-containing shot, and we haven't yet bound the
                # first post-resume detection, temporarily override the allocator so the
                # very first created track reuses the pre-anchor tid (exact parity).
                if (
                    need_detect
                    and not reuse_binding_used
                    and reuse_tid_for_first_shot is not None
                    and int(shot_number) == int(first_processed_shot_number)
                    and observations  # only if we actually detected faces
                ):
                    try:
                        # Force the next allocated id to be the reused tid.
                        if hasattr(aggregator, "set_track_id_seed") and callable(getattr(aggregator, "set_track_id_seed")):
                            aggregator.set_track_id_seed(int(reuse_tid_for_first_shot))
                        elif hasattr(aggregator, "next_track_id"):
                            setattr(aggregator, "next_track_id", int(reuse_tid_for_first_shot))
                        elif hasattr(aggregator, "_next_track_id"):
                            setattr(aggregator, "_next_track_id", int(reuse_tid_for_first_shot))
                        logging.info("resume: temporarily overriding next_track_id -> %d (shot=%d)",
                                     int(reuse_tid_for_first_shot), int(shot_number))
                    except Exception:
                        logging.exception("resume: failed to set temporary allocator for reuse tid; continuing without binding")

                if observations and observations[0].source == Source.DETECTED:
                    # Expand detection list for logging
                    dets = []
                    for ob in observations:
                        x1, y1, x2, y2 = [int(v) for v in ob.bbox[:4]]
                        dets.append({
                            "bbox": (x1, y1, x2, y2),
                            "conf": float(ob.confidence) if ob.confidence is not None else None
                        })

                    logging.info(
                        "DETECTION frame=%d shot=%d count=%d dets=%s",
                        frame_idx, int(shot_number), len(dets), dets
                    )

                    # For each detection, compute IoU vs every current aggregator track and log the best
                    for idx, ob in enumerate(observations):
                        best_tid = None
                        best_iou = -1.0
                        best_last_bbox = None
                        reasons = []
                        for tr in aggregator.tracks:
                            if tr.is_closed():
                                reasons.append(f"skip track_id={tr.track_id} (closed)")
                                continue
                            last_bbox = tr.get_last_bbox()
                            if last_bbox is None:
                                reasons.append(f"skip track_id={tr.track_id} (no last_bbox)")
                                continue
                            iou = float(compute_iou(last_bbox, ob.bbox))
                            if iou > best_iou:
                                best_iou = iou
                                best_tid = tr.track_id
                                best_last_bbox = last_bbox

                        # Decision log relative to threshold
                        iou_thresh_eff = float(getattr(aggregator, "iou_threshold", 0.5))
                        can_bind = best_iou >= iou_thresh_eff if best_iou >= 0 else False

                        logging.info(
                            "MATCH frame=%d det_idx=%d det_bbox=%s best_tid=%s best_iou=%.4f (thresh=%.3f) can_bind=%s best_last_bbox=%s",
                            frame_idx, idx, tuple(int(v) for v in ob.bbox[:4]),
                            (int(best_tid) if best_tid is not None else None),
                            best_iou, iou_thresh_eff, bool(can_bind),
                            (tuple(int(v) for v in best_last_bbox) if best_last_bbox else None)
                        )

                        if reasons:
                            logging.debug("  skipped tracks: %s", reasons)

                    # Snapshot aggregator state *before* the assignment happens
                    _log_tracks_state(
                        f"PRE-ASSIGN state at frame={frame_idx} shot={shot_number}",
                        aggregator.tracks, level=logging.INFO
                    )

                # Add current frame observations to aggregator
                created_count = aggregator.update_tracks_with_frame(frame_idx, observations)
 
                if _is_resume and shot_idx == 0 and need_detect:
                    # On the very first DET frame after resume, verify whether we extended or created.
                    if created_count == 0:
                        logging.info("RESUME OK: first DET after anchor extended an existing track (no new track created).")
                    else:
                        logging.warning("RESUME NOTE: first DET after anchor created %d new track(s). "
                                        "This is expected only if IoU<threshold or geometry changed.", created_count)

                # POST-ASSIGN diagnostics (created_count now defined)
                logging.info(
                    "POST-ASSIGN frame=%d shot=%d created=%d open_now=%d total=%d",
                    frame_idx, int(shot_number), int(created_count),
                    sum(1 for t in aggregator.tracks if not t.is_closed()),
                    len(aggregator.tracks)
                )

                # Log exactly which tracks received a DET observation at this frame
                agg_det_now = aggregator.observations_at(frame_idx, source=Source.DETECTED, require_track_id=True)
                if agg_det_now:
                    logging.info(
                        "ASSIGNED-DETS frame=%d: %s",
                        frame_idx,
                        [{"tid": int(o.track_id), "bbox": tuple(int(v) for v in o.bbox[:4])} for o in agg_det_now]
                    )
                else:
                    logging.info("ASSIGNED-DETS frame=%d: none", frame_idx)


                # Mark binding as used exactly once on the first detection frame where it applied.
                if (
                    need_detect
                    and not reuse_binding_used
                    and reuse_tid_for_first_shot is not None
                    and int(shot_number) == int(first_processed_shot_number)
                    and created_count > 0
                ):
                    reuse_binding_used = True
                    logging.info("resume: reuse binding applied (shot=%d, reused tid=%d).",
                                 int(shot_number), int(reuse_tid_for_first_shot))

                agg_det = aggregator.observations_at(frame_idx, source=Source.DETECTED, require_track_id=True)
                agg_trk = aggregator.observations_at(frame_idx, source=Source.TRACKED,  require_track_id=True)
                logging.debug(
                    "POST-ASSIGN: frame=%d det_assigned=%d trk_assigned=%d created=%d open_tracks=%d",
                    frame_idx, len(agg_det), len(agg_trk), created_count,
                    sum(1 for t in aggregator.tracks if not t.is_closed()),
                )

                # Checkpoint observations
                if checkpoint:
                    frame_obs = aggregator.observations_at(frame_idx, require_track_id=True)
                    if frame_obs:
                        checkpoint.add_observations(shot_number, frame_idx, frame_obs)
                    if created_count:
                        checkpoint.on_new_tracks(created_count)

                # If faces detected, init tracker with aggregator IDs and seed validator
                if need_detect:
                    det_obs = aggregator.observations_at(frame_idx, source=Source.DETECTED, require_track_id=True)
                    if det_obs:
                        boxes_xywh = [
                            (b[0], b[1], b[2] - b[0], b[3] - b[1]) 
                            for b in (obs.bbox for obs in det_obs)
                        ]
                        track_ids = [obs.track_id for obs in det_obs]
                        face_tracker.init_trackers(frame, boxes_xywh, track_ids)

                        # --- seed validator baseline with the just-initialized tracker state ---
                        boxes_map = dict(zip(track_ids, boxes_xywh))
                        validator.set_baseline(boxes_map, frame_idx)
            
                        tracker_active = True
                    else:
                        tracker_active = False

                if checkpoint:
                    checkpoint.on_frame(frame_idx)

            # End of shot: batch embed tracks with aligned faces
            for track in aggregator.tracks:
                crops = [obs.aligned_face for obs in track.observations if obs.aligned_face is not None]
                if not crops:
                    continue
                embs = embedder.get_embedding_batch(crops, batch_size=embedding_batch_size_max)
                if not isinstance(embs, np.ndarray):
                    raise TypeError(f"Embedder must return np.ndarray, got {type(embs)}")
                if embs.ndim != 2 or embs.shape[1] != 512:
                    raise ValueError(f"Embedder returned invalid array shape {embs.shape}; expected (K,512)")
                if embs.dtype != np.float32:
                    embs = np.asarray(embs, dtype=np.float32, order="C")
                aggregator.attach_embeddings(track.track_id, embs)

                if checkpoint and embs.size:
                    last_idx = max(o.frame_idx for o in track.observations if o.aligned_face is not None)
                    checkpoint.add_embeddings(shot_number, track.track_id, last_idx, embs)

            aggregator.finalize_tracks()

            # Assign segment_id per shot, seeded to continue numbering after any rehydrated tracks
            seed = int(segment_id_seed_by_shot.get(int(shot_number), 0))
            try:
                _ = aggregator.resolve_segment_ids(
                    segment_id_counter=seed,
                    embedding_threshold=embedding_thresh
                )
            except RuntimeError as e:
                logging.error("segment-id resolution skipped for shot=%d due to missing embeddings: %s",
                              int(shot_number), e)

            all_tracks.extend(aggregator.tracks)

            ####################-------------######################
            # Safety: this check only applies on true resume (anchor > 0)
            if _is_resume and shot_idx == 0:
                # Build a consolidated key list only for the anchor-containing shot,
                # legacy (prior_tracks) must come entirely before the new shot part.
                def _keys_for_shot(tracks, shot_num):
                    out = []
                    for tr in tracks or []:
                        if int(getattr(tr, "shot_id", -1)) != int(shot_num):
                            continue
                        for o in getattr(tr, "observations", []) or []:
                            out.append((int(o.frame_idx), int(tr.track_id)))
                    return sorted(out)

                legacy_keys = _keys_for_shot(prior_tracks, shot_number)
                new_keys    = _keys_for_shot(aggregator.tracks, shot_number)
                if legacy_keys and new_keys and legacy_keys[-1][0] >= new_keys[0][0]:
                    raise ResumeSafetyError(f"non-monotone concat at shot={shot_number}: "
                                            f"legacy_last={legacy_keys[-1][0]} new_first={new_keys[0][0]}")
                logging.info("resume: legacy/new concat OK for shot=%d (legacy_last=%s new_first=%s)",
                            shot_number,
                            legacy_keys[-1][0] if legacy_keys else None,
                            new_keys[0][0] if new_keys else None)
            ######################--------------###################

            if checkpoint:
                checkpoint.on_shot_done()

    return all_tracks
